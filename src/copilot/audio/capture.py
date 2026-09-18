"""Windows system-audio (WASAPI loopback) capture.

`LoopbackCapture` opens the loopback endpoint of a render device - i.e. it
records what Windows is *playing*, which is the interviewer voice coming out of
Teams/Zoom/Meet - converts it to mono PCM16 at the configured rate and hands
fixed-size chunks to a callback.

The PortAudio callback runs on a realtime audio thread, so it does only cheap
numpy work and never touches Qt or the network.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .processing import StreamingResampler, downmix_to_mono, float_to_pcm16, rms_level

log = logging.getLogger(__name__)

ChunkCallback = Callable[[bytes, float], None]
"""Called as callback(pcm16_mono_bytes, rms_level) from the audio thread."""


class AudioCaptureError(RuntimeError):
    """Raised when a loopback device cannot be opened or disappears."""


@dataclass(frozen=True)
class LoopbackDevice:
    index: int
    name: str
    channels: int
    sample_rate: int
    is_default: bool = False

    @property
    def label(self) -> str:
        suffix = "  (default)" if self.is_default else ""
        return f"{self.name} - {self.sample_rate} Hz, {self.channels}ch{suffix}"


def _pyaudio():
    """Import PyAudioWPatch, with a clear message when it is unavailable."""
    try:
        import pyaudiowpatch
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise AudioCaptureError(
            "PyAudioWPatch is not installed. System-audio capture requires Windows "
            "with WASAPI loopback support (pip install PyAudioWPatch)."
        ) from exc
    return pyaudiowpatch


def list_loopback_devices() -> list[LoopbackDevice]:
    """Every WASAPI loopback endpoint, default render device first."""
    pa_module = _pyaudio()
    devices: list[LoopbackDevice] = []
    audio = pa_module.PyAudio()
    try:
        try:
            wasapi = audio.get_host_api_info_by_type(pa_module.paWASAPI)
            default_out = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
            default_name = str(default_out.get("name", ""))
        except Exception:  # pragma: no cover - host api missing
            default_name = ""

        for info in audio.get_loopback_device_info_generator():
            name = str(info.get("name", "unknown"))
            devices.append(
                LoopbackDevice(
                    index=int(info["index"]),
                    name=name,
                    channels=int(info.get("maxInputChannels") or 2),
                    sample_rate=int(info.get("defaultSampleRate") or 48000),
                    is_default=bool(default_name) and default_name in name,
                )
            )
    finally:
        audio.terminate()
    devices.sort(key=lambda d: (not d.is_default, d.name))
    return devices


def default_loopback_device() -> LoopbackDevice:
    devices = list_loopback_devices()
    if not devices:
        raise AudioCaptureError(
            "No WASAPI loopback device found. Make sure an audio output device is "
            "enabled in Windows sound settings."
        )
    return devices[0]


class LoopbackCapture:
    """Streams mono PCM16 chunks from a Windows loopback endpoint."""

    def __init__(
        self,
        on_chunk: ChunkCallback,
        device_index: int | None = None,
        target_sample_rate: int = 16000,
        chunk_ms: int = 100,
        gain: float = 1.0,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._device_index = device_index
        self._target_rate = target_sample_rate
        self._chunk_ms = max(20, chunk_ms)
        self._gain = gain

        self._audio = None
        self._stream = None
        self._resampler: StreamingResampler | None = None
        self._pending = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._device: LoopbackDevice | None = None
        self._channels = 2
        self._samples_per_chunk = max(1, int(target_sample_rate * self._chunk_ms / 1000))
        self._silence = b"\x00" * (self._samples_per_chunk * 2)
        self._last_emit = 0.0
        self._keepalive: threading.Thread | None = None
        self._stop_keepalive = threading.Event()
        self.silence_chunks_emitted = 0

    # -- lifecycle --------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def device(self) -> LoopbackDevice | None:
        return self._device

    def start(self) -> LoopbackDevice:
        if self._running:
            return self._device  # type: ignore[return-value]

        pa_module = _pyaudio()
        device = self._resolve_device()
        self._audio = pa_module.PyAudio()
        self._resampler = StreamingResampler(device.sample_rate, self._target_rate)
        self._pending = np.zeros(0, dtype=np.float32)
        self._channels = device.channels

        # ~20 ms of device audio per PortAudio callback keeps latency low.
        frames_per_buffer = max(256, int(device.sample_rate * 0.02))
        try:
            self._stream = self._audio.open(
                format=pa_module.paInt16,
                channels=device.channels,
                rate=device.sample_rate,
                input=True,
                input_device_index=device.index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._callback,
            )
        except Exception as exc:
            self._teardown()
            raise AudioCaptureError(
                f"Could not open loopback device {device.name!r}: {exc}"
            ) from exc

        self._device = device
        self._running = True
        self._paused = False
        self._last_emit = time.monotonic()
        self._stop_keepalive.clear()
        self._keepalive = threading.Thread(
            target=self._keepalive_loop, name="loopback-keepalive", daemon=True
        )
        self._keepalive.start()
        log.info(
            "Capturing system audio from %r (%d Hz, %dch) -> %d Hz mono",
            device.name, device.sample_rate, device.channels, self._target_rate,
        )
        return device

    def _keepalive_loop(self) -> None:
        """Emit silence when the loopback endpoint goes quiet.

        Windows stops delivering loopback buffers entirely when nothing is
        playing - not silence, *nothing*.  A transcriber that receives no audio
        at all never sees the pause that ends a sentence, so the question is
        never committed, and the server eventually drops an idle connection.
        Filling the gap with real silence keeps the timeline continuous.
        """
        interval = self._chunk_ms / 1000.0
        while not self._stop_keepalive.wait(interval / 2):
            if not self._running or self._paused:
                continue
            now = time.monotonic()
            if now - self._last_emit < interval * 1.5:
                continue
            self._last_emit = now
            self.silence_chunks_emitted += 1
            try:
                self._on_chunk(self._silence, 0.0)
            except Exception:
                log.exception("Keep-alive chunk delivery failed")

    def pause(self) -> None:
        """Stop forwarding audio but keep the device open for a fast resume."""
        self._paused = True
        with self._lock:
            self._pending = np.zeros(0, dtype=np.float32)

    def resume(self) -> None:
        if self._resampler is not None:
            self._resampler.reset()
        self._paused = False

    def stop(self) -> None:
        self._running = False
        self._paused = False
        self._stop_keepalive.set()
        thread, self._keepalive = self._keepalive, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._teardown()
        log.info("System audio capture stopped")

    def _teardown(self) -> None:
        stream, audio = self._stream, self._audio
        self._stream = self._audio = None
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if audio is not None:
            try:
                audio.terminate()
            except Exception:
                pass
        with self._lock:
            self._pending = np.zeros(0, dtype=np.float32)

    def _resolve_device(self) -> LoopbackDevice:
        if self._device_index is None:
            return default_loopback_device()
        for device in list_loopback_devices():
            if device.index == self._device_index:
                return device
        log.warning(
            "Loopback device %s is no longer available; using the default device",
            self._device_index,
        )
        return default_loopback_device()

    # -- audio thread -----------------------------------------------------
    def _callback(self, in_data, frame_count, time_info, status):  # noqa: ANN001, ARG002
        pa_module = _pyaudio()
        if not self._running:
            return (None, pa_module.paComplete)
        if self._paused or not in_data:
            return (None, pa_module.paContinue)
        try:
            self._handle_audio(in_data)
        except Exception as exc:  # never let an exception kill the audio thread
            log.exception("Audio callback failed")
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    pass
        return (None, pa_module.paContinue)

    def _handle_audio(self, in_data: bytes) -> None:
        raw = np.frombuffer(in_data, dtype="<i2").astype(np.float32) / 32768.0
        mono = downmix_to_mono(raw, self._channels)
        if self._gain != 1.0:
            mono = mono * self._gain
        assert self._resampler is not None
        resampled = self._resampler.process(mono)
        if resampled.size == 0:
            return

        with self._lock:
            self._pending = np.concatenate((self._pending, resampled))
            chunks: list[np.ndarray] = []
            while self._pending.size >= self._samples_per_chunk:
                chunks.append(self._pending[: self._samples_per_chunk])
                self._pending = self._pending[self._samples_per_chunk :]

        for chunk in chunks:
            self._last_emit = time.monotonic()
            self._on_chunk(float_to_pcm16(chunk), rms_level(chunk))

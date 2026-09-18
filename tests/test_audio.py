"""Audio chunk preparation, rate conversion, and audio-device failures."""
from __future__ import annotations

import numpy as np
import pytest

from copilot.audio import capture as capture_module
from copilot.audio.capture import AudioCaptureError, LoopbackCapture, LoopbackDevice
from copilot.audio.processing import (
    SUPPORTED_PCM_RATES,
    StreamingResampler,
    audio_format_for,
    downmix_to_mono,
    float_to_pcm16,
    pcm16_to_float,
    rms_level,
)


# =========================================================================
# PCM conversion
# =========================================================================
def test_float_to_pcm16_roundtrip():
    samples = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    restored = pcm16_to_float(float_to_pcm16(samples))
    assert np.allclose(samples, restored, atol=1e-4)


def test_float_to_pcm16_clips_instead_of_wrapping():
    """Out-of-range input must saturate; wrapping would sound like a click."""
    loud = np.array([3.0, -3.0], dtype=np.float32)
    values = np.frombuffer(float_to_pcm16(loud), dtype="<i2")
    assert values[0] == 32767
    assert values[1] == -32767


def test_float_to_pcm16_is_little_endian_16bit():
    """ElevenLabs expects 16-bit little-endian PCM, so byte order matters."""
    data = float_to_pcm16(np.array([0.5], dtype=np.float32))
    assert len(data) == 2
    low, high = data[0], data[1]
    assert (high << 8 | low) == 16383  # 0.5 * 32767, low byte first
    assert np.frombuffer(data, dtype="<i2")[0] == 16383


def test_empty_input_produces_empty_output():
    assert float_to_pcm16(np.zeros(0, dtype=np.float32)) == b""


def test_downmix_averages_channels():
    interleaved = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(downmix_to_mono(interleaved, 2), [0.5, 0.5])


def test_downmix_drops_a_trailing_partial_frame():
    interleaved = np.array([1.0, 1.0, 0.0], dtype=np.float32)  # 1.5 stereo frames
    assert downmix_to_mono(interleaved, 2).tolist() == [1.0]


def test_mono_passthrough():
    samples = np.array([0.1, 0.2], dtype=np.float32)
    assert np.allclose(downmix_to_mono(samples, 1), samples)


def test_rms_level():
    assert rms_level(np.array([1.0, -1.0], dtype=np.float32)) == pytest.approx(1.0)
    assert rms_level(np.zeros(0, dtype=np.float32)) == 0.0


# =========================================================================
# Audio format negotiation
# =========================================================================
@pytest.mark.parametrize("rate", SUPPORTED_PCM_RATES)
def test_audio_format_for_supported_rates(rate):
    assert audio_format_for(rate) == f"pcm_{rate}"


def test_audio_format_rejects_unsupported_rate():
    with pytest.raises(ValueError, match="not a supported"):
        audio_format_for(32000)


# =========================================================================
# Resampling
# =========================================================================
@pytest.mark.parametrize(
    "source,target", [(48000, 16000), (44100, 16000), (22050, 16000), (16000, 16000), (48000, 24000)]
)
def test_resampler_output_length_matches_ratio(source, target):
    resampler = StreamingResampler(source, target)
    output = resampler.process(np.zeros(source, dtype=np.float32))
    assert abs(len(output) - target) <= 1


def test_resampler_preserves_tone_frequency():
    """A 440 Hz tone must still be 440 Hz after 48k -> 16k."""
    resampler = StreamingResampler(48000, 16000)
    t = np.arange(48000) / 48000
    output = resampler.process(0.5 * np.sin(2 * np.pi * 440 * t))

    window = output[4000:8096] * np.hanning(4096)
    peak_hz = float(np.argmax(np.abs(np.fft.rfft(window)))) * 16000 / 4096
    assert abs(peak_hz - 440) < 10


def test_resampler_is_chunk_boundary_invariant():
    """Chunked and whole-buffer processing must agree exactly, or we get clicks."""
    whole = StreamingResampler(48000, 16000)
    chunked = StreamingResampler(48000, 16000)
    signal = np.random.RandomState(7).randn(48000).astype(np.float32) * 0.3

    a = whole.process(signal)
    b = np.concatenate([chunked.process(signal[i : i + 777]) for i in range(0, len(signal), 777)])
    assert len(a) == len(b)
    assert np.max(np.abs(a - b)) < 1e-9


def test_resampler_attenuates_frequencies_above_target_nyquist():
    """Anti-aliasing must kill a 7 kHz tone when downsampling to 16 kHz... """
    resampler = StreamingResampler(48000, 16000)
    t = np.arange(48000) / 48000
    # 10 kHz is above the 8 kHz Nyquist of the 16 kHz target, so it must not
    # fold back into the audible band as a phantom low tone.
    output = resampler.process(0.5 * np.sin(2 * np.pi * 10000 * t))
    assert rms_level(output[2000:]) < 0.02


def test_passthrough_resampler_is_identity():
    resampler = StreamingResampler(16000, 16000)
    signal = np.random.RandomState(1).randn(1000).astype(np.float32)
    assert np.allclose(resampler.process(signal), signal)


def test_resampler_rejects_invalid_rates():
    with pytest.raises(ValueError):
        StreamingResampler(0, 16000)


def test_resampler_reset_clears_state():
    resampler = StreamingResampler(48000, 16000)
    resampler.process(np.ones(4800, dtype=np.float32))
    resampler.reset()
    first = resampler.process(np.zeros(4800, dtype=np.float32))
    assert np.max(np.abs(first)) < 1e-6


# =========================================================================
# Capture: chunking and device failures
# =========================================================================
class _FakeStream:
    def __init__(self) -> None:
        self.stopped = False
        self.closed = False

    def stop_stream(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _FakePyAudio:
    def __init__(self, fail_open: bool = False) -> None:
        self.fail_open = fail_open
        self.terminated = False
        self.stream = _FakeStream()
        self.open_kwargs = None

    def open(self, **kwargs):
        self.open_kwargs = kwargs
        if self.fail_open:
            raise OSError("device unavailable")
        return self.stream

    def terminate(self):
        self.terminated = True


@pytest.fixture
def fake_device(monkeypatch):
    device = LoopbackDevice(index=9, name="Fake Speakers [Loopback]", channels=2,
                            sample_rate=48000, is_default=True)
    monkeypatch.setattr(capture_module, "default_loopback_device", lambda: device)
    monkeypatch.setattr(capture_module, "list_loopback_devices", lambda: [device])
    return device


def _install_fake_pyaudio(monkeypatch, audio):
    module = type("Mod", (), {
        "PyAudio": lambda: audio,
        "paInt16": 8,
        "paWASAPI": 13,
        "paContinue": 0,
        "paComplete": 1,
    })
    monkeypatch.setattr(capture_module, "_pyaudio", lambda: module)
    return module


def test_capture_emits_fixed_size_chunks(monkeypatch, fake_device):
    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)

    chunks: list[bytes] = []
    capture = LoopbackCapture(
        on_chunk=lambda pcm, level: chunks.append(pcm),
        target_sample_rate=16000,
        chunk_ms=100,
    )
    capture.start()

    # One second of 48 kHz stereo int16 -> ten 100 ms mono 16 kHz chunks.
    frames = np.zeros(48000 * 2, dtype="<i2")
    capture._handle_audio(frames.tobytes())

    assert len(chunks) == 10
    assert all(len(chunk) == 3200 for chunk in chunks)  # 16000 * 0.1 * 2 bytes
    capture.stop()
    assert audio.stream.stopped and audio.stream.closed and audio.terminated


def test_capture_opens_the_device_with_the_expected_parameters(monkeypatch, fake_device):
    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None, target_sample_rate=16000)
    capture.start()

    assert audio.open_kwargs["rate"] == 48000          # device rate, not target
    assert audio.open_kwargs["channels"] == 2
    assert audio.open_kwargs["input"] is True
    assert audio.open_kwargs["input_device_index"] == 9
    capture.stop()


def test_capture_reports_a_disconnected_device(monkeypatch, fake_device):
    audio = _FakePyAudio(fail_open=True)
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None)

    with pytest.raises(AudioCaptureError, match="Could not open loopback device"):
        capture.start()
    assert not capture.is_running
    assert audio.terminated  # resources released even on failure


def test_capture_raises_when_no_device_exists(monkeypatch):
    monkeypatch.setattr(capture_module, "list_loopback_devices", lambda: [])
    with pytest.raises(AudioCaptureError, match="No WASAPI loopback device"):
        capture_module.default_loopback_device()


def test_capture_falls_back_when_the_selected_device_vanishes(monkeypatch, fake_device):
    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None, device_index=999)
    device = capture.start()
    assert device.index == fake_device.index  # fell back to the default
    capture.stop()


def test_pause_stops_forwarding_audio(monkeypatch, fake_device):
    audio = _FakePyAudio()
    module = _install_fake_pyaudio(monkeypatch, audio)
    chunks: list[bytes] = []
    capture = LoopbackCapture(on_chunk=lambda pcm, level: chunks.append(pcm))
    capture.start()

    frames = np.zeros(48000 * 2, dtype="<i2").tobytes()
    capture._callback(frames, 48000, None, 0)
    assert chunks

    chunks.clear()
    capture.pause()
    assert capture._callback(frames, 48000, None, 0) == (None, module.paContinue)
    assert chunks == []

    capture.resume()
    capture._callback(frames, 48000, None, 0)
    assert chunks
    capture.stop()


def test_callback_survives_a_processing_error(monkeypatch, fake_device):
    audio = _FakePyAudio()
    module = _install_fake_pyaudio(monkeypatch, audio)
    seen: list[Exception] = []
    capture = LoopbackCapture(
        on_chunk=lambda *_: (_ for _ in ()).throw(RuntimeError("ui exploded")),
        on_error=seen.append,
    )
    capture.start()
    # A failure downstream must not tear down the realtime audio thread.
    result = capture._callback(np.zeros(9600, dtype="<i2").tobytes(), 4800, None, 0)
    assert result == (None, module.paContinue)
    assert seen and isinstance(seen[0], RuntimeError)
    capture.stop()


# =========================================================================
# Silence keep-alive
# =========================================================================
def test_keepalive_emits_silence_when_the_device_goes_quiet(monkeypatch, fake_device):
    """Windows stops delivering loopback buffers when nothing is playing.

    Without a filler the transcriber never sees the pause that ends a sentence,
    so the question is never committed and the connection eventually drops.
    """
    import time

    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    chunks: list[bytes] = []
    capture = LoopbackCapture(
        on_chunk=lambda pcm, level: chunks.append(pcm), chunk_ms=20, target_sample_rate=16000
    )
    capture.start()
    try:
        time.sleep(0.35)
    finally:
        capture.stop()

    assert capture.silence_chunks_emitted > 0
    assert chunks, "no keep-alive audio was produced"
    expected = int(16000 * 0.02) * 2
    assert all(len(chunk) == expected for chunk in chunks)
    assert all(chunk == b"\x00" * expected for chunk in chunks)


def test_keepalive_stays_quiet_while_real_audio_flows(monkeypatch, fake_device):
    import time

    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None, chunk_ms=50, target_sample_rate=16000)
    capture.start()
    try:
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            # 50 ms of 48 kHz stereo, i.e. exactly one output chunk.
            capture._handle_audio(np.zeros(2400 * 2, dtype="<i2").tobytes())
            time.sleep(0.02)
        assert capture.silence_chunks_emitted == 0
    finally:
        capture.stop()


def test_keepalive_is_silent_while_paused(monkeypatch, fake_device):
    import time

    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None, chunk_ms=20)
    capture.start()
    capture.pause()
    try:
        time.sleep(0.25)
        assert capture.silence_chunks_emitted == 0
    finally:
        capture.stop()


def test_keepalive_thread_stops_with_the_capture(monkeypatch, fake_device):
    audio = _FakePyAudio()
    _install_fake_pyaudio(monkeypatch, audio)
    capture = LoopbackCapture(on_chunk=lambda *_: None, chunk_ms=20)
    capture.start()
    thread = capture._keepalive
    capture.stop()
    assert thread is not None and not thread.is_alive()

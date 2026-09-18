"""PCM conversion and rate conversion for the capture -> ElevenLabs path.

WASAPI hands us whatever the render device is using (commonly 48 kHz stereo
float32).  ElevenLabs wants mono 16-bit little-endian PCM at one of its
supported rates.  This module does that conversion with no SciPy dependency:

* `downmix_to_mono`  - average the channels.
* `float_to_pcm16`   - clip and scale to int16.
* `StreamingResampler` - streaming rational (L/M) polyphase resampler with a
  windowed-sinc anti-aliasing filter and carry-over state, so chunk boundaries
  do not introduce clicks.

Everything here is pure numpy so it is unit-testable without audio hardware.
"""
from __future__ import annotations

import math
from math import gcd

import numpy as np

# Rates the ElevenLabs realtime endpoint accepts as `pcm_<rate>`.
SUPPORTED_PCM_RATES: tuple[int, ...] = (8000, 16000, 22050, 24000, 44100, 48000)


def audio_format_for(sample_rate: int) -> str:
    """`audio_format` query value for a sample rate (raises if unsupported)."""
    if sample_rate not in SUPPORTED_PCM_RATES:
        raise ValueError(
            f"{sample_rate} Hz is not a supported ElevenLabs PCM rate; "
            f"choose one of {SUPPORTED_PCM_RATES}"
        )
    return f"pcm_{sample_rate}"


def downmix_to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Average interleaved `channels` into a single float32 channel."""
    if channels <= 1:
        return samples.astype(np.float32, copy=False).reshape(-1)
    flat = samples.astype(np.float32, copy=False).reshape(-1)
    usable = (flat.size // channels) * channels
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return flat[:usable].reshape(-1, channels).mean(axis=1)


def pcm16_to_float(data: bytes) -> np.ndarray:
    """int16 little-endian bytes -> float32 in [-1, 1)."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 in [-1, 1] -> int16 little-endian bytes, with clipping."""
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def rms_level(samples: np.ndarray) -> float:
    """Root-mean-square level of a float block, used for the UI meter."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _kaiser_sinc(num_taps: int, cutoff: float, beta: float = 8.0) -> np.ndarray:
    """Low-pass FIR via windowed sinc.  `cutoff` is in cycles/sample (0..0.5)."""
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    h = 2.0 * cutoff * np.sinc(2.0 * cutoff * n)
    h *= np.kaiser(num_taps, beta)
    total = h.sum()
    if total != 0:
        h /= total
    return h


class StreamingResampler:
    """Rational L/M resampler that keeps filter state between calls.

    Implemented as a polyphase filter: for output sample `n`,

        y[n] = L * sum_i  h[p + i*L] * x[base - i]

    with `p = (n*M) mod L` and `base = floor(n*M / L)`, so only `taps_per_phase`
    multiply-adds are needed per output sample regardless of L.
    """

    def __init__(self, source_rate: int, target_rate: int, taps_per_phase: int = 32) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        divisor = gcd(source_rate, target_rate)
        self.up = target_rate // divisor      # L
        self.down = source_rate // divisor    # M
        self.passthrough = self.up == 1 and self.down == 1

        if self.passthrough:
            self.taps_per_phase = 0
            self._phases = np.zeros((1, 0), dtype=np.float64)
        else:
            self.taps_per_phase = max(taps_per_phase, 8)
            num_taps = self.taps_per_phase * self.up
            # Anti-alias at the lower of the two Nyquist limits, expressed on
            # the L-times-upsampled grid; 0.98 leaves a small transition band.
            cutoff = 0.98 * 0.5 / max(self.up, self.down)
            taps = _kaiser_sinc(num_taps, cutoff)
            # phase p holds taps h[p], h[p+L], h[p+2L], ...
            self._phases = taps.reshape(self.taps_per_phase, self.up).T.copy()

        self._history = np.zeros(max(self.taps_per_phase - 1, 0), dtype=np.float64)
        self._consumed = 0    # input samples already retired from the stream
        self._out_index = 0   # next output sample index

    @property
    def ratio(self) -> float:
        return self.up / self.down

    def reset(self) -> None:
        self._history[:] = 0.0
        self._consumed = 0
        self._out_index = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Feed a mono float block, get back the resampled float block."""
        block = np.asarray(samples, dtype=np.float64).reshape(-1)
        if self.passthrough:
            return block.astype(np.float32)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)

        buf = np.concatenate((self._history, block))
        # Absolute index of buf[0] within the whole input stream.
        origin = self._consumed - self._history.size
        last_available = origin + buf.size - 1

        # Output indices whose newest input sample `base` is available.
        n0 = self._out_index
        # base(n) = floor(n*M/L) <= last_available  =>  n <= (last_available+1)*L/M - 1
        n_end = ((last_available + 1) * self.up) // self.down
        if n_end <= n0:
            self._history = buf[-self._history.size:] if self._history.size else buf[:0]
            self._consumed = origin + buf.size
            return np.zeros(0, dtype=np.float32)

        n = np.arange(n0, n_end, dtype=np.int64)
        nm = n * self.down
        phase = nm % self.up
        base = nm // self.up

        # Gather the taps_per_phase newest inputs for each output sample.
        local = base - origin                                   # index into buf
        offsets = np.arange(self.taps_per_phase, dtype=np.int64)  # i
        idx = local[:, None] - offsets[None, :]
        np.clip(idx, 0, buf.size - 1, out=idx)                  # only bites at t=0
        window = buf[idx]                                       # (out, taps)
        coeffs = self._phases[phase]                            # (out, taps)
        out = self.up * np.einsum("ij,ij->i", window, coeffs)

        # Retain the samples still needed by future outputs.
        keep = self._history.size
        if keep:
            self._history = buf[-keep:].copy()
        self._consumed = origin + buf.size
        self._out_index = int(n_end)
        return out.astype(np.float32)

    def expected_output_length(self, input_length: int) -> int:
        """Approximate output length for an input block (for tests / sizing)."""
        if self.passthrough:
            return input_length
        return int(math.floor(input_length * self.ratio))

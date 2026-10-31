"""Pulse Code Modulation (PCM) with Uniform Quantization and Mu-Law Companding.

Owner: Student 1 (Module A: Source and Baseband Lab)

Covers:
- Uniform quantizer (mid-rise) configurable from 1 to 12 bits.
- ITU-T G.711 mu-law companding (mu = 255) for dynamic range compression.
- Accurate empirical SQNR computation and comparison with theoretical 6.02n + 1.76 dB.
- Audio synthesis and WAV conversion.
- Continuous-time Nyquist sampling and aliasing demonstration.
"""

from __future__ import annotations
from dataclasses import dataclass
import io
import wave
import numpy as np
import scipy.signal


def mu_law_compress(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Apply ITU-T G.711 mu-law compression characteristic.

    Formula:
        y = sgn(x) * ln(1 + mu * |x|) / ln(1 + mu)   for |x| <= 1.0

    Parameters
    ----------
    x : np.ndarray
        Normalized input signal in [-1.0, +1.0].
    mu : float
        Companding parameter (standard speech: mu = 255).

    Returns
    -------
    np.ndarray
        Compressed signal bounded in [-1.0, +1.0].
    """
    x_clipped = np.clip(x, -1.0, 1.0)
    return np.sign(x_clipped) * (np.log(1.0 + mu * np.abs(x_clipped)) / np.log(1.0 + mu))


def mu_law_expand(y: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Apply ITU-T G.711 mu-law expansion (inverse characteristic).

    Formula:
        x = sgn(y) * ((1 + mu)^|y| - 1) / mu   for |y| <= 1.0

    Parameters
    ----------
    y : np.ndarray
        Compressed/quantized signal in [-1.0, +1.0].
    mu : float
        Companding parameter (mu = 255).

    Returns
    -------
    np.ndarray
        Expanded reconstructed signal in [-1.0, +1.0].
    """
    y_clipped = np.clip(y, -1.0, 1.0)
    return np.sign(y_clipped) * (((1.0 + mu) ** np.abs(y_clipped) - 1.0) / mu)


def uniform_quantize(
    x: np.ndarray,
    n_bits: int,
    v_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perform mid-rise uniform quantization over [-v_max, +v_max].

    Step size:
        Delta = 2 * v_max / (2^n)
    Quantizer levels:
        q_k = -v_max + (k + 0.5) * Delta,  k in [0, 2^n - 1]

    Parameters
    ----------
    x : np.ndarray
        Input continuous/sampled signal.
    n_bits : int
        Number of quantization bits (1 to 12).
    v_max : float
        Peak full-scale voltage.

    Returns
    -------
    x_hat : np.ndarray
        Quantized reconstructed signal.
    codes : np.ndarray
        Integer quantization indices in [0, 2^n - 1].
    delta : float
        Step size.
    """
    if n_bits < 1 or n_bits > 16:
        raise ValueError("n_bits must be between 1 and 16.")

    num_levels = 1 << n_bits
    delta = 2.0 * v_max / num_levels

    # Symmetrical clipping to [-v_max, +v_max]
    x_clipped = np.clip(x, -v_max, v_max)

    # Mid-rise index calculation
    indices = np.floor((x_clipped + v_max) / delta).astype(np.int64)
    indices = np.clip(indices, 0, num_levels - 1)

    # Mid-rise reconstruction: center of interval
    x_hat = -v_max + (indices + 0.5) * delta
    return x_hat, indices, delta


def calculate_sqnr(original: np.ndarray, quantized: np.ndarray) -> float:
    """Measure empirical Signal-to-Quantization-Noise Ratio (SQNR) in decibels.

    SQNR_dB = 10 * log10( sum(|x|^2) / sum(|x - x_hat|^2) )

    Parameters
    ----------
    original : np.ndarray
        Unquantized original signal.
    quantized : np.ndarray
        Quantized reconstructed signal.

    Returns
    -------
    float
        SQNR in dB.
    """
    orig = np.asarray(original, dtype=np.float64)
    quant = np.asarray(quantized, dtype=np.float64)

    signal_power = float(np.mean(orig**2))
    noise_power = float(np.mean((orig - quant) ** 2))

    if noise_power <= 1e-18:
        return 100.0  # Avoid division by zero for lossless/near-lossless
    if signal_power <= 1e-18:
        return -100.0

    return 10.0 * np.log10(signal_power / noise_power)


def theoretical_sqnr_sine(n_bits: int) -> float:
    """Theoretical SQNR for a full-scale sinusoid: 6.02 * n + 1.76 dB."""
    return 6.0206 * float(n_bits) + 1.7609


@dataclass
class PCMEncodingResult:
    """Results from PCM encoding/decoding pipeline."""
    original_signal: np.ndarray
    quantized_signal: np.ndarray
    indices: np.ndarray
    bitstream: np.ndarray
    measured_sqnr_db: float
    theoretical_sqnr_db: float
    step_size: float
    companded: bool


class PCMCodec:
    """Full PCM Codec pipeline supporting uniform and mu-law companded modes."""

    def __init__(self, n_bits: int = 8, companding: bool = False, mu: float = 255.0) -> None:
        self.n_bits = n_bits
        self.companding = companding
        self.mu = mu

    def encode_and_decode(self, signal: np.ndarray, v_max: float = 1.0) -> PCMEncodingResult:
        """Encode an analog/discrete signal into bits and reconstruct it."""
        x = np.asarray(signal, dtype=np.float64)

        if self.companding:
            # Compress -> Uniform Quantize -> Expand
            x_norm = x / v_max
            x_comp = mu_law_compress(x_norm, self.mu)
            q_comp, indices, delta = uniform_quantize(x_comp, self.n_bits, v_max=1.0)
            x_hat = mu_law_expand(q_comp, self.mu) * v_max
        else:
            x_hat, indices, delta = uniform_quantize(x, self.n_bits, v_max=v_max)

        # Convert indices to binary bitstream
        # Vectorized bit unpack
        bit_matrix = ((indices[:, None] & (1 << np.arange(self.n_bits - 1, -1, -1))) > 0).astype(np.uint8)
        bitstream = bit_matrix.ravel()

        sqnr_meas = calculate_sqnr(x, x_hat)
        sqnr_theo = theoretical_sqnr_sine(self.n_bits)

        return PCMEncodingResult(
            original_signal=x,
            quantized_signal=x_hat,
            indices=indices,
            bitstream=bitstream,
            measured_sqnr_db=sqnr_meas,
            theoretical_sqnr_db=sqnr_theo,
            step_size=delta,
            companded=self.companding,
        )


def generate_synthetic_tone(
    freq_hz: float = 440.0,
    duration_s: float = 1.0,
    sampling_rate_hz: float = 8000.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean synthetic sinusoidal test tone.

    Returns
    -------
    t : np.ndarray
        Time vector.
    signal : np.ndarray
        Sinusoidal values.
    """
    num_samples = int(duration_s * sampling_rate_hz)
    t = np.arange(num_samples) / sampling_rate_hz
    sig = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return t, sig


def signal_to_wav_bytes(signal: np.ndarray, sampling_rate_hz: int = 8000) -> bytes:
    """Export a normalized float signal in [-1.0, 1.0] to in-memory 16-bit WAV bytes."""
    sig_clipped = np.clip(signal, -1.0, 1.0)
    int_samples = (sig_clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate_hz)
        wf.writeframes(int_samples.tobytes())
    return buffer.getvalue()


def demonstrate_aliasing(
    signal_freq_hz: float = 120.0,
    sampling_rate_hz: float = 160.0,  # Below Nyquist rate of 240 Hz
    duration_s: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Demonstrate sampling below the Nyquist rate and subsequent spectral folding.

    When Fs < 2 * f0, the sampled signal manifests at the alias frequency:
        f_alias = |f0 - round(f0 / Fs) * Fs|

    Returns
    -------
    dict
        Continuous time/signal, sampled points, reconstructed alias signal, and frequencies.
    """
    # High-resolution "continuous" reference
    fs_cont = max(20000.0, 50.0 * signal_freq_hz)
    t_cont = np.linspace(0.0, duration_s, int(duration_s * fs_cont), endpoint=False)
    x_cont = np.cos(2.0 * np.pi * signal_freq_hz * t_cont)

    # Actual sampled discrete points
    num_samples = int(duration_s * sampling_rate_hz)
    t_sampled = np.arange(num_samples) / sampling_rate_hz
    x_sampled = np.cos(2.0 * np.pi * signal_freq_hz * t_sampled)

    # Theoretical alias frequency in baseband [0, Fs/2]
    k = np.round(signal_freq_hz / sampling_rate_hz)
    alias_freq = abs(signal_freq_hz - k * sampling_rate_hz)

    # Perceived reconstructed continuous alias waveform
    x_reconstructed = np.cos(2.0 * np.pi * alias_freq * t_cont)

    return {
        "t_cont": t_cont,
        "x_cont": x_cont,
        "t_sampled": t_sampled,
        "x_sampled": x_sampled,
        "x_reconstructed": x_reconstructed,
        "original_freq_hz": signal_freq_hz,
        "sampling_rate_hz": sampling_rate_hz,
        "nyquist_rate_hz": 2.0 * signal_freq_hz,
        "alias_freq_hz": alias_freq,
        "is_aliased": bool(sampling_rate_hz < 2.0 * signal_freq_hz),
    }

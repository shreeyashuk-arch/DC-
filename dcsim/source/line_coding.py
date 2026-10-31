"""Line coding encoders, waveform synthesizers, Welch PSD analysis, and comparison tables.

Owner: Student 1 (Module A: Source and Baseband Lab)

Supported Codes:
1. NRZ-L (Non-Return-to-Zero Level)
2. RZ (Unipolar Return-to-Zero, 50% duty cycle)
3. Manchester (Biphase Level)
4. AMI (Alternate Mark Inversion / Bipolar)
"""

from __future__ import annotations
from typing import Literal
import numpy as np
import scipy.signal

LineCodeType = Literal["NRZ-L", "RZ", "Manchester", "AMI"]


def encode_line_code(
    bits: np.ndarray,
    scheme: LineCodeType,
    samples_per_bit: int = 32,
    voltage: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize discrete time-domain waveform for a bit sequence and line code.

    Parameters
    ----------
    bits : np.ndarray
        1D array of binary digits (0 or 1).
    scheme : LineCodeType
        "NRZ-L", "RZ", "Manchester", or "AMI".
    samples_per_bit : int
        Oversampling factor L (samples per bit duration Tb). Must be even.
    voltage : float
        Peak signaling voltage V.

    Returns
    -------
    t_norm : np.ndarray
        Time vector normalized to bit duration Tb (e.g., [0, N_bits]).
    waveform : np.ndarray
        Synthesized discrete voltage waveform.
    """
    if samples_per_bit % 2 != 0:
        samples_per_bit += 1  # Ensure clean integer half-periods

    b = np.asarray(bits, dtype=np.int8).ravel()
    n_bits = len(b)
    if n_bits == 0:
        return np.empty(0), np.empty(0)

    L = samples_per_bit
    half_L = L // 2
    total_samples = n_bits * L
    t_norm = np.arange(total_samples) / float(L)

    scheme_upper = scheme.upper()

    if scheme_upper == "NRZ-L":
        # 1 -> +V, 0 -> -V
        levels = np.where(b == 1, voltage, -voltage)
        waveform = np.repeat(levels, L)

    elif scheme_upper == "RZ":
        # 1 -> +V for first half, 0 for second half; 0 -> 0 for entire bit
        waveform = np.zeros(total_samples, dtype=np.float64)
        for i, bit in enumerate(b):
            if bit == 1:
                waveform[i * L : i * L + half_L] = voltage

    elif scheme_upper == "MANCHESTER":
        # Standard IEEE 802.3: 1 -> +V first half, -V second half; 0 -> -V first half, +V second half
        waveform = np.zeros(total_samples, dtype=np.float64)
        for i, bit in enumerate(b):
            if bit == 1:
                waveform[i * L : i * L + half_L] = voltage
                waveform[i * L + half_L : (i + 1) * L] = -voltage
            else:
                waveform[i * L : i * L + half_L] = -voltage
                waveform[i * L + half_L : (i + 1) * L] = voltage

    elif scheme_upper == "AMI":
        # 0 -> 0; 1 -> alternates between +V and -V
        waveform = np.zeros(total_samples, dtype=np.float64)
        polarity = voltage
        for i, bit in enumerate(b):
            if bit == 1:
                waveform[i * L : (i + 1) * L] = polarity
                polarity = -polarity  # Alternate on every mark
            else:
                waveform[i * L : (i + 1) * L] = 0.0

    else:
        raise ValueError(f"Unsupported line coding scheme: {scheme}")

    return t_norm, waveform


def compute_line_code_psd(
    scheme: LineCodeType,
    num_bits: int = 10000,
    samples_per_bit: int = 32,
    voltage: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) using Welch's averaged periodogram.

    Parameters
    ----------
    scheme : LineCodeType
        Line code scheme.
    num_bits : int
        Number of random bits for statistical averaging.
    samples_per_bit : int
        Samples per bit.
    voltage : float
        Signaling voltage.
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    freq_norm : np.ndarray
        Frequency normalized to bit rate (f / Rb).
    psd : np.ndarray
        Power spectral density values (V^2 / Hz).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)
    _, waveform = encode_line_code(bits, scheme, samples_per_bit=samples_per_bit, voltage=voltage)

    # Welch PSD estimation
    fs = float(samples_per_bit)  # fs / Rb = L
    nperseg = min(2048, len(waveform) // 4)
    freqs, psd = scipy.signal.welch(
        waveform,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    return freqs, psd


def get_line_code_summary_table() -> list[dict[str, str]]:
    """Return comparative engineering summary table for all four line codes."""
    return [
        {
            "Code": "NRZ-L",
            "DC Content": "High (continuous DC bias)",
            "Bandwidth": "Rb / 2 (null at Rb)",
            "Self-Clocking": "Poor (fails on long runs of identical bits)",
            "Error Detection": "None (no inherent structure)",
            "Key Applications": "Digital logic, RS-232, memory buses",
        },
        {
            "Code": "RZ",
            "DC Content": "Moderate (positive mean)",
            "Bandwidth": "Rb (first null at 2*Rb)",
            "Self-Clocking": "Moderate (clock line present, but runs of 0s lack transitions)",
            "Error Detection": "None",
            "Key Applications": "Baseband synchronization testing, optical links",
        },
        {
            "Code": "Manchester",
            "DC Content": "Zero (strictly balanced biphase)",
            "Bandwidth": "2 * Rb (wider bandwidth required)",
            "Self-Clocking": "Excellent (guaranteed transition in every bit)",
            "Error Detection": "Good (missing transition indicates bit error)",
            "Key Applications": "10BASE-T Ethernet, RFID, aerospace data buses",
        },
        {
            "Code": "AMI",
            "DC Content": "Zero (alternating polarities eliminate DC)",
            "Bandwidth": "Rb (null at DC and null at Rb)",
            "Self-Clocking": "Poor for long zero runs (solved by B8ZS/HDB3)",
            "Error Detection": "Built-in (bipolar violation detection)",
            "Key Applications": "T1/E1 telecommunication carrier systems, ISDN",
        },
    ]

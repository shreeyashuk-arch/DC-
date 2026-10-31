"""Global mathematical conventions and AWGN noise models for digital communication simulations.

CONVENTIONS APPLIED:
1. Eb/N0 in dB: User-facing SNR parameter. Eb represents the average energy per
   INFORMATION bit (not per channel/coded bit, nor per symbol).
2. Sample-measured Eb:
       Eb = sum(|x|^2) / number_of_information_bits
   Measuring Eb directly from the synthesized discrete waveform x guarantees that
   pulse-shaping gain, carrier scaling, oversampling factors, and channel coding rates
   (R = k/n) are accounted for automatically without manual fudge factors.
3. AWGN Noise Power Spectral Density N0:
       N0 = Eb / (10^(Eb_N0_dB / 10))
4. Noise Variance per Dimension:
   - For real-valued signals: noise ~ N(0, sigma^2) with sigma^2 = N0 / 2.
   - For complex-valued signals: noise = w_I + j*w_Q with w_I, w_Q ~ N(0, N0 / 2),
     giving total complex noise variance E[|w|^2] = N0.
5. Theoretical Gaussian tail integral (Q-function):
       Q(x) = 0.5 * erfc(x / sqrt(2))
"""

from __future__ import annotations
import numpy as np
import scipy.special


def q_function(x: float | np.ndarray) -> float | np.ndarray:
    """Evaluate Gaussian tail probability Q(x) = P(X > x) for X ~ N(0, 1).

    Computed via standard complementary error function:
        Q(x) = 0.5 * erfc(x / sqrt(2))

    Parameters
    ----------
    x : float or np.ndarray
        Argument to the Q-function.

    Returns
    -------
    float or np.ndarray
        Tail probability value in [0, 1].
    """
    x_arr = np.asarray(x, dtype=np.float64)
    val = 0.5 * scipy.special.erfc(x_arr / np.sqrt(2.0))
    if np.isscalar(x):
        return float(val)
    return val


def erfc_from_q(q_val: float | np.ndarray) -> float | np.ndarray:
    """Invert or relate Q-function value to erfc."""
    return 2.0 * np.asarray(q_val, dtype=np.float64)


def measure_eb(signal: np.ndarray, num_info_bits: int) -> float:
    """Compute the actual transmitted energy per information bit Eb.

    Parameters
    ----------
    signal : np.ndarray
        Discrete transmitted waveform (real or complex samples).
    num_info_bits : int
        Total number of raw information bits transmitted.

    Returns
    -------
    float
        Measured energy per information bit (sum(|x|^2) / num_info_bits).

    Raises
    ------
    ValueError
        If num_info_bits is not positive or signal is empty.
    """
    if num_info_bits <= 0:
        raise ValueError("num_info_bits must be strictly positive.")
    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    total_energy = float(np.sum(np.abs(signal) ** 2))
    return total_energy / float(num_info_bits)


def calculate_n0(eb: float, eb_n0_db: float) -> float:
    """Calculate the one-sided noise spectral density N0 from Eb and Eb/N0 in dB.

    Parameters
    ----------
    eb : float
        Energy per information bit.
    eb_n0_db : float
        Eb/N0 ratio in decibels (dB).

    Returns
    -------
    float
        Noise spectral density N0.
    """
    eb_n0_linear = 10.0 ** (eb_n0_db / 10.0)
    if eb_n0_linear <= 0.0:
        raise ValueError("Eb/N0 linear must be positive.")
    return eb / eb_n0_linear


def add_awgn_noise(
    signal: np.ndarray,
    eb_n0_db: float,
    num_info_bits: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float]:
    """Add Additive White Gaussian Noise (AWGN) to a transmitted signal.

    Enforces the global project convention:
      - Noise variance is N0/2 per real dimension.
      - Eb is measured directly from the discrete samples.

    Parameters
    ----------
    signal : np.ndarray
        Transmitted waveform samples (1D array, real or complex).
    eb_n0_db : float
        Eb/N0 ratio in decibels.
    num_info_bits : int
        Number of raw information bits carried by the signal.
    rng : np.random.Generator, optional
        NumPy random number generator for reproducible seeds.

    Returns
    -------
    noisy_signal : np.ndarray
        Received signal after AWGN contamination.
    eb : float
        Measured energy per information bit.
    n0 : float
        Calculated noise spectral density N0.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    eb = measure_eb(signal, num_info_bits)
    n0 = calculate_n0(eb, eb_n0_db)

    # Variance per dimension is N0 / 2
    sigma_per_dim = np.sqrt(n0 / 2.0)

    if np.iscomplexobj(signal):
        noise_i = rng.normal(0.0, sigma_per_dim, size=signal.shape)
        noise_q = rng.normal(0.0, sigma_per_dim, size=signal.shape)
        noise = noise_i + 1j * noise_q
    else:
        noise = rng.normal(0.0, sigma_per_dim, size=signal.shape)

    noisy_signal = signal + noise
    return noisy_signal, eb, n0


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> tuple[float, int, int]:
    """Calculate the Bit Error Rate (BER) between transmitted and received bits.

    Parameters
    ----------
    tx_bits : np.ndarray
        Ground-truth transmitted bits (0s and 1s).
    rx_bits : np.ndarray
        Demodulated/decoded received bits (0s and 1s).

    Returns
    -------
    ber : float
        Bit error rate (bit_errors / total_bits).
    bit_errors : int
        Total number of mismatched bits.
    total_bits : int
        Total number of compared bits.
    """
    tx = np.asarray(tx_bits, dtype=np.int8).ravel()
    rx = np.asarray(rx_bits, dtype=np.int8).ravel()

    min_len = min(len(tx), len(rx))
    if min_len == 0:
        return 0.0, 0, 0

    bit_errors = int(np.sum(tx[:min_len] != rx[:min_len]))
    # Add penalty for any length mismatch
    length_diff = abs(len(tx) - len(rx))
    total_errors = bit_errors + length_diff
    total_bits = max(len(tx), len(rx))

    ber = float(total_errors) / float(total_bits)
    return ber, total_errors, total_bits

"""Flat Rayleigh Fading Channel Model with Coherent Detection and Perfect CSI.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)

Covers:
- Zero-mean complex Gaussian fading envelope |h| ~ Rayleigh with E[|h|^2] = 1.0.
- Fast fading (i.i.d. symbol-by-symbol) and block fading (bursty error conditions).
- Coherent Maximum Ratio / Matched Phase Receiver with Perfect Channel State Information (CSI).
- Analytical theoretical BER benchmark: Pb = 0.5 * (1 - sqrt(g / (1 + g))), g = average Eb/N0.
- Massive performance gap comparison against AWGN channel.
"""

from __future__ import annotations
import numpy as np
from dcsim.core.conventions import q_function, add_awgn_noise, calculate_ber, measure_eb


def theoretical_rayleigh_bpsk_ber(eb_n0_db: float | np.ndarray) -> np.ndarray:
    """Theoretical BER for coherent BPSK over flat Rayleigh fading with perfect CSI.

    Formula:
        Pb = 0.5 * ( 1 - sqrt( g / (1 + g) ) )
    where g = 10^(Eb_N0_dB / 10) is the average SNR per bit.
    """
    g = 10.0 ** (np.asarray(eb_n0_db, dtype=np.float64) / 10.0)
    return 0.5 * (1.0 - np.sqrt(g / (1.0 + g)))


class RayleighChannel:
    """Simulates a flat Rayleigh fading channel with additive white Gaussian noise."""

    def __init__(self, block_length: int = 1, rng: np.random.Generator | None = None) -> None:
        """Initialize channel.

        Parameters
        ----------
        block_length : int
            Number of consecutive symbols sharing the same fading realization h.
            block_length = 1 corresponds to ideal rapid/fast fading.
            block_length > 1 corresponds to slow/block fading inducing error bursts.
        rng : np.random.Generator, optional
            NumPy random generator.
        """
        self.block_length = max(1, block_length)
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def apply(
        self,
        signal: np.ndarray,
        eb_n0_db: float,
        num_info_bits: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Transmit complex or real signal through flat Rayleigh fading channel.

        Returns
        -------
        y : np.ndarray
            Faded and noise-corrupted received signal: y = h * x + w.
        h : np.ndarray
            Channel fading coefficients (perfect CSI reference).
        eb : float
            Measured energy per information bit.
        n0 : float
            Noise power spectral density.
        """
        x = np.asarray(signal, dtype=np.complex128)
        num_symbols = len(x)

        # Generate complex Gaussian fading coefficients: h_I, h_Q ~ N(0, 0.5)
        # E[|h|^2] = E[h_I^2] + E[h_Q^2] = 0.5 + 0.5 = 1.0
        num_blocks = int(np.ceil(num_symbols / self.block_length))
        h_blocks_i = self.rng.normal(0.0, 1.0 / np.sqrt(2.0), size=num_blocks)
        h_blocks_q = self.rng.normal(0.0, 1.0 / np.sqrt(2.0), size=num_blocks)
        h_blocks = h_blocks_i + 1j * h_blocks_q

        h = np.repeat(h_blocks, self.block_length)[:num_symbols]

        # Fading multiplication
        faded_signal = h * x

        # AWGN addition according to global convention
        noisy_signal, eb, n0 = add_awgn_noise(faded_signal, eb_n0_db, num_info_bits, rng=self.rng)
        return noisy_signal, h, eb, n0


def simulate_rayleigh_bpsk(
    num_bits: int,
    eb_n0_db: float,
    block_length: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Simulate coherent BPSK transmission over flat Rayleigh fading with perfect CSI."""
    if rng is None:
        rng = np.random.default_rng(42)

    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)
    symbols = np.where(bits == 1, 1.0, -1.0).astype(np.complex128)

    channel = RayleighChannel(block_length=block_length, rng=rng)
    y, h, eb, n0 = channel.apply(symbols, eb_n0_db, num_info_bits=num_bits)

    # Maximum Ratio Combining / Phase matched equalizer with perfect CSI
    # Decision statistic: Re{ h^* * y }
    decision_stat = np.real(np.conj(h) * y)
    rx_bits = (decision_stat > 0.0).astype(np.int8)

    ber, errors, _ = calculate_ber(bits, rx_bits)
    theo_rayleigh = float(theoretical_rayleigh_bpsk_ber(eb_n0_db))
    theo_awgn = float(q_function(np.sqrt(2.0 * (10.0 ** (eb_n0_db / 10.0)))))

    return {
        "ber_sim": ber,
        "ber_theo_rayleigh": theo_rayleigh,
        "ber_theo_awgn": theo_awgn,
        "awgn_vs_rayleigh_gap_ratio": theo_rayleigh / (theo_awgn + 1e-12),
        "errors": errors,
        "fading_envelope": np.abs(h),
        "decision_statistics": decision_stat,
    }

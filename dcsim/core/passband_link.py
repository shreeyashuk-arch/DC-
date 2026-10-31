"""Real passband digital communication transceivers with correlator receivers.

Implements:
- Modulation: ASK (OOK), coherent BFSK (orthogonal tones, integer cycles per bit),
  BPSK, and QPSK (Gray coded).
- Passband waveform synthesis using an integer number of carrier cycles per bit/symbol.
- Coherent correlator receivers computing projections onto orthonormal basis signals.
- Extraction of decision statistics for signal-space constellation diagrams.
- Theoretical BER benchmark functions for all supported schemes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from .conventions import q_function, measure_eb, calculate_n0, add_awgn_noise, calculate_ber

SchemeType = Literal["BPSK", "QPSK", "BFSK", "OOK"]


@dataclass
class ModulatedSignal:
    """Container for modulated passband waveform and metadata."""
    waveform: np.ndarray
    scheme: SchemeType
    num_info_bits: int
    samples_per_bit: int
    carrier_freq_hz: float
    sampling_rate_hz: float
    basis_signals: dict[str, np.ndarray]


@dataclass
class DemodulationResult:
    """Container for demodulated bits and signal-space decision statistics."""
    rx_bits: np.ndarray
    decision_statistics: np.ndarray
    threshold: float | None = None


def theoretical_ber(scheme: SchemeType, eb_n0_db: float | np.ndarray) -> np.ndarray:
    """Calculate theoretical Bit Error Rate (BER) for passband modulation schemes.

    Formulas:
      - BPSK:  Pb = Q(sqrt(2 * Eb/N0))
      - QPSK:  Pb = Q(sqrt(2 * Eb/N0))   (Gray coded, identical to BPSK)
      - BFSK:  Pb = Q(sqrt(Eb/N0))       (Coherent orthogonal BFSK)
      - OOK:   Pb = Q(sqrt(Eb/N0))       (Equiprobable ASK with optimal threshold)

    where Q(x) = 0.5 * erfc(x / sqrt(2)).

    Parameters
    ----------
    scheme : SchemeType
        One of "BPSK", "QPSK", "BFSK", "OOK".
    eb_n0_db : float or np.ndarray
        Eb/N0 in decibels.

    Returns
    -------
    np.ndarray
        Theoretical BER values.
    """
    gamma_b = 10.0 ** (np.asarray(eb_n0_db, dtype=np.float64) / 10.0)
    scheme_upper = scheme.upper()

    if scheme_upper in ("BPSK", "QPSK"):
        return q_function(np.sqrt(2.0 * gamma_b))
    elif scheme_upper in ("BFSK", "OOK"):
        return q_function(np.sqrt(gamma_b))
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Supported: BPSK, QPSK, BFSK, OOK.")


class PassbandModulator:
    """Synthesizes real passband continuous/sampled waveforms with integer carrier cycles."""

    def __init__(
        self,
        samples_per_bit: int = 32,
        carrier_cycles_per_bit: int = 4,
        sampling_rate_hz: float = 32000.0,
    ) -> None:
        """Initialize modulator parameters.

        Parameters
        ----------
        samples_per_bit : int
            Number of discrete samples per information bit (L).
        carrier_cycles_per_bit : int
            Integer carrier cycles completed during one bit period Tb.
        sampling_rate_hz : float
            Sampling rate Fs in Hertz.
        """
        self.L = samples_per_bit
        self.m = carrier_cycles_per_bit
        self.Fs = sampling_rate_hz
        self.Tb = self.L / self.Fs
        self.fc = self.m / self.Tb

        # Construct normalized orthonormal carrier basis for duration Tb
        t_b = np.arange(self.L) / self.Fs
        # Basis for BPSK & OOK
        cos_carrier = np.cos(2.0 * np.pi * self.fc * t_b)
        self.phi_cos = cos_carrier / np.sqrt(np.sum(cos_carrier**2))

        # BFSK Orthogonal tone bases (m1 and m2 cycles per bit)
        self.m1 = self.m
        self.m2 = self.m + 2  # Integer difference guarantees exact orthogonality
        c1 = np.cos(2.0 * np.pi * (self.m1 / self.Tb) * t_b)
        c2 = np.cos(2.0 * np.pi * (self.m2 / self.Tb) * t_b)
        self.phi_fsk1 = c1 / np.sqrt(np.sum(c1**2))
        self.phi_fsk2 = c2 / np.sqrt(np.sum(c2**2))

        # QPSK Basis for symbol duration Ts = 2 * Tb (2*L samples)
        t_s = np.arange(2 * self.L) / self.Fs
        Ts = 2.0 * self.Tb
        fc_qpsk = (2 * self.m) / Ts  # Maintains the same carrier frequency
        qpsk_cos = np.cos(2.0 * np.pi * fc_qpsk * t_s)
        qpsk_sin = -np.sin(2.0 * np.pi * fc_qpsk * t_s)
        self.phi_qpsk_i = qpsk_cos / np.sqrt(np.sum(qpsk_cos**2))
        self.phi_qpsk_q = qpsk_sin / np.sqrt(np.sum(qpsk_sin**2))

    def modulate(self, bits: np.ndarray, scheme: SchemeType) -> ModulatedSignal:
        """Modulate a bit sequence into a passband waveform.

        Parameters
        ----------
        bits : np.ndarray
            1D array of binary digits (0 or 1).
        scheme : SchemeType
            "BPSK", "QPSK", "BFSK", or "OOK".

        Returns
        -------
        ModulatedSignal
            Synthesized waveform and associated parameters.
        """
        b = np.asarray(bits, dtype=np.int8).ravel()
        n_bits = len(b)
        if n_bits == 0:
            raise ValueError("Input bits cannot be empty.")

        scheme_upper = scheme.upper()

        if scheme_upper == "BPSK":
            # Map 1 -> +1, 0 -> -1
            # Waveform for bit n: a_n * phi_cos (has unit energy per bit)
            symbols = np.where(b == 1, 1.0, -1.0)
            waveform = np.kron(symbols, self.phi_cos)
            bases = {"phi_cos": self.phi_cos}

        elif scheme_upper == "OOK":
            # Map 1 -> sqrt(2)*phi_cos, 0 -> 0
            # Average energy per bit: 0.5 * 2 + 0.5 * 0 = 1.0
            symbols = np.where(b == 1, np.sqrt(2.0), 0.0)
            waveform = np.kron(symbols, self.phi_cos)
            bases = {"phi_cos": self.phi_cos}

        elif scheme_upper == "BFSK":
            # Bit 1 -> phi_fsk1, Bit 0 -> phi_fsk2
            # Each basis has unit energy, so energy per bit = 1.0
            symbols_1 = np.where(b == 1, 1.0, 0.0)
            symbols_0 = np.where(b == 0, 1.0, 0.0)
            waveform = np.kron(symbols_1, self.phi_fsk1) + np.kron(symbols_0, self.phi_fsk2)
            bases = {"phi_fsk1": self.phi_fsk1, "phi_fsk2": self.phi_fsk2}

        elif scheme_upper == "QPSK":
            # Gray coded QPSK: group into pairs (b0, b1)
            # If odd length, pad with trailing 0
            if n_bits % 2 != 0:
                b = np.append(b, 0)
                n_bits += 1

            b_reshaped = b.reshape(-1, 2)
            b0 = b_reshaped[:, 0]
            b1 = b_reshaped[:, 1]

            # Gray code mapping:
            # (0, 0) -> (+1, +1) / sqrt(2)
            # (0, 1) -> (-1, +1) / sqrt(2)
            # (1, 1) -> (-1, -1) / sqrt(2)
            # (1, 0) -> (+1, -1) / sqrt(2)
            # Notice adjacent constellation points differ by exactly 1 bit.
            a_i = np.where(b0 == 0, 1.0, -1.0) / np.sqrt(2.0)
            a_q = np.where(b1 == 0, 1.0, -1.0) / np.sqrt(2.0)

            # Energy per symbol: (a_i^2 + a_q^2) * 2 = 1 * 2 = 2 (energy for 2 info bits)
            # So energy per information bit is exactly 1.0
            scale = np.sqrt(2.0)
            wave_i = np.kron(a_i * scale, self.phi_qpsk_i)
            wave_q = np.kron(a_q * scale, self.phi_qpsk_q)
            waveform = wave_i + wave_q
            bases = {"phi_qpsk_i": self.phi_qpsk_i, "phi_qpsk_q": self.phi_qpsk_q}

        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

        return ModulatedSignal(
            waveform=waveform,
            scheme=scheme_upper,  # type: ignore
            num_info_bits=n_bits,
            samples_per_bit=self.L,
            carrier_freq_hz=self.fc,
            sampling_rate_hz=self.Fs,
            basis_signals=bases,
        )


class PassbandDemodulator:
    """Correlator receiver computing signal-space decision statistics."""

    def __init__(self, modulator: PassbandModulator) -> None:
        """Initialize demodulator using the transmitter's basis parameters."""
        self.mod = modulator

    def demodulate(
        self,
        noisy_waveform: np.ndarray,
        scheme: SchemeType,
        num_info_bits: int,
        eb_estimate: float = 1.0,
    ) -> DemodulationResult:
        """Demodulate a received noisy passband waveform via matched correlation.

        Parameters
        ----------
        noisy_waveform : np.ndarray
            Received samples after AWGN channel.
        scheme : SchemeType
            "BPSK", "QPSK", "BFSK", or "OOK".
        num_info_bits : int
            Original number of information bits transmitted.
        eb_estimate : float
            Measured or nominal energy per information bit (used for optimal OOK threshold).

        Returns
        -------
        DemodulationResult
            Decoded bits and decision statistics.
        """
        scheme_upper = scheme.upper()
        L = self.mod.L

        if scheme_upper == "BPSK":
            # Segment into bit intervals of length L
            num_bits = len(noisy_waveform) // L
            segments = noisy_waveform[: num_bits * L].reshape(num_bits, L)
            # Correlate with orthonormal basis phi_cos
            decision_stats = segments @ self.mod.phi_cos  # Shape: (num_bits,)
            # Decision threshold at 0.0
            rx_bits = (decision_stats > 0.0).astype(np.int8)
            stats_2d = np.column_stack([decision_stats, np.zeros_like(decision_stats)])
            return DemodulationResult(rx_bits=rx_bits[:num_info_bits], decision_statistics=stats_2d, threshold=0.0)

        elif scheme_upper == "OOK":
            num_bits = len(noisy_waveform) // L
            segments = noisy_waveform[: num_bits * L].reshape(num_bits, L)
            decision_stats = segments @ self.mod.phi_cos
            # Optimal threshold for equiprobable OOK is halfway between 0 and sqrt(2*Eb)
            threshold = 0.5 * np.sqrt(2.0 * eb_estimate)
            rx_bits = (decision_stats > threshold).astype(np.int8)
            stats_2d = np.column_stack([decision_stats, np.zeros_like(decision_stats)])
            return DemodulationResult(rx_bits=rx_bits[:num_info_bits], decision_statistics=stats_2d, threshold=threshold)

        elif scheme_upper == "BFSK":
            num_bits = len(noisy_waveform) // L
            segments = noisy_waveform[: num_bits * L].reshape(num_bits, L)
            # Correlate with tone 1 basis and tone 2 basis
            r1 = segments @ self.mod.phi_fsk1
            r2 = segments @ self.mod.phi_fsk2
            # Decision: tone 1 (bit 1) if r1 > r2, else tone 2 (bit 0)
            rx_bits = (r1 > r2).astype(np.int8)
            stats_2d = np.column_stack([r1, r2])
            return DemodulationResult(rx_bits=rx_bits[:num_info_bits], decision_statistics=stats_2d, threshold=0.0)

        elif scheme_upper == "QPSK":
            symbol_len = 2 * L
            num_symbols = len(noisy_waveform) // symbol_len
            segments = noisy_waveform[: num_symbols * symbol_len].reshape(num_symbols, symbol_len)

            # Correlate with I and Q orthonormal bases
            r_i = segments @ self.mod.phi_qpsk_i
            r_q = segments @ self.mod.phi_qpsk_q

            # Gray demapping:
            # r_i > 0 -> b0 = 0, r_i < 0 -> b0 = 1
            # r_q > 0 -> b1 = 0, r_q < 0 -> b1 = 1
            b0 = (r_i < 0.0).astype(np.int8)
            b1 = (r_q < 0.0).astype(np.int8)

            rx_bits = np.column_stack([b0, b1]).ravel()
            stats_2d = np.column_stack([r_i, r_q])
            return DemodulationResult(rx_bits=rx_bits[:num_info_bits], decision_statistics=stats_2d, threshold=0.0)

        else:
            raise ValueError(f"Unsupported scheme: {scheme}")


def simulate_passband_link(
    bits: np.ndarray,
    scheme: SchemeType,
    eb_n0_db: float,
    samples_per_bit: int = 32,
    carrier_cycles_per_bit: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[float, DemodulationResult, ModulatedSignal, float, float]:
    """End-to-end simulation of passband transmission through AWGN channel.

    Returns
    -------
    ber : float
        Measured Bit Error Rate.
    demod_res : DemodulationResult
        Demodulated bits and decision statistics.
    mod_sig : ModulatedSignal
        Transmitted waveform.
    eb : float
        Measured energy per info bit.
    n0 : float
        Noise spectral density.
    """
    modulator = PassbandModulator(
        samples_per_bit=samples_per_bit,
        carrier_cycles_per_bit=carrier_cycles_per_bit,
    )
    demodulator = PassbandDemodulator(modulator)

    mod_sig = modulator.modulate(bits, scheme)
    noisy_wave, eb, n0 = add_awgn_noise(
        mod_sig.waveform,
        eb_n0_db=eb_n0_db,
        num_info_bits=len(bits),
        rng=rng,
    )
    demod_res = demodulator.demodulate(
        noisy_wave,
        scheme=scheme,
        num_info_bits=len(bits),
        eb_estimate=eb,
    )
    ber, _, _ = calculate_ber(bits, demod_res.rx_bits)
    return ber, demod_res, mod_sig, eb, n0

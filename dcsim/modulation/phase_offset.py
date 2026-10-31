"""Carrier Phase Offset Impairment and Demodulation Degradation.

Owner: Student 2 (Module B: Modulation and Detection Lab)

Covers:
- Carrier phase error phi in [-180 deg, +180 deg].
- Constellation rotation in complex signal space.
- Bit error rate degradation as a function of phase error phi.
- Analytical BPSK theoretical benchmark with phase offset: Pb = Q( sqrt(2*Eb/N0) * cos(phi) ).
"""

from __future__ import annotations
import numpy as np
from dcsim.core.conventions import q_function, add_awgn_noise, calculate_ber


def theoretical_bpsk_phase_offset_ber(eb_n0_db: float, phi_rad: float | np.ndarray) -> np.ndarray:
    """Theoretical BER for BPSK under coherent receiver carrier phase offset phi.

    Formula:
        Pb = Q( sqrt(2 * Eb/N0) * cos(phi) )

    Parameters
    ----------
    eb_n0_db : float
        Eb/N0 in dB.
    phi_rad : float or np.ndarray
        Carrier phase error in radians.

    Returns
    -------
    np.ndarray
        Bit error probability.
    """
    gamma_b = 10.0 ** (eb_n0_db / 10.0)
    cos_phi = np.cos(np.asarray(phi_rad, dtype=np.float64))
    arg = np.sqrt(2.0 * gamma_b) * cos_phi
    return q_function(arg)


def apply_phase_offset(complex_symbols: np.ndarray, phi_rad: float) -> np.ndarray:
    """Rotate complex symbols by carrier phase offset phi.

    r = s * exp(j * phi)
    """
    return complex_symbols * np.exp(1j * phi_rad)


def simulate_phase_offset_impact(
    scheme: str,
    eb_n0_db: float,
    phi_deg: float,
    num_bits: int = 40000,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Simulate transmission of BPSK or QPSK under carrier phase error phi.

    Returns
    -------
    dict
        Simulated BER, theoretical BER, clean symbols, rotated noisy symbols, and phase sweep curves.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    phi_rad = np.radians(phi_deg)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)

    scheme_upper = scheme.upper()

    if scheme_upper == "BPSK":
        # Baseband BPSK: +1.0 for bit 1, -1.0 for bit 0
        symbols = np.where(bits == 1, 1.0, -1.0).astype(np.complex128)
        # Apply phase offset before or during transmission
        rotated = apply_phase_offset(symbols, phi_rad)
        noisy, eb, n0 = add_awgn_noise(rotated, eb_n0_db, num_bits, rng=rng)

        # Standard coherent receiver assuming phi = 0 slices real axis
        rx_bits = (np.real(noisy) > 0.0).astype(np.int8)
        ber, errors, _ = calculate_ber(bits, rx_bits)
        theo_ber = float(theoretical_bpsk_phase_offset_ber(eb_n0_db, phi_rad))

    elif scheme_upper == "QPSK":
        # QPSK Gray mapped pairs
        if len(bits) % 2 != 0:
            bits = np.append(bits, 0)
        quads = bits.reshape(-1, 2)
        # 00 -> (+1, +1)/sqrt(2), etc.
        a_i = np.where(quads[:, 0] == 0, 1.0, -1.0) / np.sqrt(2.0)
        a_q = np.where(quads[:, 1] == 0, 1.0, -1.0) / np.sqrt(2.0)
        symbols = a_i + 1j * a_q

        rotated = apply_phase_offset(symbols, phi_rad)
        noisy, eb, n0 = add_awgn_noise(rotated, eb_n0_db, len(bits), rng=rng)

        # Slice real and imag
        dec_i = (np.real(noisy) < 0.0).astype(np.int8)
        dec_q = (np.imag(noisy) < 0.0).astype(np.int8)
        rx_bits = np.column_stack([dec_i, dec_q]).ravel()
        ber, errors, _ = calculate_ber(bits, rx_bits)
        # Approximate QPSK under small phase error: projection onto I/Q axes
        theo_ber = float(theoretical_bpsk_phase_offset_ber(eb_n0_db, phi_rad))

    else:
        raise ValueError(f"Phase offset study implemented for BPSK and QPSK, got {scheme}")

    return {
        "ber_sim": ber,
        "ber_theo": theo_ber,
        "phi_deg": phi_deg,
        "phi_rad": phi_rad,
        "tx_symbols": symbols,
        "rx_symbols": noisy,
        "num_bits": len(bits),
    }

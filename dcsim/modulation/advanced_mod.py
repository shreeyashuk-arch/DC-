"""Advanced Digital Modulation Schemes in Complex Baseband.

Owner: Student 2 (Module B: Modulation and Detection Lab)

Implements:
1. 8-PSK (Gray-coded constellation, ML detection, theoretical bound).
2. 16-QAM (Gray-coded square constellation, ML detection, theoretical bound).
3. DBPSK (Non-coherent Differential BPSK, delay-line product detector, exact theoretical Pb).
4. BFSK (Non-coherent orthogonal tone envelope detector, exact theoretical Pb).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from dcsim.core.conventions import q_function, add_awgn_noise, calculate_ber, measure_eb

AdvancedSchemeType = Literal["8-PSK", "16-QAM", "DBPSK", "Non-coherent BFSK"]

# 8-PSK Gray mapping table: phase index k in {0..7} -> 3 bits
# Order of Gray code: 000, 001, 011, 010, 110, 111, 101, 100
GRAY_8PSK_BITS_TO_IDX = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 1): 2,
    (0, 1, 0): 3,
    (1, 1, 0): 4,
    (1, 1, 1): 5,
    (1, 0, 1): 6,
    (1, 0, 0): 7,
}
GRAY_8PSK_IDX_TO_BITS = {idx: np.array(bits, dtype=np.int8) for bits, idx in GRAY_8PSK_BITS_TO_IDX.items()}

# 16-QAM 2-bit Gray map for I and Q coordinates
GRAY_4PAM_BITS_TO_LEVEL = {
    (0, 0): -3.0,
    (0, 1): -1.0,
    (1, 1): 1.0,
    (1, 0): 3.0,
}
LEVELS_4PAM = np.array([-3.0, -1.0, 1.0, 3.0])
GRAY_4PAM_LEVEL_TO_BITS = {
    -3.0: np.array([0, 0], dtype=np.int8),
    -1.0: np.array([0, 1], dtype=np.int8),
    1.0: np.array([1, 1], dtype=np.int8),
    3.0: np.array([1, 0], dtype=np.int8),
}


def theoretical_advanced_ber(scheme: AdvancedSchemeType, eb_n0_db: float | np.ndarray) -> np.ndarray:
    """Calculate theoretical Bit Error Rate for advanced modulation schemes.

    Formulas:
      - 8-PSK:             Pb ~ (2 / log2(M)) * Q( sqrt(2 * log2(M) * Eb/N0) * sin(pi/M) )
      - 16-QAM:           Pb ~ 0.75 * Q( sqrt(0.8 * Eb/N0) )
      - DBPSK:            Pb = 0.5 * exp(-Eb/N0)
      - Non-coherent FSK: Pb = 0.5 * exp(-Eb / (2*N0))
    """
    gamma_b = 10.0 ** (np.asarray(eb_n0_db, dtype=np.float64) / 10.0)
    scheme_clean = scheme.strip().upper()

    if scheme_clean in ("8-PSK", "8PSK"):
        M = 8.0
        k = np.log2(M)
        arg = np.sqrt(2.0 * k * gamma_b) * np.sin(np.pi / M)
        return (2.0 / k) * q_function(arg)

    elif scheme_clean in ("16-QAM", "16QAM"):
        return 0.75 * q_function(np.sqrt(0.8 * gamma_b))

    elif scheme_clean == "DBPSK":
        return 0.5 * np.exp(-gamma_b)

    elif "BFSK" in scheme_clean or "NON-COHERENT" in scheme_clean:
        return 0.5 * np.exp(-0.5 * gamma_b)

    else:
        raise ValueError(f"Unknown scheme: {scheme}")


# ---------------- 8-PSK ----------------
def modulate_8psk(bits: np.ndarray) -> tuple[np.ndarray, int]:
    """Modulate binary bits into complex baseband 8-PSK symbols with Gray coding."""
    b = np.asarray(bits, dtype=np.int8).ravel()
    orig_len = len(b)
    rem = orig_len % 3
    if rem != 0:
        b = np.append(b, np.zeros(3 - rem, dtype=np.int8))

    triplets = b.reshape(-1, 3)
    symbols = np.zeros(len(triplets), dtype=np.complex128)

    for i, t in enumerate(triplets):
        idx = GRAY_8PSK_BITS_TO_IDX[tuple(t)]
        theta = 2.0 * np.pi * idx / 8.0
        symbols[i] = np.exp(1j * theta)

    return symbols, orig_len


def demodulate_8psk(symbols: np.ndarray, num_info_bits: int) -> np.ndarray:
    """ML Demodulation of 8-PSK symbols to binary bits."""
    phases = np.angle(symbols) % (2.0 * np.pi)
    # 8 constellation angles at 2*pi*k / 8
    # Nearest integer sector
    idx = np.round(phases / (2.0 * np.pi / 8.0)).astype(int) % 8

    recovered_bits = np.zeros((len(symbols), 3), dtype=np.int8)
    for i, k in enumerate(idx):
        recovered_bits[i] = GRAY_8PSK_IDX_TO_BITS[k]

    return recovered_bits.ravel()[:num_info_bits]


# ---------------- 16-QAM ----------------
def modulate_16qam(bits: np.ndarray) -> tuple[np.ndarray, int]:
    """Modulate binary bits into 16-QAM complex symbols with Gray coding."""
    b = np.asarray(bits, dtype=np.int8).ravel()
    orig_len = len(b)
    rem = orig_len % 4
    if rem != 0:
        b = np.append(b, np.zeros(4 - rem, dtype=np.int8))

    quads = b.reshape(-1, 4)
    num_symbols = len(quads)
    symbols = np.zeros(num_symbols, dtype=np.complex128)

    # Scale to normalize average symbol energy to 1.0 (mean(|s|^2) = 1.0)
    # Unscaled 16-QAM grid: {-3, -1, 1, 3} on I and Q -> E_s = 10.0
    scale = 1.0 / np.sqrt(10.0)

    for i, q in enumerate(quads):
        i_lvl = GRAY_4PAM_BITS_TO_LEVEL[(q[0], q[1])]
        q_lvl = GRAY_4PAM_BITS_TO_LEVEL[(q[2], q[3])]
        symbols[i] = (i_lvl + 1j * q_lvl) * scale

    return symbols, orig_len


def demodulate_16qam(symbols: np.ndarray, num_info_bits: int) -> np.ndarray:
    """Independent I/Q slicing ML detector for Gray-coded 16-QAM."""
    scale = np.sqrt(10.0)
    scaled_syms = symbols * scale

    i_parts = np.real(scaled_syms)
    q_parts = np.imag(scaled_syms)

    # Decision boundaries for levels [-3, -1, 1, 3] are at -2, 0, +2
    def slice_4pam(val: np.ndarray) -> np.ndarray:
        levels = np.zeros_like(val)
        levels[val < -2.0] = -3.0
        levels[(val >= -2.0) & (val < 0.0)] = -1.0
        levels[(val >= 0.0) & (val < 2.0)] = 1.0
        levels[val >= 2.0] = 3.0
        return levels

    i_dec = slice_4pam(i_parts)
    q_dec = slice_4pam(q_parts)

    out_bits = np.zeros((len(symbols), 4), dtype=np.int8)
    for k in range(len(symbols)):
        out_bits[k, 0:2] = GRAY_4PAM_LEVEL_TO_BITS[i_dec[k]]
        out_bits[k, 2:4] = GRAY_4PAM_LEVEL_TO_BITS[q_dec[k]]

    return out_bits.ravel()[:num_info_bits]


# ---------------- DBPSK (Non-coherent) ----------------
def modulate_dbpsk(bits: np.ndarray) -> np.ndarray:
    """Differentially encode binary bits into DBPSK symbols.

    Mapping:
        bit 0 -> phase change 0 (s_k = s_{k-1})
        bit 1 -> phase change pi (s_k = -s_{k-1})
    Initial reference symbol: s_0 = +1.0
    """
    b = np.asarray(bits, dtype=np.int8).ravel()
    num_symbols = len(b) + 1
    symbols = np.zeros(num_symbols, dtype=np.complex128)
    symbols[0] = 1.0 + 0j

    curr_phase = 0.0
    for i, bit in enumerate(b):
        if bit == 1:
            curr_phase += np.pi
        symbols[i + 1] = np.exp(1j * curr_phase)

    return symbols


def demodulate_dbpsk(symbols: np.ndarray, num_info_bits: int) -> np.ndarray:
    """Non-coherent delay-product detection of DBPSK symbols.

    Computes Re{ r_k * conj(r_{k-1}) }:
        > 0 -> bit 0
        < 0 -> bit 1
    """
    # Delay-multiply consecutive symbols
    product = symbols[1:] * np.conj(symbols[:-1])
    rx_bits = (np.real(product) < 0.0).astype(np.int8)
    return rx_bits[:num_info_bits]


# ---------------- Non-coherent BFSK ----------------
def modulate_noncoherent_bfsk(bits: np.ndarray) -> np.ndarray:
    """Synthesize orthogonal 2-channel baseband symbols for non-coherent BFSK.

    Bit 1 -> [1.0, 0.0]  (Tone 1 active)
    Bit 0 -> [0.0, 1.0]  (Tone 2 active)
    """
    b = np.asarray(bits, dtype=np.int8).ravel()
    # Output shape: (num_bits, 2)
    symbols = np.zeros((len(b), 2), dtype=np.complex128)
    symbols[b == 1, 0] = 1.0
    symbols[b == 0, 1] = 1.0
    return symbols


def demodulate_noncoherent_bfsk(noisy_channels: np.ndarray) -> np.ndarray:
    """Envelope detection for non-coherent BFSK across the two orthogonal channels.

    Decides bit 1 if |r1| > |r2|, else bit 0.
    """
    env1 = np.abs(noisy_channels[:, 0])
    env2 = np.abs(noisy_channels[:, 1])
    return (env1 > env2).astype(np.int8)


# ---------------- Master Simulator ----------------
def simulate_advanced_mod(
    bits: np.ndarray,
    scheme: AdvancedSchemeType,
    eb_n0_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Simulate advanced complex baseband modulation over AWGN channel.

    Returns
    -------
    ber : float
        Bit error rate.
    rx_bits : np.ndarray
        Demodulated bits.
    symbols_for_plot : np.ndarray
        Received complex symbols for constellation diagram.
    eb : float
        Measured energy per info bit.
    n0 : float
        Noise spectral density.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_info_bits = len(bits)
    scheme_clean = scheme.strip().upper()

    if scheme_clean in ("8-PSK", "8PSK"):
        tx_symbols, _ = modulate_8psk(bits)
        noisy_syms, eb, n0 = add_awgn_noise(tx_symbols, eb_n0_db, n_info_bits, rng=rng)
        rx_bits = demodulate_8psk(noisy_syms, n_info_bits)
        plot_syms = noisy_syms

    elif scheme_clean in ("16-QAM", "16QAM"):
        tx_symbols, _ = modulate_16qam(bits)
        noisy_syms, eb, n0 = add_awgn_noise(tx_symbols, eb_n0_db, n_info_bits, rng=rng)
        rx_bits = demodulate_16qam(noisy_syms, n_info_bits)
        plot_syms = noisy_syms

    elif scheme_clean == "DBPSK":
        tx_symbols = modulate_dbpsk(bits)
        # Note: tx_symbols has N+1 elements for N bits
        noisy_syms, eb, n0 = add_awgn_noise(tx_symbols, eb_n0_db, n_info_bits, rng=rng)
        rx_bits = demodulate_dbpsk(noisy_syms, n_info_bits)
        plot_syms = noisy_syms[1:] * np.conj(noisy_syms[:-1])

    elif "BFSK" in scheme_clean or "NON-COHERENT" in scheme_clean:
        tx_channels = modulate_noncoherent_bfsk(bits)
        # Flatten for noise addition then restore
        flat_tx = tx_channels.ravel()
        noisy_flat, eb, n0 = add_awgn_noise(flat_tx, eb_n0_db, n_info_bits, rng=rng)
        noisy_channels = noisy_flat.reshape(len(bits), 2)
        rx_bits = demodulate_noncoherent_bfsk(noisy_channels)
        # 2D envelope decision points
        plot_syms = noisy_channels[:, 0] + 1j * noisy_channels[:, 1]

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    ber, _, _ = calculate_ber(bits, rx_bits)
    return ber, rx_bits, plot_syms, eb, n0

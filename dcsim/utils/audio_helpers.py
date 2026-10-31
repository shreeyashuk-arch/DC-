"""Audio synthesis, metrics, and conversion utilities."""

from __future__ import annotations
import numpy as np


def compute_audio_snr(orig: np.ndarray, rx: np.ndarray) -> float:
    """Compute empirical Signal-to-Noise Ratio (SNR) in dB between two audio arrays."""
    a1 = np.asarray(orig, dtype=np.float64).ravel()
    a2 = np.asarray(rx, dtype=np.float64).ravel()

    min_len = min(len(a1), len(a2))
    if min_len == 0:
        return 0.0

    a1 = a1[:min_len]
    a2 = a2[:min_len]

    sig_pow = float(np.mean(a1**2))
    err_pow = float(np.mean((a1 - a2) ** 2))

    if err_pow <= 1e-15:
        return 100.0
    if sig_pow <= 1e-15:
        return -100.0

    return 10.0 * np.log10(sig_pow / err_pow)

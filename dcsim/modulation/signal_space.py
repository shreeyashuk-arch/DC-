"""Signal-Space Analysis and First-Principles Gram-Schmidt Orthogonalization.

Owner: Student 2 (Module B: Modulation and Detection Lab)

Implements:
1. Pure first-principles Gram-Schmidt Orthogonalization Procedure from scratch.
2. Orthonormal basis extraction and basis dimensionality determination.
3. Signal vector coordinates s_ij = <s_i, phi_j>.
4. Signal energy E_i = ||s_i||^2 and pairwise Euclidean distance matrix D_ij = ||s_i - s_j||.
5. Signal reconstruction verification: s_i_hat = sum_j s_ij * phi_j.
6. Numerical orthonormality verification: <phi_i, phi_j> = delta_ij.
7. Presets: Orthogonal BFSK, QPSK, 8-PSK, 16-QAM, Walsh-Hadamard.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np


@dataclass
class GramSchmidtResult:
    """Encapsulates the mathematical output of the Gram-Schmidt procedure."""
    basis_functions: np.ndarray       # Shape: (D, N) - D orthonormal basis vectors
    coordinates: np.ndarray             # Shape: (M, D) - Signal coordinates in basis
    reconstructed_signals: np.ndarray   # Shape: (M, N) - Reconstructed waveforms
    signal_energies: np.ndarray         # Shape: (M,)   - Energy of each signal
    distance_matrix: np.ndarray         # Shape: (M, M) - Pairwise Euclidean distances
    orthonormality_matrix: np.ndarray   # Shape: (D, D) - Inner products <phi_i, phi_j>
    dimension: int                      # D <= M
    max_reconstruction_error: float     # Maximum deviation ||s_i - s_hat_i||


def gram_schmidt_orthogonalize(
    signals: np.ndarray,
    energy_threshold: float = 1e-7,
) -> GramSchmidtResult:
    """Execute Gram-Schmidt Orthogonalization from first principles on M discrete signals.

    Given signals s_1, s_2, ..., s_M each of length N:
      1. psi_1 = s_1,  phi_1 = psi_1 / ||psi_1||
      2. For k = 2..M:
           psi_k = s_k - sum_{j=1}^{d} <s_k, phi_j> * phi_j
           If ||psi_k||^2 > energy_threshold:
               phi_{d+1} = psi_k / ||psi_k||
               d = d + 1

    Parameters
    ----------
    signals : np.ndarray
        Array of shape (M, N) containing M discrete waveforms of length N.
    energy_threshold : float
        Numerical zero threshold for detecting linear dependence.

    Returns
    -------
    GramSchmidtResult
        Complete orthonormal basis, coordinates, metrics, and reconstruction checks.
    """
    S = np.asarray(signals, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    M, N = S.shape

    basis_list: list[np.ndarray] = []

    for k in range(M):
        vec = S[k].copy()
        # Subtract projections onto all existing orthonormal basis functions
        for phi in basis_list:
            proj_coeff = float(np.dot(S[k], phi))
            vec -= proj_coeff * phi

        residual_energy = float(np.dot(vec, vec))
        if residual_energy > energy_threshold:
            # Independent direction found; normalize to unit energy
            phi_new = vec / np.sqrt(residual_energy)
            basis_list.append(phi_new)

    D = len(basis_list)
    if D == 0:
        # All signals were identically zero
        basis_array = np.zeros((1, N), dtype=np.float64)
        D = 1
    else:
        basis_array = np.array(basis_list)  # Shape: (D, N)

    # Compute signal coordinates s_ij = <s_i, phi_j>
    coords = S @ basis_array.T  # Shape: (M, D)

    # Reconstruct signals from basis
    reconstructed = coords @ basis_array  # Shape: (M, N)
    max_recon_err = float(np.max(np.abs(S - reconstructed)))

    # Compute signal energies: E_i = ||s_i||^2
    energies = np.sum(S**2, axis=1)

    # Compute pairwise Euclidean distance matrix in D-dimensional signal space
    # D_ik = ||s_i - s_k||
    diff = coords[:, None, :] - coords[None, :, :]  # Shape: (M, M, D)
    dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

    # Numerical orthonormality verification: <phi_i, phi_j> should equal identity matrix I_D
    ortho_matrix = basis_array @ basis_array.T

    return GramSchmidtResult(
        basis_functions=basis_array,
        coordinates=coords,
        reconstructed_signals=reconstructed,
        signal_energies=energies,
        distance_matrix=dist_matrix,
        orthonormality_matrix=ortho_matrix,
        dimension=D,
        max_reconstruction_error=max_recon_err,
    )


def compute_signal_space_parameters(res: GramSchmidtResult) -> dict[str, float]:
    """Compute aggregate signal-space parameters (E_avg, d_min, dimensionality)."""
    energies = res.signal_energies
    dist_mat = res.distance_matrix
    M = len(energies)

    # d_min: minimum non-zero pairwise distance
    if M > 1:
        # Mask diagonal
        masked = np.where(np.eye(M, dtype=bool), np.inf, dist_mat)
        d_min = float(np.min(masked))
    else:
        d_min = 0.0

    return {
        "num_signals": float(M),
        "dimension": float(res.dimension),
        "avg_energy": float(np.mean(energies)),
        "max_energy": float(np.max(energies)),
        "min_energy": float(np.min(energies)),
        "min_euclidean_distance": d_min,
        "max_reconstruction_error": res.max_reconstruction_error,
    }


def get_signal_space_preset(preset_name: str, N: int = 128) -> tuple[np.ndarray, list[str]]:
    """Generate preset signal sets for interactive demonstration and laboratory testing.

    Supported presets:
      - 'BFSK': 2 orthogonal harmonic tones (D=2)
      - 'QPSK': 4 phase-shifted carrier waveforms (D=2)
      - '8-PSK': 8 phase-shifted carrier waveforms (D=2)
      - '16-QAM': 16 quadrature amplitude waveforms (D=2)
      - 'Walsh-Hadamard': 4 mutually orthogonal Walsh sequences (D=4)
    """
    name = preset_name.strip().upper()
    t = np.linspace(0.0, 1.0, N, endpoint=False)

    if "BFSK" in name:
        # Orthogonal tones: f1 = 2 Hz, f2 = 4 Hz
        s1 = np.cos(2.0 * np.pi * 2.0 * t)
        s2 = np.cos(2.0 * np.pi * 4.0 * t)
        signals = np.array([s1, s2])
        labels = ["s1(t) = cos(4πt) [Bit 1]", "s2(t) = cos(8πt) [Bit 0]"]

    elif "QPSK" in name:
        # 4 signals at 45, 135, 225, 315 deg
        signals = []
        labels = []
        angles = [np.pi / 4.0, 3.0 * np.pi / 4.0, 5.0 * np.pi / 4.0, 7.0 * np.pi / 4.0]
        for i, theta in enumerate(angles):
            sig = np.cos(2.0 * np.pi * 3.0 * t + theta)
            signals.append(sig)
            labels.append(f"s_{i+1}(t) [Phase {int(np.degrees(theta))}°]")
        signals = np.array(signals)

    elif "8-PSK" in name or "8PSK" in name:
        signals = []
        labels = []
        for i in range(8):
            theta = 2.0 * np.pi * i / 8.0
            sig = np.cos(2.0 * np.pi * 3.0 * t + theta)
            signals.append(sig)
            labels.append(f"s_{i}(t) [{int(np.degrees(theta))}°]")
        signals = np.array(signals)

    elif "16-QAM" in name or "16QAM" in name:
        # 16-QAM rectangular grid using I and Q carriers
        signals = []
        labels = []
        c_i = np.cos(2.0 * np.pi * 3.0 * t)
        c_q = -np.sin(2.0 * np.pi * 3.0 * t)
        levels = [-3.0, -1.0, 1.0, 3.0]
        count = 0
        for ai in levels:
            for aq in levels:
                sig = ai * c_i + aq * c_q
                signals.append(sig)
                labels.append(f"QAM-16 ({int(ai)}, {int(aq)})")
                count += 1
        signals = np.array(signals)

    elif "WALSH" in name or "HADAMARD" in name:
        # Order-4 Hadamard matrix
        H4 = np.array([
            [1.0,  1.0,  1.0,  1.0],
            [1.0, -1.0,  1.0, -1.0],
            [1.0,  1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0,  1.0],
        ])
        signals = np.repeat(H4, N // 4, axis=1)
        labels = [f"Walsh Code W{i}" for i in range(4)]

    else:
        raise ValueError(f"Unknown preset: {preset_name}")

    return signals, labels

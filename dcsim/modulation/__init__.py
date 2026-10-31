"""Module B: Modulation, Detection, Signal-Space, and Impairments Lab (Owner: Student 2).

Implements:
- Advanced Modulation: 8-PSK (Gray), 16-QAM (Gray), Non-coherent DBPSK, Non-coherent BFSK.
- Signal-Space Analysis: First-principles Gram-Schmidt Orthogonalization, basis coordinates,
  signal energy, and pairwise Euclidean distance matrices.
- Phase Impairments: Carrier phase offset simulation and BPSK theoretical degradation Q(sqrt(2Eb/N0)*cos(phi)).
- Modulation Tradeoff Dashboard: Spectral efficiency, SNR thresholds for BER=1e-4, and selection guidelines.
"""

from .advanced_mod import (
    modulate_8psk,
    demodulate_8psk,
    modulate_16qam,
    demodulate_16qam,
    modulate_dbpsk,
    demodulate_dbpsk,
    modulate_noncoherent_bfsk,
    demodulate_noncoherent_bfsk,
    simulate_advanced_mod,
    theoretical_advanced_ber,
)
from .signal_space import (
    gram_schmidt_orthogonalize,
    compute_signal_space_parameters,
    get_signal_space_preset,
    GramSchmidtResult,
)
from .phase_offset import (
    apply_phase_offset,
    simulate_phase_offset_impact,
    theoretical_bpsk_phase_offset_ber,
)
from .comparison import (
    get_modulation_comparison_metrics,
    get_scheme_recommendation,
    evaluate_all_schemes_ber,
)

__all__ = [
    "modulate_8psk",
    "demodulate_8psk",
    "modulate_16qam",
    "demodulate_16qam",
    "modulate_dbpsk",
    "demodulate_dbpsk",
    "modulate_noncoherent_bfsk",
    "demodulate_noncoherent_bfsk",
    "simulate_advanced_mod",
    "theoretical_advanced_ber",
    "gram_schmidt_orthogonalize",
    "compute_signal_space_parameters",
    "get_signal_space_preset",
    "GramSchmidtResult",
    "apply_phase_offset",
    "simulate_phase_offset_impact",
    "theoretical_bpsk_phase_offset_ber",
    "get_modulation_comparison_metrics",
    "get_scheme_recommendation",
    "evaluate_all_schemes_ber",
]

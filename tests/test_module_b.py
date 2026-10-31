"""Test suite for Module B: Modulation and Detection Lab (Student 2).

Verifies:
1. Noiseless round-trip for 8-PSK, 16-QAM, DBPSK, Non-coherent BFSK.
2. Statistically sound simulated BER agreement with theoretical benchmarks.
3. Numerical orthonormality and lossless signal reconstruction in Gram-Schmidt tool.
4. Carrier phase offset impact on BPSK constellation and BER.
5. Modulation comparison metrics and recommendation engine.
"""

import numpy as np
import pytest

from dcsim.modulation.advanced_mod import (
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
from dcsim.modulation.signal_space import (
    gram_schmidt_orthogonalize,
    compute_signal_space_parameters,
    get_signal_space_preset,
)
from dcsim.modulation.phase_offset import (
    simulate_phase_offset_impact,
    theoretical_bpsk_phase_offset_ber,
)
from dcsim.modulation.comparison import (
    get_modulation_comparison_metrics,
    get_scheme_recommendation,
)


@pytest.mark.parametrize("scheme", ["8-PSK", "16-QAM", "DBPSK", "Non-coherent BFSK"])
def test_advanced_noiseless_roundtrip(scheme):
    """Verify lossless transmission across all advanced schemes in noiseless channel."""
    rng = np.random.default_rng(2026)
    bits = rng.integers(0, 2, size=300, dtype=np.int8)

    # Use very high SNR (40 dB) to represent noiseless channel
    ber, rx_bits, _, _, _ = simulate_advanced_mod(bits, scheme, eb_n0_db=40.0, rng=rng)  # type: ignore
    assert ber == 0.0
    assert np.array_equal(bits, rx_bits)


@pytest.mark.parametrize("scheme,eb_n0_db,rel_tol", [
    ("8-PSK", 8.0, 0.20),
    ("16-QAM", 9.0, 0.20),
    ("DBPSK", 6.0, 0.20),
    ("Non-coherent BFSK", 7.0, 0.20),
])
def test_advanced_ber_matches_theory(scheme, eb_n0_db, rel_tol):
    """Verify statistical agreement between simulated and theoretical BER."""
    num_bits = 100_000
    rng = np.random.default_rng(101)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)

    ber_sim, _, _, _, _ = simulate_advanced_mod(bits, scheme, eb_n0_db=eb_n0_db, rng=rng)  # type: ignore
    ber_theo = float(theoretical_advanced_ber(scheme, eb_n0_db))  # type: ignore

    rel_error = abs(ber_sim - ber_theo) / ber_theo
    assert rel_error <= rel_tol, (
        f"{scheme} at {eb_n0_db} dB: sim BER={ber_sim:.6f}, "
        f"theo BER={ber_theo:.6f}, rel_err={rel_error:.3f} > {rel_tol}"
    )


def test_gram_schmidt_orthonormality_and_reconstruction():
    """Verify Gram-Schmidt output is numerically orthonormal and reconstructs original signals."""
    for preset in ["BFSK", "QPSK", "Walsh-Hadamard"]:
        signals, _ = get_signal_space_preset(preset, N=64)
        res = gram_schmidt_orthogonalize(signals)

        # 1. Orthonormality check: <phi_i, phi_j> == delta_ij
        D = res.dimension
        expected_identity = np.eye(D)
        assert np.allclose(res.orthonormality_matrix, expected_identity, atol=1e-6), (
            f"Orthonormality violation in {preset}"
        )

        # 2. Exact reconstruction check: s_hat == s
        assert res.max_reconstruction_error < 1e-6, (
            f"Reconstruction error {res.max_reconstruction_error} in {preset}"
        )

        # 3. Dimensionality check
        if preset == "BFSK":
            assert res.dimension == 2
        elif preset == "QPSK":
            assert res.dimension == 2
        elif preset == "Walsh-Hadamard":
            assert res.dimension == 4


def test_carrier_phase_offset():
    """Verify phase offset causes expected BER degradation on BPSK."""
    eb_n0_db = 6.0
    rng = np.random.default_rng(555)

    # At 0 degrees: BER matches nominal theoretical BPSK
    res_0 = simulate_phase_offset_impact("BPSK", eb_n0_db=eb_n0_db, phi_deg=0.0, num_bits=50000, rng=rng)
    assert abs(res_0["ber_sim"] - res_0["ber_theo"]) / res_0["ber_theo"] < 0.20

    # At 90 degrees: complete loss of lock -> BER ~ 0.5
    res_90 = simulate_phase_offset_impact("BPSK", eb_n0_db=eb_n0_db, phi_deg=90.0, num_bits=50000, rng=rng)
    assert pytest.approx(res_90["ber_sim"], abs=0.05) == 0.5


def test_modulation_comparison_table():
    """Verify modulation comparison table completeness and metrics."""
    metrics = get_modulation_comparison_metrics()
    assert len(metrics) == 8
    schemes = [m["Scheme"] for m in metrics]
    assert "BPSK" in schemes
    assert "16-QAM" in schemes

    # Test recommendation engine
    rec_sat = get_scheme_recommendation(bandwidth_limited=True, power_limited=False, phase_jitter_severe=False, constant_envelope_required=True)
    assert rec_sat["recommended_scheme"] == "8-PSK"

    rec_power = get_scheme_recommendation(bandwidth_limited=False, power_limited=True, phase_jitter_severe=False, constant_envelope_required=False)
    assert "BPSK" in rec_power["recommended_scheme"]

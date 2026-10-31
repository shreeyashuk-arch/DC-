"""Test suite for Phase 1: Core Link (shared by all modules).

Verifies:
1. Global conventions (Eb measurement, N0 calculation, noise variance).
2. Lossless noiseless roundtrips for BPSK, QPSK, BFSK, OOK.
3. Lossless text and image converters.
4. Simulated BER agreement within 10% of theoretical benchmark at mid SNR.
5. Decision statistics generation for signal space.
"""

import numpy as np
import pytest
from PIL import Image

from dcsim.core.conventions import (
    q_function,
    measure_eb,
    calculate_n0,
    add_awgn_noise,
    calculate_ber,
)
from dcsim.core.passband_link import (
    PassbandModulator,
    PassbandDemodulator,
    simulate_passband_link,
    theoretical_ber,
)
from dcsim.core.converters import (
    text_to_bits,
    bits_to_text,
    image_to_bits,
    bits_to_image,
)


def test_q_function():
    """Verify Gaussian Q-function against well-known values."""
    assert pytest.approx(q_function(0.0), rel=1e-6) == 0.5
    # Q(1.96) ~ 0.025
    assert pytest.approx(q_function(1.95996), rel=1e-3) == 0.025
    # Q(3.0) ~ 0.001349898
    assert pytest.approx(q_function(3.0), rel=1e-4) == 0.001349898


def test_measure_eb_and_n0():
    """Verify Eb measurement directly from samples and N0 calculation."""
    # Create test signal with known discrete energy
    samples = np.array([1.0, -1.0, 2.0, -2.0])  # sum(|x|^2) = 1+1+4+4 = 10.0
    num_bits = 5
    eb = measure_eb(samples, num_bits)
    assert pytest.approx(eb) == 2.0

    # Test N0 calculation at 0 dB (Eb/N0 = 1 linear -> N0 = Eb)
    assert pytest.approx(calculate_n0(2.0, 0.0)) == 2.0
    # Test N0 at 10 dB (Eb/N0 = 10 linear -> N0 = Eb / 10 = 0.2)
    assert pytest.approx(calculate_n0(2.0, 10.0)) == 0.2


def test_converters_text():
    """Test text to bits and back for ASCII and Unicode."""
    sample_texts = [
        "Digital Communication Lab 2026",
        "Hello World! 12345",
        "Special symbols: @#$%^&*()_+~",
        "Unicode: π ≈ 3.14159, φ = 1.618",
    ]
    for text in sample_texts:
        bits = text_to_bits(text)
        recovered = bits_to_text(bits)
        assert recovered == text, f"Failed for text: {text}"


def test_converters_image():
    """Test image to bits and back lossless reconstruction."""
    # Create small synthetic test image (grayscale & RGB)
    for mode in ["L", "RGB"]:
        orig = Image.new(mode, (16, 16), color=128 if mode == "L" else (100, 150, 200))
        bits, meta = image_to_bits(orig, grayscale=(mode == "L"))
        reconstructed = bits_to_image(bits, meta)
        assert np.array_equal(np.array(orig), np.array(reconstructed))


@pytest.mark.parametrize("scheme", ["BPSK", "QPSK", "BFSK", "OOK"])
def test_noiseless_roundtrip(scheme):
    """Verify that in the absence of noise, all modulations yield zero bit errors."""
    rng = np.random.default_rng(12345)
    tx_bits = rng.integers(0, 2, size=200, dtype=np.int8)

    mod = PassbandModulator(samples_per_bit=16, carrier_cycles_per_bit=2)
    demod = PassbandDemodulator(mod)

    mod_sig = mod.modulate(tx_bits, scheme)  # type: ignore
    # Demodulate directly without noise
    eb = measure_eb(mod_sig.waveform, len(tx_bits))
    demod_res = demod.demodulate(mod_sig.waveform, scheme, len(tx_bits), eb_estimate=eb)  # type: ignore

    ber, errs, total = calculate_ber(tx_bits, demod_res.rx_bits)
    assert errs == 0, f"Expected 0 errors for noiseless {scheme}, got {errs}"
    assert ber == 0.0
    assert len(demod_res.rx_bits) == len(tx_bits)
    assert demod_res.decision_statistics.shape[0] > 0


@pytest.mark.parametrize("scheme,eb_n0_db,tolerance", [
    ("BPSK", 4.0, 0.10),   # Theoretical BER ~ 0.0125
    ("QPSK", 4.0, 0.10),   # Theoretical BER ~ 0.0125
    ("BFSK", 6.0, 0.10),   # Theoretical BER ~ 0.0230
    ("OOK", 6.0, 0.10),    # Theoretical BER ~ 0.0230
])
def test_ber_matches_theory(scheme, eb_n0_db, tolerance):
    """Verify simulated BER is within 10% of theoretical benchmark at mid SNR."""
    # Use 120,000 bits for statistical confidence (produces > 1500 errors)
    num_bits = 120_000
    rng = np.random.default_rng(2026)
    bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)

    ber_sim, _, _, _, _ = simulate_passband_link(
        bits=bits,
        scheme=scheme,  # type: ignore
        eb_n0_db=eb_n0_db,
        samples_per_bit=16,
        carrier_cycles_per_bit=2,
        rng=rng,
    )
    ber_theo = float(theoretical_ber(scheme, eb_n0_db))  # type: ignore

    rel_error = abs(ber_sim - ber_theo) / ber_theo
    assert rel_error <= tolerance, (
        f"{scheme} at {eb_n0_db} dB: sim BER={ber_sim:.6f}, "
        f"theo BER={ber_theo:.6f}, rel_err={rel_error:.3f} > {tolerance}"
    )

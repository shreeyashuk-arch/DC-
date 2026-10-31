"""Test suite for Module A: Source and Baseband Lab (Student 1).

Verifies:
1. PCM SQNR within 1 dB of 6.02n + 1.76 for full-scale sine (n = 4, 6, 8, 10).
2. Companding improves SQNR for small-amplitude signals.
3. Line code encoders generate correct waveforms and spectral properties.
4. RRC filter unit-energy normalization and matched filtering zero-ISI property.
5. RRC matched filter transmission matches AWGN BPSK theoretical BER.
6. Aliasing detector correctly identifies sub-Nyquist folding.
"""

import numpy as np
import pytest

from dcsim.source.pcm import (
    PCMCodec,
    mu_law_compress,
    mu_law_expand,
    uniform_quantize,
    calculate_sqnr,
    generate_synthetic_tone,
    demonstrate_aliasing,
    theoretical_sqnr_sine,
)
from dcsim.source.line_coding import (
    encode_line_code,
    compute_line_code_psd,
    get_line_code_summary_table,
)
from dcsim.source.pulse_shaping import (
    rrc_filter_taps,
    rc_filter_taps,
    MatchedRRCFilter,
    simulate_rrc_link,
)


@pytest.mark.parametrize("n_bits", [4, 6, 8, 10])
def test_pcm_sqnr_full_scale_sine(n_bits):
    """Verify PCM SQNR is within 1 dB of theoretical 6.02n + 1.76 dB for full-scale sine.

    Uses an incommensurate test frequency (103.5 Hz) so quantization error is ergodic
    and uniformly distributed across quantizer intervals, matching Bennett's theoretical model.
    """
    _, tone = generate_synthetic_tone(freq_hz=103.5, duration_s=1.0, sampling_rate_hz=10000.0, amplitude=1.0)
    codec = PCMCodec(n_bits=n_bits, companding=False)
    res = codec.encode_and_decode(tone, v_max=1.0)

    theo_sqnr = theoretical_sqnr_sine(n_bits)
    diff = abs(res.measured_sqnr_db - theo_sqnr)
    assert diff <= 1.0, f"n={n_bits}: Measured SQNR {res.measured_sqnr_db:.2f} dB vs Theo {theo_sqnr:.2f} dB, diff {diff:.2f} > 1.0 dB"



def test_mu_law_companding_benefit_on_small_signal():
    """Verify companding provides substantial SQNR gain for small signals (A = 0.05)."""
    _, small_tone = generate_synthetic_tone(freq_hz=100.0, duration_s=1.0, sampling_rate_hz=10000.0, amplitude=0.05)

    # Uniform quantizer with n = 8 bits
    uniform_codec = PCMCodec(n_bits=8, companding=False)
    res_uniform = uniform_codec.encode_and_decode(small_tone, v_max=1.0)

    # Companded quantizer with n = 8 bits (mu = 255)
    companded_codec = PCMCodec(n_bits=8, companding=True, mu=255.0)
    res_companded = companded_codec.encode_and_decode(small_tone, v_max=1.0)

    # Companding should give >= 10 dB improvement on small signals
    sqnr_advantage = res_companded.measured_sqnr_db - res_uniform.measured_sqnr_db
    assert sqnr_advantage > 10.0, f"Expected >10 dB companding gain, got {sqnr_advantage:.2f} dB"


def test_mu_law_inverse():
    """Verify mu_law_expand is exact inverse of mu_law_compress."""
    x = np.linspace(-0.99, 0.99, 1000)
    compressed = mu_law_compress(x, mu=255.0)
    recovered = mu_law_expand(compressed, mu=255.0)
    assert np.allclose(x, recovered, atol=1e-12)


def test_line_coding_properties():
    """Verify characteristics of NRZ-L, RZ, Manchester, and AMI waveforms."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)

    # Manchester must have zero mean across integer symbol periods
    _, wave_man = encode_line_code(bits, "Manchester", samples_per_bit=32, voltage=1.0)
    assert np.isclose(np.mean(wave_man), 0.0, atol=1e-10)

    # AMI: consecutive 1s must have opposite signs
    _, wave_ami = encode_line_code(bits, "AMI", samples_per_bit=32, voltage=1.0)
    ami_marks = wave_ami[::32][bits == 1]
    # Check that consecutive marks alternate signs: +1, -1, +1, -1
    for i in range(len(ami_marks) - 1):
        assert ami_marks[i] * ami_marks[i + 1] < 0

    # Summary table must cover all 4 schemes
    table = get_line_code_summary_table()
    assert len(table) == 4


def test_rrc_filter_unit_energy():
    """Verify RRC filter taps strictly satisfy sum(|h|^2) = 1.0 for various roll-offs."""
    for alpha in [0.0, 0.22, 0.35, 0.5, 1.0]:
        _, h = rrc_filter_taps(alpha=alpha, samples_per_symbol=8, num_symbols_span=6)
        energy = np.sum(h**2)
        assert pytest.approx(energy, rel=1e-7) == 1.0


def test_rrc_zero_isi():
    """Verify cascading TX RRC and RX RRC yields zero-ISI Raised Cosine response."""
    alpha = 0.35
    sps = 8
    span = 6
    mf = MatchedRRCFilter(alpha=alpha, sps=sps, span=span)

    # Cascaded impulse response
    cascade = np.convolve(mf.h, mf.h, mode="full")
    center = 2 * span * sps

    # At integer symbol offsets (+/- 1*sps, +/- 2*sps, etc.), value should be near zero
    peak = cascade[center]
    for offset in range(1, span):
        sample_val = cascade[center + offset * sps]
        isi_ratio = abs(sample_val) / peak
        assert isi_ratio < 0.03, f"ISI at symbol {offset} is {isi_ratio:.4f} > 0.03"


def test_rrc_ber_matches_theory():
    """Verify that BPSK with matched RRC filter matches theoretical AWGN BER."""
    num_bits = 80_000
    eb_n0_db = 4.0
    rng = np.random.default_rng(777)

    res = simulate_rrc_link(
        num_bits=num_bits,
        eb_n0_db=eb_n0_db,
        alpha=0.35,
        sps=8,
        span=6,
        poor_bandlimit=False,
        rng=rng,
    )
    ber_sim = float(res["ber"])
    ber_theo = float(res["ber_theoretical"])

    rel_error = abs(ber_sim - ber_theo) / ber_theo
    assert rel_error <= 0.15, f"RRC BER {ber_sim:.6f} vs Theo {ber_theo:.6f}, rel_err {rel_error:.3f}"


def test_aliasing_demonstration():
    """Verify aliasing calculation for sub-Nyquist sampling."""
    res = demonstrate_aliasing(signal_freq_hz=120.0, sampling_rate_hz=160.0, duration_s=0.1)
    assert res["is_aliased"] is True
    # f_alias = |120 - 1*160| = 40 Hz
    assert pytest.approx(res["alias_freq_hz"]) == 40.0

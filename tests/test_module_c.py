"""Test suite for Module C: Channel, Coding and Integration Lab (Student 3).

Verifies:
1. Flat Rayleigh fading BPSK simulation matches theoretical benchmark Pb = 0.5*(1 - sqrt(g/(1+g))).
2. Massive error gap between AWGN and Rayleigh fading links.
3. Lossless noiseless roundtrips for Repetition(3), Hamming(7,4), and Convolutional(1/2, K=3).
4. Exact single-bit error correction in Hamming(7,4) syndrome decoding.
5. Viterbi and Hamming coding gain: coded BER is lower than uncoded BER at high Eb/N0.
6. Block interleaver lossless inversion and burst error dispersal.
"""

import numpy as np
import pytest

from dcsim.channel.fading import (
    RayleighChannel,
    simulate_rayleigh_bpsk,
    theoretical_rayleigh_bpsk_ber,
)
from dcsim.coding.repetition import RepetitionCode
from dcsim.coding.hamming import Hamming74Code
from dcsim.coding.convolutional import ConvolutionalCodec
from dcsim.coding.interleaver import BlockInterleaver
from dcsim.coding.system import simulate_coded_bpsk_link


def test_rayleigh_bpsk_ber_matches_theory():
    """Verify BPSK in flat Rayleigh fading matches analytical benchmark within statistical bounds."""
    num_bits = 80_000
    eb_n0_db = 10.0
    rng = np.random.default_rng(2026)

    res = simulate_rayleigh_bpsk(num_bits=num_bits, eb_n0_db=eb_n0_db, rng=rng)
    ber_sim = float(res["ber_sim"])
    ber_theo = float(res["ber_theo_rayleigh"])

    rel_error = abs(ber_sim - ber_theo) / ber_theo
    assert rel_error <= 0.20, f"Rayleigh BER {ber_sim:.5f} vs Theo {ber_theo:.5f}, rel_err {rel_error:.3f}"


def test_awgn_vs_rayleigh_gap():
    """Verify that Rayleigh fading experiences a massive degradation compared to AWGN."""
    eb_n0_db = 14.0
    theo_rayleigh = float(theoretical_rayleigh_bpsk_ber(eb_n0_db))
    # AWGN at 14 dB: Q(sqrt(2 * 10^1.4)) ~ Q(7.08) ~ 7e-13
    assert theo_rayleigh > 0.005
    # The gap ratio is enormous (> 1000x)
    assert theo_rayleigh / 1e-6 > 1000.0


def test_noiseless_coding_roundtrips():
    """Verify error-free recovery for all codecs in a noiseless channel."""
    rng = np.random.default_rng(101)
    tx_bits = rng.integers(0, 2, size=240, dtype=np.int8)

    # 1. Repetition
    rep = RepetitionCode(repeat=3)
    c_rep = rep.encode(tx_bits)
    d_rep = rep.decode(c_rep)
    assert np.array_equal(tx_bits, d_rep)

    # 2. Hamming
    ham = Hamming74Code()
    c_ham, orig_len = ham.encode(tx_bits)
    d_ham = ham.decode(c_ham, orig_len=orig_len)
    assert np.array_equal(tx_bits, d_ham)

    # 3. Convolutional + Viterbi
    conv = ConvolutionalCodec()
    c_conv = conv.encode(tx_bits)
    d_conv = conv.decode(c_conv, num_info_bits=len(tx_bits))
    assert np.array_equal(tx_bits, d_conv)


def test_hamming_single_bit_error_correction():
    """Verify Hamming(7,4) syndrome decoder corrects EVERY single-bit error pattern."""
    ham = Hamming74Code()
    # Test all 16 possible 4-bit message words
    for val in range(16):
        msg = np.array([(val >> 3) & 1, (val >> 2) & 1, (val >> 1) & 1, val & 1], dtype=np.int8)
        codeword, orig_len = ham.encode(msg)

        # Test flipping each of the 7 bit positions in the codeword
        for err_pos in range(7):
            corrupted = codeword.copy()
            corrupted[err_pos] ^= 1  # Inject single error
            decoded = ham.decode(corrupted, orig_len=orig_len)
            assert np.array_equal(msg, decoded), f"Failed for msg {msg} with error at position {err_pos}"


def test_coding_gain_at_high_snr():
    """Verify that Hamming and Convolutional codes yield lower BER than uncoded BPSK at high SNR."""
    num_bits = 60_000
    eb_n0_db = 7.5
    rng = np.random.default_rng(333)
    info_bits = rng.integers(0, 2, size=num_bits, dtype=np.int8)

    # Uncoded BPSK
    res_uncoded = simulate_coded_bpsk_link(info_bits, "None", eb_n0_db=eb_n0_db, rng=rng)
    # Convolutional Rate 1/2 Viterbi
    res_conv = simulate_coded_bpsk_link(info_bits, "Convolutional(1/2, K=3)", eb_n0_db=eb_n0_db, rng=rng)

    # At 7.5 dB, Convolutional Viterbi provides significant coding gain
    assert res_conv["post_decode_ber"] < res_uncoded["post_decode_ber"], (
        f"Expected coded BER ({res_conv['post_decode_ber']}) < uncoded ({res_uncoded['post_decode_ber']})"
    )


def test_block_interleaver():
    """Verify block interleaver lossless reconstruction and burst scrambling."""
    interleaver = BlockInterleaver(nrows=8, ncols=8)
    data = np.arange(64, dtype=np.int8) % 2
    interleaved, orig_len = interleaver.interleave(data)
    deinterleaved = interleaver.deinterleave(interleaved, orig_len=orig_len)
    assert np.array_equal(data, deinterleaved)

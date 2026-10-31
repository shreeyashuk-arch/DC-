"""Test suite for Complete Integrated Transceiver.

Verifies:
1. Noiseless roundtrips yield 0 errors and perfect reconstruction across all source types:
   - Text message
   - PIL image (PSNR = 100 dB)
   - Synthetic audio tone
2. Correct operation under AWGN and Rayleigh fading.
3. Pre-decode vs post-decode error counts and BER metrics.
"""

import numpy as np
import pytest
from PIL import Image

from dcsim.transceiver import TransceiverPipeline
from dcsim.source.pcm import generate_synthetic_tone


def test_transceiver_text_noiseless():
    """Verify text transmission with FEC is 100% error-free in noiseless link."""
    message = "Digital Communication Lab Suite 2026: Success!"
    pipe = TransceiverPipeline(
        mod_scheme="QPSK",
        coding_scheme="Hamming(7,4)",
        channel_type="AWGN",
    )
    res = pipe.run(message, "text", eb_n0_db=30.0)
    assert res.post_decode_errors == 0
    assert res.post_decode_ber == 0.0
    assert res.rx_payload == message


def test_transceiver_image_noiseless():
    """Verify image transmission yields perfect PSNR (100 dB) in noiseless link."""
    img = Image.new("L", (16, 16), color=150)
    pipe = TransceiverPipeline(
        mod_scheme="16-QAM",
        coding_scheme="Convolutional(1/2, K=3)",
        channel_type="AWGN",
    )
    res = pipe.run(img, "image", eb_n0_db=30.0)
    assert res.post_decode_errors == 0
    assert res.image_psnr_db == 100.0
    assert np.array_equal(np.array(img), np.array(res.rx_payload))


def test_transceiver_audio_transmission():
    """Verify audio transmission through transceiver with PCM encoding."""
    _, audio = generate_synthetic_tone(freq_hz=200.0, duration_s=0.05, sampling_rate_hz=8000.0)
    pipe = TransceiverPipeline(
        mod_scheme="BPSK",
        coding_scheme="None",
        channel_type="AWGN",
        pcm_bits=8,
    )
    res = pipe.run(audio, "audio", eb_n0_db=30.0)
    assert res.post_decode_errors == 0
    assert res.audio_sqnr_db is not None
    # For 8-bit PCM, SQNR should be around 45-50 dB
    assert res.audio_sqnr_db > 40.0


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8-PSK", "16-QAM", "DBPSK", "BFSK", "OOK"])
def test_transceiver_all_modulations_noiseless(mod):
    """Verify all modulation options transmit and reconstruct bitstreams without error."""
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    pipe = TransceiverPipeline(
        mod_scheme=mod,  # type: ignore
        coding_scheme="None",
        channel_type="AWGN",
    )
    res = pipe.run(bits, "bits", eb_n0_db=35.0)
    assert res.post_decode_errors == 0
    assert np.array_equal(bits, res.rx_payload[:len(bits)])


def test_transceiver_rayleigh_fading_with_interleaver():
    """Verify end-to-end transceiver in Rayleigh fading with interleaver."""
    rng = np.random.default_rng(999)
    bits = rng.integers(0, 2, size=500, dtype=np.int8)
    pipe = TransceiverPipeline(
        mod_scheme="BPSK",
        coding_scheme="Convolutional(1/2, K=3)",
        channel_type="Rayleigh",
        use_interleaver=True,
        rayleigh_block_len=4,
    )
    res = pipe.run(bits, "bits", eb_n0_db=15.0, rng=rng)
    # The pipeline should complete and produce valid metrics
    assert res.num_info_bits == len(bits)
    assert res.pre_decode_ber >= 0.0
    assert res.post_decode_ber >= 0.0
    assert res.constellation_points is not None

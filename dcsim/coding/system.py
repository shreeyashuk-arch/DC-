"""Coded Link Simulation Engine and Coding Gain Measurement.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)
"""

from __future__ import annotations
from typing import Literal
import numpy as np
from dcsim.core.conventions import q_function, add_awgn_noise, calculate_ber, measure_eb
from dcsim.coding.repetition import RepetitionCode
from dcsim.coding.hamming import Hamming74Code
from dcsim.coding.convolutional import ConvolutionalCodec
from dcsim.coding.interleaver import BlockInterleaver
from dcsim.channel.fading import RayleighChannel

CodingSchemeType = Literal["None", "Repetition(3)", "Hamming(7,4)", "Convolutional(1/2, K=3)"]


def simulate_coded_bpsk_link(
    info_bits: np.ndarray,
    code_scheme: CodingSchemeType,
    eb_n0_db: float,
    channel_type: Literal["AWGN", "Rayleigh"] = "AWGN",
    use_interleaver: bool = False,
    rayleigh_block_len: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, float | np.ndarray]:
    """Simulate BPSK transmission through AWGN or Rayleigh fading with optional FEC and interleaving.

    Rate scaling convention:
      Eb is measured per information bit. Transmitting N info bits coded as N/R channel bits
      automatically dilutes energy to R*Eb per coded symbol without manual fudge factors.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    b = np.asarray(info_bits, dtype=np.int8).ravel()
    num_info_bits = len(b)

    # 1. Channel Encoding
    code_upper = code_scheme.upper()
    orig_coded_len = None

    if "HAMMING" in code_upper:
        codec_h = Hamming74Code()
        coded_bits, orig_coded_len = codec_h.encode(b)
        code_rate = codec_h.rate
    elif "CONVOLUTIONAL" in code_upper or "VITERBI" in code_upper:
        codec_c = ConvolutionalCodec()
        coded_bits = codec_c.encode(b)
        code_rate = codec_c.rate
    elif "REPETITION" in code_upper:
        codec_r = RepetitionCode(repeat=3)
        coded_bits = codec_r.encode(b)
        code_rate = codec_r.rate
    else:
        coded_bits = b.copy()
        code_rate = 1.0

    # 2. Optional Interleaving
    if use_interleaver:
        interleaver = BlockInterleaver(nrows=16, ncols=16)
        interleaved_bits, orig_inter_len = interleaver.interleave(coded_bits)
    else:
        interleaved_bits = coded_bits
        orig_inter_len = None

    # 3. BPSK Modulation: bit 1 -> +1.0, bit 0 -> -1.0
    channel_symbols = np.where(interleaved_bits == 1, 1.0, -1.0).astype(np.complex128)

    # 4. Channel Transmission with Sample-Measured Eb
    if channel_type.upper() == "RAYLEIGH":
        channel = RayleighChannel(block_length=rayleigh_block_len, rng=rng)
        y, h, eb, n0 = channel.apply(channel_symbols, eb_n0_db, num_info_bits=num_info_bits)
        # Perfect CSI Equalization
        decision_stat = np.real(np.conj(h) * y)
    else:
        # AWGN Channel
        noisy_syms, eb, n0 = add_awgn_noise(channel_symbols, eb_n0_db, num_info_bits=num_info_bits, rng=rng)
        decision_stat = np.real(noisy_syms)

    # 5. Demodulation (Hard Decisions on Channel Bits)
    rx_channel_bits = (decision_stat > 0.0).astype(np.int8)

    # Pre-decoding (raw channel) error rate
    pre_ber, pre_errors, _ = calculate_ber(interleaved_bits, rx_channel_bits)

    # 6. Optional Deinterleaving
    if use_interleaver:
        deinterleaved_bits = interleaver.deinterleave(rx_channel_bits, orig_len=orig_inter_len)
    else:
        deinterleaved_bits = rx_channel_bits

    # 7. Channel Decoding
    if "HAMMING" in code_upper:
        decoded_info = codec_h.decode(deinterleaved_bits, orig_len=orig_coded_len)
    elif "CONVOLUTIONAL" in code_upper or "VITERBI" in code_upper:
        decoded_info = codec_c.decode(deinterleaved_bits, num_info_bits=num_info_bits)
    elif "REPETITION" in code_upper:
        decoded_info = codec_r.decode(deinterleaved_bits)[:num_info_bits]
    else:
        decoded_info = deinterleaved_bits[:num_info_bits]

    # Post-decoding (final info bit) error rate
    post_ber, post_errors, _ = calculate_ber(b, decoded_info)

    # Uncoded theoretical reference
    if channel_type.upper() == "RAYLEIGH":
        g = 10.0 ** (eb_n0_db / 10.0)
        theo_uncoded_ber = 0.5 * (1.0 - np.sqrt(g / (1.0 + g)))
    else:
        theo_uncoded_ber = float(q_function(np.sqrt(2.0 * (10.0 ** (eb_n0_db / 10.0)))))

    return {
        "pre_decode_ber": pre_ber,
        "pre_decode_errors": pre_errors,
        "post_decode_ber": post_ber,
        "post_decode_errors": post_errors,
        "theo_uncoded_ber": theo_uncoded_ber,
        "code_rate": code_rate,
        "eb": eb,
        "n0": n0,
        "decoded_bits": decoded_info,
        "tx_info_bits": b,
    }

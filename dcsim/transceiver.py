"""End-to-End Digital Transceiver Simulation Pipeline.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)

Integrated Architecture:
Source -> Source Encoder (PCM/Converters) -> Channel Encoder (FEC) -> Interleaver ->
Modulator -> Channel (AWGN/Rayleigh) -> Demodulator -> Deinterleaver -> Channel Decoder ->
Reconstruction Sink.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from PIL import Image

from dcsim.core.conventions import measure_eb, add_awgn_noise, calculate_ber
from dcsim.core.converters import text_to_bits, bits_to_text, image_to_bits, bits_to_image
from dcsim.source.pcm import PCMCodec
from dcsim.coding.repetition import RepetitionCode
from dcsim.coding.hamming import Hamming74Code
from dcsim.coding.convolutional import ConvolutionalCodec
from dcsim.coding.interleaver import BlockInterleaver
from dcsim.channel.fading import RayleighChannel
from dcsim.utils.image_helpers import calculate_image_psnr
from dcsim.utils.audio_helpers import compute_audio_snr
from dcsim.modulation.advanced_mod import (
    modulate_8psk,
    demodulate_8psk,
    modulate_16qam,
    demodulate_16qam,
    modulate_dbpsk,
    demodulate_dbpsk,
    modulate_noncoherent_bfsk,
    demodulate_noncoherent_bfsk,
)

SourceType = Literal["text", "image", "audio", "bits"]
ModSchemeType = Literal["BPSK", "QPSK", "8-PSK", "16-QAM", "DBPSK", "BFSK", "OOK"]
CodingType = Literal["None", "Repetition(3)", "Hamming(7,4)", "Convolutional(1/2, K=3)"]
ChannelType = Literal["AWGN", "Rayleigh"]


@dataclass
class TransceiverResult:
    """Diagnostic data and reconstructed payload across all stages of the transceiver."""
    source_type: SourceType
    num_info_bits: int
    num_coded_bits: int
    code_rate: float
    pre_decode_errors: int
    pre_decode_ber: float
    post_decode_errors: int
    post_decode_ber: float
    measured_eb: float
    calculated_n0: float
    rx_payload: Any
    image_psnr_db: float | None = None
    audio_sqnr_db: float | None = None
    constellation_points: np.ndarray | None = None


class TransceiverPipeline:
    """Configurable multi-stage digital communication transceiver."""

    def __init__(
        self,
        mod_scheme: ModSchemeType = "BPSK",
        coding_scheme: CodingType = "None",
        channel_type: ChannelType = "AWGN",
        use_interleaver: bool = False,
        rayleigh_block_len: int = 1,
        pcm_bits: int = 8,
        pcm_companding: bool = False,
    ) -> None:
        self.mod_scheme = mod_scheme
        self.coding_scheme = coding_scheme
        self.channel_type = channel_type
        self.use_interleaver = use_interleaver
        self.rayleigh_block_len = rayleigh_block_len
        self.pcm_bits = pcm_bits
        self.pcm_companding = pcm_companding

    def run(
        self,
        source_data: Any,
        source_type: SourceType,
        eb_n0_db: float,
        rng: np.random.Generator | None = None,
    ) -> TransceiverResult:
        """Execute end-to-end transceiver transmission."""
        if rng is None:
            rng = np.random.default_rng(42)

        # ---------------- STAGE 1: SOURCE ENCODING ----------------
        image_meta = None
        pcm_codec = None
        pcm_vmax = 1.0

        if source_type == "text":
            info_bits = text_to_bits(str(source_data))
        elif source_type == "image":
            # Keep small images as-is; downsample only large uploaded images for snappy GUI simulation
            img_obj = source_data if isinstance(source_data, Image.Image) else Image.open(source_data)
            if img_obj.size[0] > 48 or img_obj.size[1] > 48:
                img_to_send = img_obj.resize((32, 32), Image.Resampling.LANCZOS).convert("L")
            else:
                img_to_send = img_obj.convert("L")
            info_bits, image_meta = image_to_bits(img_to_send, grayscale=True)
            source_ref = img_to_send
        elif source_type == "audio":
            audio_samples = np.asarray(source_data, dtype=np.float64)
            pcm_vmax = float(np.max(np.abs(audio_samples))) if np.max(np.abs(audio_samples)) > 0 else 1.0
            pcm_codec = PCMCodec(n_bits=self.pcm_bits, companding=self.pcm_companding)
            pcm_res = pcm_codec.encode_and_decode(audio_samples, v_max=pcm_vmax)
            info_bits = pcm_res.bitstream
        elif source_type == "bits":
            info_bits = np.asarray(source_data, dtype=np.int8).ravel()
        else:
            raise ValueError(f"Unknown source type: {source_type}")

        num_info_bits = len(info_bits)
        if num_info_bits == 0:
            raise ValueError("Source yielded 0 information bits.")

        # ---------------- STAGE 2: CHANNEL ENCODING ----------------
        coding_upper = self.coding_scheme.upper()
        orig_coded_len = None

        if "HAMMING" in coding_upper:
            codec_h = Hamming74Code()
            coded_bits, orig_coded_len = codec_h.encode(info_bits)
            code_rate = codec_h.rate
        elif "CONVOLUTIONAL" in coding_upper or "VITERBI" in coding_upper:
            codec_c = ConvolutionalCodec()
            coded_bits = codec_c.encode(info_bits)
            code_rate = codec_c.rate
        elif "REPETITION" in coding_upper:
            codec_r = RepetitionCode(repeat=3)
            coded_bits = codec_r.encode(info_bits)
            code_rate = codec_r.rate
        else:
            coded_bits = info_bits.copy()
            code_rate = 1.0

        # ---------------- STAGE 3: INTERLEAVING ----------------
        if self.use_interleaver:
            interleaver = BlockInterleaver(nrows=16, ncols=16)
            interleaved_bits, orig_inter_len = interleaver.interleave(coded_bits)
        else:
            interleaved_bits = coded_bits
            orig_inter_len = None

        # ---------------- STAGE 4: MODULATION ----------------
        mod_upper = self.mod_scheme.upper()
        # Modulate into complex baseband symbols
        if mod_upper == "BPSK":
            tx_symbols = np.where(interleaved_bits == 1, 1.0, -1.0).astype(np.complex128)
            plot_symbols = tx_symbols
        elif mod_upper == "QPSK":
            b_pad = interleaved_bits.copy()
            if len(b_pad) % 2 != 0:
                b_pad = np.append(b_pad, 0)
            quads = b_pad.reshape(-1, 2)
            ai = np.where(quads[:, 0] == 0, 1.0, -1.0) / np.sqrt(2.0)
            aq = np.where(quads[:, 1] == 0, 1.0, -1.0) / np.sqrt(2.0)
            tx_symbols = ai + 1j * aq
            plot_symbols = tx_symbols
        elif mod_upper == "8-PSK":
            tx_symbols, _ = modulate_8psk(interleaved_bits)
            plot_symbols = tx_symbols
        elif mod_upper == "16-QAM":
            tx_symbols, _ = modulate_16qam(interleaved_bits)
            plot_symbols = tx_symbols
        elif mod_upper == "DBPSK":
            tx_symbols = modulate_dbpsk(interleaved_bits)
            plot_symbols = tx_symbols
        elif "BFSK" in mod_upper:
            # 2-channel baseband
            tx_ch = modulate_noncoherent_bfsk(interleaved_bits)
            tx_symbols = tx_ch.ravel()
            plot_symbols = tx_symbols
        elif mod_upper == "OOK":
            tx_symbols = np.where(interleaved_bits == 1, np.sqrt(2.0), 0.0).astype(np.complex128)
            plot_symbols = tx_symbols
        else:
            raise ValueError(f"Unsupported modulation: {self.mod_scheme}")

        # ---------------- STAGE 5: CHANNEL IMPAIRMENTS ----------------
        if self.channel_type.upper() == "RAYLEIGH":
            chan = RayleighChannel(block_length=self.rayleigh_block_len, rng=rng)
            noisy_symbols, h, eb, n0 = chan.apply(tx_symbols, eb_n0_db, num_info_bits=num_info_bits)
            # Perfect CSI Derotation
            equalized_symbols = np.conj(h) * noisy_symbols / (np.abs(h) ** 2 + 1e-12)
            recv_for_demod = equalized_symbols
            plot_points = equalized_symbols
        else:
            noisy_symbols, eb, n0 = add_awgn_noise(tx_symbols, eb_n0_db, num_info_bits=num_info_bits, rng=rng)
            recv_for_demod = noisy_symbols
            plot_points = noisy_symbols

        # ---------------- STAGE 6: DEMODULATION ----------------
        if mod_upper == "BPSK":
            rx_raw_bits = (np.real(recv_for_demod) > 0.0).astype(np.int8)[: len(interleaved_bits)]
        elif mod_upper == "QPSK":
            dec_i = (np.real(recv_for_demod) < 0.0).astype(np.int8)
            dec_q = (np.imag(recv_for_demod) < 0.0).astype(np.int8)
            rx_raw_bits = np.column_stack([dec_i, dec_q]).ravel()[: len(interleaved_bits)]
        elif mod_upper == "8-PSK":
            rx_raw_bits = demodulate_8psk(recv_for_demod, len(interleaved_bits))
        elif mod_upper == "16-QAM":
            rx_raw_bits = demodulate_16qam(recv_for_demod, len(interleaved_bits))
        elif mod_upper == "DBPSK":
            rx_raw_bits = demodulate_dbpsk(recv_for_demod, len(interleaved_bits))
        elif "BFSK" in mod_upper:
            noisy_ch = recv_for_demod.reshape(len(interleaved_bits), 2)
            rx_raw_bits = demodulate_noncoherent_bfsk(noisy_ch)
        elif mod_upper == "OOK":
            rx_raw_bits = (np.real(recv_for_demod) > (np.sqrt(2.0 * eb) / 2.0)).astype(np.int8)[: len(interleaved_bits)]

        # Pre-decoding channel error metrics
        pre_ber, pre_errors, _ = calculate_ber(interleaved_bits, rx_raw_bits)

        # ---------------- STAGE 7: DEINTERLEAVING ----------------
        if self.use_interleaver:
            deinterleaved_bits = interleaver.deinterleave(rx_raw_bits, orig_len=orig_inter_len)
        else:
            deinterleaved_bits = rx_raw_bits

        # ---------------- STAGE 8: CHANNEL DECODING ----------------
        if "HAMMING" in coding_upper:
            decoded_bits = codec_h.decode(deinterleaved_bits, orig_len=orig_coded_len)
        elif "CONVOLUTIONAL" in coding_upper or "VITERBI" in coding_upper:
            decoded_bits = codec_c.decode(deinterleaved_bits, num_info_bits=num_info_bits)
        elif "REPETITION" in coding_upper:
            decoded_bits = codec_r.decode(deinterleaved_bits)[:num_info_bits]
        else:
            decoded_bits = deinterleaved_bits[:num_info_bits]

        post_ber, post_errors, _ = calculate_ber(info_bits, decoded_bits)

        # ---------------- STAGE 9: PAYLOAD RECONSTRUCTION ----------------
        image_psnr = None
        audio_sqnr = None

        if source_type == "text":
            rx_payload = bits_to_text(decoded_bits)
        elif source_type == "image":
            rx_image = bits_to_image(decoded_bits, image_meta)
            image_psnr, _ = calculate_image_psnr(source_ref, rx_image)
            rx_payload = rx_image
        elif source_type == "audio":
            # Reconstruct quantized indices from bits
            n_b = self.pcm_bits
            num_words = len(decoded_bits) // n_b
            if num_words > 0:
                bit_matrix = decoded_bits[: num_words * n_b].reshape(num_words, n_b)
                powers = 1 << np.arange(n_b - 1, -1, -1)
                rx_indices = bit_matrix @ powers
                num_levels = 1 << n_b
                delta = 2.0 * pcm_vmax / num_levels
                x_hat = -pcm_vmax + (rx_indices + 0.5) * delta
                if self.pcm_companding:
                    x_hat = (x_hat / pcm_vmax)
                    x_hat = np.sign(x_hat) * (((1.0 + 255.0) ** np.abs(x_hat) - 1.0) / 255.0) * pcm_vmax
            else:
                x_hat = np.zeros_like(source_data)
            audio_sqnr = compute_audio_snr(source_data, x_hat)
            rx_payload = x_hat
        else:
            rx_payload = decoded_bits

        return TransceiverResult(
            source_type=source_type,
            num_info_bits=num_info_bits,
            num_coded_bits=len(coded_bits),
            code_rate=code_rate,
            pre_decode_errors=pre_errors,
            pre_decode_ber=pre_ber,
            post_decode_errors=post_errors,
            post_decode_ber=post_ber,
            measured_eb=eb,
            calculated_n0=n0,
            rx_payload=rx_payload,
            image_psnr_db=image_psnr,
            audio_sqnr_db=audio_sqnr,
            constellation_points=plot_points,
        )

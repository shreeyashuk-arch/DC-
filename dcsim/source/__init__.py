"""Module A: Source and Baseband Transmission Lab (Owner: Student 1).

Implements:
- Pulse Code Modulation (PCM): Uniform quantization (1-12 bits), mu-law companding (mu=255),
  SQNR analysis, WAV audio encoding, and Nyquist aliasing demonstration.
- Line Coding: NRZ-L, RZ, Manchester, AMI waveform generators, Welch PSD analysis,
  and characteristic comparison tables.
- Pulse Shaping & Bandwidth: Root-Raised Cosine (RRC) matched filter pair, eye diagrams,
  Inter-Symbol Interference (ISI) demonstration, and AWGN BER verification.
"""

from .pcm import (
    PCMCodec,
    mu_law_compress,
    mu_law_expand,
    uniform_quantize,
    calculate_sqnr,
    generate_synthetic_tone,
    demonstrate_aliasing,
)
from .line_coding import (
    encode_line_code,
    compute_line_code_psd,
    get_line_code_summary_table,
    LineCodeType,
)
from .pulse_shaping import (
    rrc_filter_taps,
    rc_filter_taps,
    MatchedRRCFilter,
    generate_eye_diagram_data,
    simulate_rrc_link,
)

__all__ = [
    "PCMCodec",
    "mu_law_compress",
    "mu_law_expand",
    "uniform_quantize",
    "calculate_sqnr",
    "generate_synthetic_tone",
    "demonstrate_aliasing",
    "encode_line_code",
    "compute_line_code_psd",
    "get_line_code_summary_table",
    "LineCodeType",
    "rrc_filter_taps",
    "rc_filter_taps",
    "MatchedRRCFilter",
    "generate_eye_diagram_data",
    "simulate_rrc_link",
]

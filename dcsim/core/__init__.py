"""Core communication primitives, energy normalization conventions, passband transceiver links,
and source conversion utilities.
"""

from .conventions import (
    q_function,
    erfc_from_q,
    measure_eb,
    calculate_n0,
    add_awgn_noise,
    calculate_ber,
)
from .passband_link import (
    PassbandModulator,
    PassbandDemodulator,
    simulate_passband_link,
    theoretical_ber,
)
from .converters import (
    text_to_bits,
    bits_to_text,
    image_to_bits,
    bits_to_image,
)

__all__ = [
    "q_function",
    "erfc_from_q",
    "measure_eb",
    "calculate_n0",
    "add_awgn_noise",
    "calculate_ber",
    "PassbandModulator",
    "PassbandDemodulator",
    "simulate_passband_link",
    "theoretical_ber",
    "text_to_bits",
    "bits_to_text",
    "image_to_bits",
    "bits_to_image",
]

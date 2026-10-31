"""Repetition Code (Rate 1/3) with Majority Logic Decoding.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)
"""

from __future__ import annotations
import numpy as np


class RepetitionCode:
    """Repetition(3) Block Code."""

    def __init__(self, repeat: int = 3) -> None:
        if repeat % 2 == 0 or repeat < 3:
            raise ValueError("repeat must be an odd integer >= 3 for definitive majority decision.")
        self.repeat = repeat
        self.rate = 1.0 / float(repeat)

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode each information bit by repeating it `repeat` times."""
        b = np.asarray(info_bits, dtype=np.int8).ravel()
        return np.repeat(b, self.repeat)

    def decode(self, coded_bits: np.ndarray) -> np.ndarray:
        """Majority-logic hard decision decoding."""
        c = np.asarray(coded_bits, dtype=np.int8).ravel()
        num_codewards = len(c) // self.repeat
        reshaped = c[: num_codewards * self.repeat].reshape(num_codewards, self.repeat)
        # Sum votes: if sum > repeat / 2 -> 1, else 0
        votes = np.sum(reshaped, axis=1)
        decoded = (votes > (self.repeat / 2.0)).astype(np.int8)
        return decoded

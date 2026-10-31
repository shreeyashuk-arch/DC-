"""Systematic Hamming(7,4) Forward Error Correction with Syndrome Decoding.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)

Parameters:
- Block length n = 7
- Message length k = 4
- Code rate R = 4/7 ~ 0.5714
- Minimum distance d_min = 3 (corrects single bit errors per 7-bit codeword)
"""

from __future__ import annotations
import numpy as np


class Hamming74Code:
    """Systematic Hamming(7,4) linear block code with syndrome table lookup decoding."""

    def __init__(self) -> None:
        self.k = 4
        self.n = 7
        self.rate = 4.0 / 7.0

        # Systematic Generator Matrix G = [I_4 | P]
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.int8)

        # Parity Check Matrix H = [P^T | I_3]
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
        ], dtype=np.int8)

        # Map each syndrome tuple (s0, s1, s2) to the erroneous bit index (0..6)
        # s = r * H^T (mod 2)
        self.syndrome_to_error_idx: dict[tuple[int, int, int], int] = {}
        for col_idx in range(self.n):
            syn = tuple(self.H[:, col_idx].tolist())
            self.syndrome_to_error_idx[syn] = col_idx

    def encode(self, info_bits: np.ndarray) -> tuple[np.ndarray, int]:
        """Encode information bits into Hamming(7,4) codewords.

        Pads with zeros if length is not a multiple of 4.

        Returns
        -------
        coded_bits : np.ndarray
            Array of length (ceil(len/4) * 7).
        orig_len : int
            Original length of info_bits for unpadding.
        """
        b = np.asarray(info_bits, dtype=np.int8).ravel()
        orig_len = len(b)
        rem = orig_len % self.k
        if rem != 0:
            b = np.append(b, np.zeros(self.k - rem, dtype=np.int8))

        msg_matrix = b.reshape(-1, self.k)
        # c = m * G (mod 2)
        coded_matrix = (msg_matrix @ self.G) % 2
        return coded_matrix.ravel(), orig_len

    def decode(self, coded_bits: np.ndarray, orig_len: int | None = None) -> np.ndarray:
        """Syndrome decode received 7-bit blocks and correct any single-bit error."""
        c = np.asarray(coded_bits, dtype=np.int8).ravel()
        num_blocks = len(c) // self.n
        codewords = c[: num_blocks * self.n].reshape(num_blocks, self.n).copy()

        # Compute syndromes: s = codewords * H^T (mod 2) -> shape: (num_blocks, 3)
        syndromes = (codewords @ self.H.T) % 2

        # Error correction
        for i in range(num_blocks):
            syn = tuple(syndromes[i].tolist())
            if syn in self.syndrome_to_error_idx:
                err_idx = self.syndrome_to_error_idx[syn]
                codewords[i, err_idx] ^= 1  # Flip erroneous bit

        # Extract systematic information bits (first 4 bits of each codeword)
        decoded_info = codewords[:, : self.k].ravel()
        if orig_len is not None:
            decoded_info = decoded_info[:orig_len]

        return decoded_info

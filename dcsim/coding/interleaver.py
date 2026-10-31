"""Matrix Block Interleaver for Burst Error Dispersal in Fading Channels.

Owner: Student 3 (Module C: Channel, Coding and Integration Lab)
"""

from __future__ import annotations
import numpy as np


class BlockInterleaver:
    """Rectangular matrix block interleaver (write rows, read columns)."""

    def __init__(self, nrows: int = 16, ncols: int = 16) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.block_size = nrows * ncols

    def interleave(self, bits: np.ndarray) -> tuple[np.ndarray, int]:
        """Interleave bits by writing rows and reading columns."""
        b = np.asarray(bits, dtype=np.int8).ravel()
        orig_len = len(b)
        rem = orig_len % self.block_size
        if rem != 0:
            b = np.append(b, np.zeros(self.block_size - rem, dtype=np.int8))

        num_blocks = len(b) // self.block_size
        interleaved_blocks = []

        for k in range(num_blocks):
            blk = b[k * self.block_size : (k + 1) * self.block_size].reshape(self.nrows, self.ncols)
            # Read out by columns (transpose)
            interleaved_blocks.append(blk.T.ravel())

        return np.concatenate(interleaved_blocks), orig_len

    def deinterleave(self, bits: np.ndarray, orig_len: int | None = None) -> np.ndarray:
        """Deinterleave bits by reversing column/row ordering."""
        b = np.asarray(bits, dtype=np.int8).ravel()
        num_blocks = len(b) // self.block_size
        deinterleaved_blocks = []

        for k in range(num_blocks):
            # Coded stream was read by columns, so shape is (ncols, nrows)
            blk = b[k * self.block_size : (k + 1) * self.block_size].reshape(self.ncols, self.nrows)
            # Transpose back to (nrows, ncols) and read by rows
            deinterleaved_blocks.append(blk.T.ravel())

        if len(deinterleaved_blocks) == 0:
            return np.empty(0, dtype=np.int8)

        deinterleaved = np.concatenate(deinterleaved_blocks)
        if orig_len is not None:
            deinterleaved = deinterleaved[:orig_len]

        return deinterleaved

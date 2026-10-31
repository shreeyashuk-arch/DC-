"""Source data converters: Text and Image to binary bitstreams and back.

Supports lossless bidirectional conversion of strings and PIL images to flat NumPy
arrays of 0s and 1s, with robust error resilience for channel corrupted streams.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


def text_to_bits(text: str, encoding: str = "utf-8") -> np.ndarray:
    """Convert an arbitrary text string into a 1D NumPy array of bits (0s and 1s).

    Parameters
    ----------
    text : str
        Input string.
    encoding : str
        Text character encoding (default: utf-8).

    Returns
    -------
    np.ndarray
        1D array of uint8 values (0 or 1), length = 8 * len(encoded_bytes).
    """
    raw_bytes = text.encode(encoding)
    if len(raw_bytes) == 0:
        return np.empty(0, dtype=np.uint8)

    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    # Unpack 8 bits per byte (MSB first)
    bits = np.unpackbits(byte_array)
    return bits.astype(np.uint8)


def bits_to_text(bits: np.ndarray, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Convert a 1D NumPy array of bits back into a text string.

    Parameters
    ----------
    bits : np.ndarray
        1D array of 0s and 1s.
    encoding : str
        Target text encoding.
    errors : str
        Error handling strategy for invalid byte sequences ('replace' or 'ignore').

    Returns
    -------
    str
        Decoded text string.
    """
    b = np.asarray(bits, dtype=np.uint8).ravel()
    if len(b) == 0:
        return ""

    # Truncate to multiple of 8 bits
    valid_len = (len(b) // 8) * 8
    if valid_len == 0:
        return ""
    b_trimmed = b[:valid_len]

    packed_bytes = np.packbits(b_trimmed).tobytes()
    return packed_bytes.decode(encoding, errors=errors)


def image_to_bits(
    image_source: Image.Image | str | Path,
    resize: tuple[int, int] | None = None,
    grayscale: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert an image into a 1D binary bitstream.

    Parameters
    ----------
    image_source : PIL.Image or str or Path
        Input image object or file path.
    resize : tuple of (width, height), optional
        Optional downsampling dimensions to keep bit count manageable for fast demo.
    grayscale : bool
        If True, converts image to 8-bit grayscale ('L' mode).

    Returns
    -------
    bits : np.ndarray
        1D array of binary digits.
    metadata : dict
        Image parameters needed for exact reconstruction (shape, mode, format).
    """
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source)
    else:
        img = image_source

    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    if resize is not None:
        img = img.resize(resize, Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.uint8)
    metadata = {
        "shape": img_array.shape,
        "mode": img.mode,
        "size": img.size,
        "num_pixels": img_array.size,
    }

    bits = np.unpackbits(img_array.ravel()).astype(np.uint8)
    return bits, metadata


def bits_to_image(bits: np.ndarray, metadata: dict[str, Any]) -> Image.Image:
    """Reconstruct a PIL Image from a bitstream and metadata.

    Parameters
    ----------
    bits : np.ndarray
        1D array of bits.
    metadata : dict
        Metadata dictionary containing 'shape' and 'mode'.

    Returns
    -------
    PIL.Image.Image
        Reconstructed image.
    """
    target_shape = metadata["shape"]
    mode = metadata["mode"]
    total_bytes = int(np.prod(target_shape))
    target_bits = total_bytes * 8

    b = np.asarray(bits, dtype=np.uint8).ravel()
    if len(b) < target_bits:
        # Zero-pad if truncated
        b_padded = np.zeros(target_bits, dtype=np.uint8)
        b_padded[: len(b)] = b
    else:
        b_padded = b[:target_bits]

    byte_array = np.packbits(b_padded)
    img_array = byte_array.reshape(target_shape)
    return Image.fromarray(img_array, mode=mode)

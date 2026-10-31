"""Image quality metrics and conversion helpers."""

from __future__ import annotations
import numpy as np
from PIL import Image


def calculate_image_psnr(orig: Image.Image | np.ndarray, rx: Image.Image | np.ndarray) -> tuple[float, float]:
    """Compute Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) in dB.

    Formula:
        MSE = mean((orig - rx)^2)
        PSNR = 10 * log10( MAX_I^2 / MSE ) = 10 * log10( 255^2 / MSE )
    """
    arr1 = np.asarray(orig, dtype=np.float64)
    arr2 = np.asarray(rx, dtype=np.float64)

    if arr1.shape != arr2.shape:
        # Match dimensions if needed
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))
        slices = tuple(slice(0, s) for s in min_shape)
        arr1 = arr1[slices]
        arr2 = arr2[slices]

    mse = float(np.mean((arr1 - arr2) ** 2))
    if mse <= 1e-12:
        return 100.0, 0.0  # Perfect reconstruction

    max_pixel = 255.0
    psnr = 10.0 * np.log10((max_pixel**2) / mse)
    return psnr, mse

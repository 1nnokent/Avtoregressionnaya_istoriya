import numpy as np
from scipy.fft import dctn, idctn


def check_2d(array: np.ndarray, name: str) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D, got shape {array.shape}")


def dct2(image: np.ndarray) -> np.ndarray:
    """2D DCT-II with onb"""
    image = image.astype(np.float64)
    check_2d(image, "image")
    return np.asarray(dctn(image, type=2, norm="ortho"))


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """Inverse of DCT-II"""
    coeffs = coeffs.astype(np.float64)
    check_2d(coeffs, "coeffs")
    return np.asarray(idctn(coeffs, type=2, norm="ortho"))

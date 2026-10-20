import numpy as np


def signed_log(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(scale * np.abs(x))


def inverse_signed_log(y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y)) / scale

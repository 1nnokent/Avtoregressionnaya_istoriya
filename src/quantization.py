import numpy as np


def fit_uniform_range(values: np.ndarray) -> tuple[float, float]:
    """range [low, high] for uniform quantization. for TRAIN data only."""
    values = np.asarray(values)
    return float(values.min()), float(values.max())


def quantize_uniform(x: np.ndarray, low: float, high: float, num_bins: int) -> np.ndarray:
    """real values -> integer tokens in [0, num_bins - 1]"""
    x = np.asarray(x, dtype=np.float64)
    bins = np.floor((x - low) / (high - low) * num_bins)
    return np.clip(bins, 0, num_bins - 1).astype(np.int64)


def dequantize_uniform(tokens: np.ndarray, low: float, high: float, num_bins: int) -> np.ndarray:
    """integer tokens -> centers of their bins"""
    tokens = np.asarray(tokens)
    bin_width = (high - low) / num_bins
    return low + (tokens + 0.5) * bin_width

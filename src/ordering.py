from functools import lru_cache

import numpy as np

from .decomposition import check_2d


@lru_cache
def zigzag_indices(height: int, width: int) -> np.ndarray:
    """flat indices of an (H, W) matrix in zigzag order"""
    positions = [(i, j) for i in range(height) for j in range(width)]
    positions.sort(key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else -p[0]))
    indices = np.array([i * width + j for i, j in positions])
    indices.flags.writeable = False
    return indices


def zigzag(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    check_2d(matrix, "matrix")
    return matrix.reshape(-1)[zigzag_indices(*matrix.shape)]


def inverse_zigzag(sequence: np.ndarray, height: int, width: int) -> np.ndarray:
    sequence = np.asarray(sequence)
    if sequence.shape != (height * width,):
        raise ValueError(f"expected sequence of shape ({height * width},), got {sequence.shape}")
    
    flat = np.empty_like(sequence)
    flat[zigzag_indices(height, width)] = sequence
    return flat.reshape(height, width)

from functools import lru_cache

import numpy as np

ORDERINGS = ("zigzag", "dyadic", "radial", "block")


def _zigzag_key(i: int, j: int) -> tuple:
    return i + j, i if (i + j) % 2 else -i


_SORT_KEYS = {
    "zigzag": lambda i, j, h, w, b: _zigzag_key(i, j),
    "radial": lambda i, j, h, w, b: ((i / h) ** 2 + (j / w) ** 2, i),
    "dyadic": lambda i, j, h, w, b: (max(i, j).bit_length(), *_zigzag_key(i, j)),
    "block": lambda i, j, h, w, b: (i // b, j // b, *_zigzag_key(i % b, j % b)),
}


@lru_cache
def ordering_indices(name: str, height: int, width: int, block_size: int = 8) -> np.ndarray:
    if name not in _SORT_KEYS:
        raise ValueError(f"unknown ordering {name!r}, expected one of {ORDERINGS}")
    key = _SORT_KEYS[name]
    positions = sorted(
        ((i, j) for i in range(height) for j in range(width)),
        key=lambda p: key(p[0], p[1], height, width, block_size),
    )
    indices = np.array([i * width + j for i, j in positions])
    indices.flags.writeable = False
    return indices


def forward_ordering(coeffs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return coeffs.reshape(*coeffs.shape[:-2], -1)[..., indices]


def invert_ordering(sequence: np.ndarray, indices: np.ndarray, height: int, width: int) -> np.ndarray:
    flat = np.empty_like(sequence)
    flat[..., indices] = sequence
    return flat.reshape(*sequence.shape[:-1], height, width)

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial

import numpy as np
import pywt
from scipy.fft import dctn, idctn

DECOMPOSITIONS = ("dct", "block_dct", "dft", "wavelet")
Transform = Callable[..., np.ndarray]


def check_2d(array: np.ndarray, name: str) -> None:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D, got shape {array.shape}")


@dataclass(frozen=True)
class Decomposition:
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


def get_decomposition(name: str, **params) -> Decomposition:
    transforms: dict[str, tuple[Transform, Transform]] = {
        "dct": (dct_forward, dct_inverse),
        "block_dct": (block_dct_forward, block_dct_inverse),
        "dft": (dft_forward, dft_inverse),
        "wavelet": (wavelet_forward, wavelet_inverse),
    }
    if name not in transforms:
        raise ValueError(
            f"unknown decomposition {name!r}, expected one of {DECOMPOSITIONS}"
        )
    forward, inverse = transforms[name]
    return Decomposition(name, partial(forward, **params), partial(inverse, **params))


def dct_forward(x: np.ndarray) -> np.ndarray:
    return np.asarray(dctn(x, type=2, norm="ortho", axes=(-2, -1)))


def dct_inverse(coeffs: np.ndarray) -> np.ndarray:
    return np.asarray(idctn(coeffs, type=2, norm="ortho", axes=(-2, -1)))


def block_dct_forward(x: np.ndarray, block_size: int = 8) -> np.ndarray:
    blocks = _split_blocks(x, block_size)
    return _merge_blocks(np.asarray(dctn(blocks, type=2, norm="ortho", axes=(-3, -1))))


def block_dct_inverse(coeffs: np.ndarray, block_size: int = 8) -> np.ndarray:
    blocks = _split_blocks(coeffs, block_size)
    return _merge_blocks(np.asarray(idctn(blocks, type=2, norm="ortho", axes=(-3, -1))))


def _split_blocks(x: np.ndarray, block_size: int) -> np.ndarray:
    height, width = x.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError(
            f"image {height}x{width} can not be cut into {block_size}x{block_size} blocks"
        )
    return x.reshape(
        *x.shape[:-2], height // block_size, block_size, width // block_size, block_size
    )


def _merge_blocks(blocks: np.ndarray) -> np.ndarray:
    return blocks.reshape(*blocks.shape[:-4], blocks.shape[-4] * blocks.shape[-3], -1)


@lru_cache
def interleaved_frequencies(n: int) -> np.ndarray:
    order = [0]
    for k in range(1, n // 2 + 1):
        order += [k, n - k] if k != n - k else [k]
    return np.array(order)


def dft_forward(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(x, norm="ortho")
    hartley = (np.real(spectrum) - np.imag(spectrum)).astype(x.dtype)
    rows, cols = (
        interleaved_frequencies(x.shape[-2]),
        interleaved_frequencies(x.shape[-1]),
    )
    return hartley[..., rows[:, None], cols[None, :]]


def dft_inverse(coeffs: np.ndarray) -> np.ndarray:
    rows, cols = (
        interleaved_frequencies(coeffs.shape[-2]),
        interleaved_frequencies(coeffs.shape[-1]),
    )
    hartley = np.empty_like(coeffs)
    hartley[..., rows[:, None], cols[None, :]] = coeffs
    spectrum = np.fft.fft2(hartley, norm="ortho")
    return (np.real(spectrum) - np.imag(spectrum)).astype(coeffs.dtype)


def max_wavelet_level(height: int, width: int) -> int:
    level = 0
    while height % 2 == 0 and width % 2 == 0 and min(height, width) > 1:
        height, width, level = height // 2, width // 2, level + 1
    return level


def wavelet_forward(
    x: np.ndarray, wavelet: str = "haar", level: int | None = None
) -> np.ndarray:
    level = level if level is not None else max_wavelet_level(*x.shape[-2:])
    coeffs = pywt.wavedec2(x, wavelet, mode="periodization", level=level, axes=(-2, -1))
    array, _ = pywt.coeffs_to_array(coeffs, axes=(-2, -1))
    return array


def wavelet_inverse(
    coeffs: np.ndarray, wavelet: str = "haar", level: int | None = None
) -> np.ndarray:
    level = level if level is not None else max_wavelet_level(*coeffs.shape[-2:])
    slices = _wavelet_slices(*coeffs.shape[-2:], wavelet, level)
    parts = pywt.array_to_coeffs(coeffs, slices, output_format="wavedec2")
    return pywt.waverec2(parts, wavelet, mode="periodization", axes=(-2, -1)).astype(
        coeffs.dtype
    )


@lru_cache
def _wavelet_slices(height: int, width: int, wavelet: str, level: int) -> list:
    coeffs = pywt.wavedec2(
        np.zeros((height, width)), wavelet, mode="periodization", level=level
    )
    _, slices = pywt.coeffs_to_array(coeffs)
    return [(..., *slices[0])] + [
        {key: (..., *value) for key, value in part.items()} for part in slices[1:]
    ]

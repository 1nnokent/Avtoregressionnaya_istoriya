from dataclasses import dataclass

import numpy as np

from src.decomposition import dct2, idct2
from src.normalization import inverse_signed_log, signed_log
from src.ordering import inverse_zigzag, zigzag
from src.quantization import dequantize_uniform, fit_uniform_range, quantize_uniform


@dataclass(frozen=True)
class BaselineParams:
    height: int
    width: int
    num_bins: int
    log_scale: float
    low: float
    high: float


def image_to_sequence(image: np.ndarray, log_scale: float) -> np.ndarray:
    return signed_log(zigzag(dct2(image)), scale=log_scale)


def sequence_to_image(sequence: np.ndarray, params: BaselineParams) -> np.ndarray:
    coeffs = inverse_zigzag(inverse_signed_log(sequence, scale=params.log_scale), params.height, params.width)
    return idct2(coeffs)


def fit(train_images: np.ndarray, num_bins: int = 256, log_scale: float = 255.0) -> BaselineParams:
    train_images = np.asarray(train_images)
    sequences = np.stack([image_to_sequence(image, log_scale) for image in train_images])
    low, high = fit_uniform_range(sequences)
    height, width = train_images.shape[1:]
    return BaselineParams(height, width, num_bins, log_scale, low, high)


def image_to_tokens(image: np.ndarray, params: BaselineParams) -> np.ndarray:
    sequence = image_to_sequence(image, params.log_scale)
    return quantize_uniform(sequence, params.low, params.high, params.num_bins)


def tokens_to_image(tokens: np.ndarray, params: BaselineParams) -> np.ndarray:
    sequence = dequantize_uniform(tokens, params.low, params.high, params.num_bins)
    return sequence_to_image(sequence, params)

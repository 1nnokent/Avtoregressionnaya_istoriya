from dataclasses import dataclass, field

import numpy as np

from .decomposition import get_decomposition
from .normalization import Normalizer
from .ordering import forward_ordering, invert_ordering, ordering_indices
from .quantization import Quantizer


@dataclass
class TokenizerConfig:
    decomposition: str = "dct"
    decomposition_params: dict = field(default_factory=dict)
    ordering: str = "zigzag"
    ordering_params: dict = field(default_factory=dict)
    normalization: dict = field(default_factory=dict)
    quantization: dict = field(default_factory=dict)


DEFAULT_NORMALIZATION = {"transform": "signed_log", "log_scale": 255.0, "scaling": "minmax"}
DEFAULT_QUANTIZATION = {"method": "uniform", "num_bins": 256}


class Tokenizer:
    def __init__(self, config: TokenizerConfig | dict | None = None):
        if config is None:
            config = TokenizerConfig()
        elif isinstance(config, dict):
            config = TokenizerConfig(**config)
        self.config = config
        self.decomposition = get_decomposition(config.decomposition, **config.decomposition_params)
        self.normalizer = Normalizer(**{**DEFAULT_NORMALIZATION, **config.normalization})
        self.quantizer = Quantizer(**{**DEFAULT_QUANTIZATION, **config.quantization})
        self.image_shape: tuple[int, int, int] | None = None

    @property
    def vocab_size(self) -> int:
        return self.quantizer.num_bins

    @property
    def sequence_length(self) -> int:
        self._check_fitted()
        channels, height, width = self.image_shape
        return channels * height * width

    def sequence(self, images: np.ndarray) -> np.ndarray:
        coeffs = self.decomposition.forward(_as_nchw(images))
        ordered = forward_ordering(coeffs, self._indices(*coeffs.shape[-2:]))
        return ordered.reshape(len(ordered), -1)

    def sequence_to_images(self, sequences: np.ndarray) -> np.ndarray:
        channels, height, width = self.image_shape
        ordered = sequences.reshape(len(sequences), channels, height * width)
        return self.decomposition.inverse(invert_ordering(ordered, self._indices(height, width), height, width))

    def fit(self, train_images: np.ndarray) -> "Tokenizer":
        train_images = _as_nchw(train_images)
        self.image_shape = train_images.shape[1:]
        sequences = self.sequence(train_images)
        self.quantizer.fit(self.normalizer.fit(sequences)(sequences))
        return self

    def encode(self, images: np.ndarray) -> np.ndarray:
        self._check_fitted()
        images = _as_nchw(images)
        if images.shape[1:] != self.image_shape:
            raise ValueError(f"tokenizer was fitted on images {self.image_shape}, got {images.shape[1:]}")
        return self.quantizer.encode(self.normalizer(self.sequence(images)))

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._check_fitted()
        tokens = np.asarray(tokens).reshape(-1, self.sequence_length)
        return self.sequence_to_images(self.normalizer.inverse(self.quantizer.decode(tokens)))

    def _indices(self, height: int, width: int) -> np.ndarray:
        return ordering_indices(self.config.ordering, height, width, **self.config.ordering_params)

    def _check_fitted(self):
        if self.image_shape is None:
            raise RuntimeError("call fit() on train images first")


def _as_nchw(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        return images[:, None]
    if images.ndim != 4:
        raise ValueError(f"expected images (N, H, W) or (N, C, H, W), got shape {images.shape}")
    return images

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from .channel_transform import CHANNEL_TRANSFORMS, ChannelTransform
from .decomposition import get_decomposition
from .normalization import Normalizer
from .ordering import forward_ordering, invert_ordering, ordering_indices
from .quantization import KMeansQuantizer, Quantizer


@dataclass
class TokenizerConfig:
    channel_transform: str | None = None
    decomposition: str = "dct"
    decomposition_params: dict = field(default_factory=dict)
    ordering: str = "zigzag"
    ordering_params: dict = field(default_factory=dict)
    normalization: dict = field(default_factory=dict)
    quantization: dict = field(default_factory=dict)


DEFAULT_NORMALIZATION = {"transform": "signed_log", "log_scale": 255.0, "scaling": "minmax"}
DEFAULT_QUANTIZATION = {"method": "uniform", "num_bins": 256}
QUANTIZATION_METHODS = ("uniform", "kmeans")
STATE_FORMAT_VERSION = 1


class Tokenizer:
    def __init__(self, config: TokenizerConfig | dict | None = None):
        if config is None:
            config = TokenizerConfig()
        elif isinstance(config, dict):
            config = TokenizerConfig(**config)
        self.config = config
        self.channel_transform = (
            None
            if config.channel_transform is None
            else ChannelTransform(config.channel_transform)
        )
        self.decomposition = get_decomposition(config.decomposition, **config.decomposition_params)
        self.normalizer = Normalizer(**{**DEFAULT_NORMALIZATION, **config.normalization})
        self.quantizer = _make_quantizer(config.quantization)
        self.image_shape: tuple[int, int, int] | None = None

    @property
    def vocab_size(self) -> int:
        return self.quantizer.num_bins

    @property
    def sequence_length(self) -> int:
        image_shape = self._check_fitted()
        match self.quantizer:
            case KMeansQuantizer():
                return self.quantizer.sequence_length
            case _:
                channels, height, width = image_shape
                return channels * height * width

    def sequence(self, images: np.ndarray) -> np.ndarray:
        images = _as_nchw(images)
        channels = images.shape[1]
        transformed = self._channel_transform(channels).forward(images)
        coeffs = self.decomposition.forward(transformed)
        ordered = forward_ordering(coeffs, self._indices(*coeffs.shape[-2:]))
        return ordered.reshape(len(ordered), -1)

    def sequence_to_images(self, sequences: np.ndarray) -> np.ndarray:
        channels, height, width = self._check_fitted()
        ordered = sequences.reshape(len(sequences), channels, height * width)
        transformed = self.decomposition.inverse(
            invert_ordering(ordered, self._indices(height, width), height, width)
        )
        return self._channel_transform(channels).inverse(transformed)

    def fit(self, train_images: np.ndarray) -> "Tokenizer":
        train_images = _as_nchw(train_images)
        self.image_shape = train_images.shape[1:]
        sequences = self.sequence(train_images)
        normalized = self.normalizer.fit(sequences)(sequences)
        self.quantizer.fit(self._quantizer_input(normalized))
        return self

    def encode(self, images: np.ndarray) -> np.ndarray:
        image_shape = self._check_fitted()
        images = _as_nchw(images)
        if images.shape[1:] != image_shape:
            raise ValueError(
                f"tokenizer was fitted on images {image_shape}, got {images.shape[1:]}"
            )
        normalized = self.normalizer(self.sequence(images))
        return self.quantizer.encode(self._quantizer_input(normalized))

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._check_fitted()
        tokens = np.asarray(tokens).reshape(-1, self.sequence_length)
        normalized = self.quantizer.decode(tokens)
        if isinstance(self.quantizer, KMeansQuantizer):
            normalized = normalized.reshape(len(normalized), -1)
        return self.sequence_to_images(self.normalizer.inverse(normalized))

    def state_dict(self) -> dict[str, np.ndarray]:
        image_shape = self._check_fitted()
        state = {
            "state_format": np.asarray(STATE_FORMAT_VERSION, dtype=np.int64),
            "image_shape": np.asarray(image_shape, dtype=np.int64),
        }
        state.update(
            {f"normalizer.{key}": value for key, value in self.normalizer.state_dict().items()}
        )
        state.update(
            {f"quantizer.{key}": value for key, value in self.quantizer.state_dict().items()}
        )
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        version = int(np.asarray(state["state_format"]).item())
        if version != STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer state format {version}")
        self.image_shape = cast(
            tuple[int, int, int],
            tuple(int(value) for value in state["image_shape"]),
        )
        self.normalizer.load_state_dict(
            {
                key.removeprefix("normalizer."): value
                for key, value in state.items()
                if key.startswith("normalizer.")
            }
        )
        self.quantizer.load_state_dict(
            {
                key.removeprefix("quantizer."): value
                for key, value in state.items()
                if key.startswith("quantizer.")
            }
        )

    def _indices(self, height: int, width: int) -> np.ndarray:
        return ordering_indices(self.config.ordering, height, width, **self.config.ordering_params)

    def _quantizer_input(self, sequences: np.ndarray) -> np.ndarray:
        if not isinstance(self.quantizer, KMeansQuantizer):
            return sequences
        channels, height, width = self._check_fitted()
        return sequences.reshape(len(sequences), channels, height * width)

    def _channel_transform(self, channels: int) -> ChannelTransform:
        if self.channel_transform is not None:
            return self.channel_transform
        if channels == 1:
            return ChannelTransform("grayscale")
        if channels == 3:
            return ChannelTransform("rgb")
        raise ValueError(
            f"cannot infer channel transform for {channels} channels; "
            f"choose one of {CHANNEL_TRANSFORMS}"
        )

    def _check_fitted(self) -> tuple[int, int, int]:
        if self.image_shape is None:
            raise RuntimeError("call fit() on train images first")
        return self.image_shape


def _as_nchw(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        return images[:, None]
    if images.ndim != 4:
        raise ValueError(f"expected images (N, H, W) or (N, C, H, W), got shape {images.shape}")
    return images


def _make_quantizer(config: dict) -> Quantizer | KMeansQuantizer:
    settings = {**DEFAULT_QUANTIZATION, **config}
    method = settings.pop("method")
    match method:
        case "uniform":
            return Quantizer(method=method, **settings)
        case "kmeans":
            settings.pop("per_position", None)
            return KMeansQuantizer(**settings)
        case _:
            raise ValueError(
                f"unknown quantization {method!r}, "
                f"expected one of {QUANTIZATION_METHODS}"
            )

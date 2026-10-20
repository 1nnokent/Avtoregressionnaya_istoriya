import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

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


class _LimitedTokenizer(Tokenizer):
    """Retain the data branch implementation, with optional coefficient truncation."""

    def __init__(self, config, max_tokens=None):
        super().__init__(config)
        self.max_tokens = max_tokens

    @property
    def sequence_length(self):
        full_length = super().sequence_length
        if self.max_tokens is not None and not 1 <= self.max_tokens <= full_length:
            raise ValueError("tokenizer.max_tokens exceeds image coefficient count")
        return self.max_tokens if self.max_tokens is not None else full_length

    def sequence(self, images):
        return super().sequence(images)[:, :self.sequence_length]

    def sequence_to_images(self, sequences):
        full = np.zeros((len(sequences), int(np.prod(self.image_shape))), dtype=np.float64)
        full[:, :self.sequence_length] = sequences
        # Pad after inverse normalization: omitted transform coefficients must be zero.
        return super().sequence_to_images(full)


class ImageTokenizer:
    """Runner adapter around the data branch Tokenizer, Normalizer and Quantizer."""

    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.config["normalization"] = {**DEFAULT_NORMALIZATION, "per_position": False,
                                        **self.config.get("normalization", {})}
        self.config["quantization"] = {**DEFAULT_QUANTIZATION, "per_position": False,
                                       **self.config.get("quantization", {})}
        core_config = {key: value for key, value in self.config.items() if key != "max_tokens"}
        self.core = _LimitedTokenizer(core_config, self.config.get("max_tokens"))

    @property
    def num_bins(self):
        return self.core.vocab_size

    @property
    def seq_len(self):
        return self.core.sequence_length

    @property
    def shape(self):
        self.core._check_fitted()
        return list(self.core.image_shape)

    def fit(self, train_images):
        if self.config["quantization"]["method"] != "uniform":
            raise ValueError("Only uniform quantization can be fitted; old quantile states are for evaluation")
        images = _as_nchw(train_images)
        if len(images) == 0 or not np.isfinite(images).all():
            raise ValueError("fit requires finite, nonempty training images")
        self.core.fit(images)
        return self

    def encode(self, images):
        return self.core.encode(images)

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] != self.seq_len:
            raise ValueError(f"Expected token matrix [N, {self.seq_len}]")
        if not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0) or np.any(tokens >= self.num_bins):
            raise ValueError("Token IDs must be integers within the vocabulary")
        return np.clip(self.core.decode(tokens), 0, 1).astype(np.float32)

    def save(self, path):
        self.core._check_fitted()
        normalizer, quantizer = self.core.normalizer, self.core.quantizer
        state = {"version": 2, "config": self.config, "shape": self.shape,
                 "normalizer": {"offset": normalizer.offset.tolist(), "spread": normalizer.spread.tolist()},
                 "quantizer": {"thresholds": quantizer.thresholds.tolist(), "centers": quantizer.centers.tolist()}}
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        config = copy.deepcopy(state["config"])
        legacy_quantile = config["quantization"]["method"] == "quantile"
        if legacy_quantile:
            # Encoding/decoding use saved thresholds and centers, irrespective of their original fit method.
            config["quantization"]["method"] = "uniform"
        obj = cls(config)
        if legacy_quantile:
            obj.config["quantization"]["method"] = "quantile"
        obj.core.image_shape = tuple(state["shape"])
        if "normalizer" in state:
            obj.core.normalizer.offset = np.asarray(state["normalizer"]["offset"])
            obj.core.normalizer.spread = np.asarray(state["normalizer"]["spread"])
            obj.core.quantizer.thresholds = np.asarray(state["quantizer"]["thresholds"])
            obj.core.quantizer.centers = np.asarray(state["quantizer"]["centers"])
        else:
            # Old run files stored the signed-log range directly, without affine scaling.
            obj.core.normalizer.offset, obj.core.normalizer.spread = np.asarray(0.0), np.asarray(1.0)
            if "edges" in state:
                edges = np.asarray(state["edges"])
                if edges.ndim == 1:
                    edges = edges[:, None]
                thresholds, centers = edges[1:-1].T, ((edges[:-1] + edges[1:]) / 2).T
            else:
                low, high = np.atleast_1d(state["low"]), np.atleast_1d(state["high"])
                width = (high - low)[:, None] / obj.num_bins
                thresholds = low[:, None] + np.arange(1, obj.num_bins)[None] * width
                centers = low[:, None] + (np.arange(obj.num_bins)[None] + 0.5) * width
            obj.core.quantizer.thresholds, obj.core.quantizer.centers = thresholds, centers
        return obj

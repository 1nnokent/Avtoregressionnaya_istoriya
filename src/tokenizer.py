"""Fitted, serializable image <-> token boundary shared by training and generation."""
import json
from pathlib import Path

import numpy as np

from src.decomposition import get_decomposition
from src.normalization import inverse_signed_log, signed_log
from src.ordering import forward_ordering, invert_ordering, ordering_indices


class ImageTokenizer:
    def __init__(self, config):
        self.config = config
        self.transform = get_decomposition(config["decomposition"], **config["decomposition_params"])
        self.num_bins = config["quantization"]["num_bins"]

    def _sequence(self, images):
        images = np.asarray(images, dtype=np.float64)
        if tuple(images.shape[1:]) != tuple(self.shape):
            raise ValueError(f"Expected NCHW images with shape {self.shape}, got {images.shape}")
        coeffs = self.transform.forward(images)
        if coeffs.shape != images.shape:
            raise ValueError("Decomposition must preserve image shape; check wavelet level and dimensions")
        seq = forward_ordering(coeffs, self.indices).reshape(len(images), -1)[:, :self.seq_len]
        if self.config["normalization"]["transform"] == "signed_log":
            seq = signed_log(seq, self.config["normalization"]["log_scale"])
        return seq

    def fit(self, train_images):
        images = np.asarray(train_images)
        if images.ndim != 4 or len(images) == 0 or not np.isfinite(images).all():
            raise ValueError("fit requires finite, nonempty NCHW training images")
        self.shape = list(images.shape[1:])
        self.indices = ordering_indices(self.config["ordering"], *self.shape[-2:], **self.config["ordering_params"])
        self.seq_len = self.config["max_tokens"] or int(np.prod(self.shape))
        if not 1 <= self.seq_len <= int(np.prod(self.shape)):
            raise ValueError("tokenizer.max_tokens exceeds image coefficient count")
        sequences = self._sequence(images)
        axis = 0 if self.config["quantization"]["per_position"] else None
        self.low = np.min(sequences, axis=axis)
        self.high = np.maximum(np.max(sequences, axis=axis), self.low + 1e-8)
        if self.config["quantization"]["method"] == "quantile":
            self.edges = np.quantile(sequences, np.linspace(0, 1, self.num_bins + 1), axis=axis)
        return self

    def encode(self, images):
        seq = self._sequence(images)
        if self.config["quantization"]["method"] == "uniform":
            tokens = np.floor((seq - self.low) / (self.high - self.low) * self.num_bins)
        elif self.config["quantization"]["per_position"]:
            tokens = np.stack([np.searchsorted(self.edges[1:-1, i], seq[:, i], side="right")
                               for i in range(self.seq_len)], axis=1)
        else:
            tokens = np.searchsorted(self.edges[1:-1], seq, side="right")
        return np.clip(tokens, 0, self.num_bins - 1).astype(np.int64)

    def decode(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or tokens.shape[1] != self.seq_len:
            raise ValueError(f"Expected token matrix [N, {self.seq_len}]")
        if not np.issubdtype(tokens.dtype, np.integer) or np.any(tokens < 0) or np.any(tokens >= self.num_bins):
            raise ValueError("Token IDs must be integers within the vocabulary")
        if self.config["quantization"]["method"] == "uniform":
            seq = self.low + (tokens + 0.5) * (self.high - self.low) / self.num_bins
        else:
            centers = (self.edges[:-1] + self.edges[1:]) / 2
            seq = centers[tokens, np.arange(self.seq_len)] if centers.ndim == 2 else centers[tokens]
        if self.config["normalization"]["transform"] == "signed_log":
            seq = inverse_signed_log(seq, self.config["normalization"]["log_scale"])
        # Omitted coefficients are exactly zero in the transform domain.
        full = np.zeros((len(tokens), int(np.prod(self.shape))), dtype=np.float64)
        full[:, :self.seq_len] = seq
        ordered = full.reshape(len(tokens), self.shape[0], -1)
        coeffs = invert_ordering(ordered, self.indices, *self.shape[-2:])
        return np.clip(self.transform.inverse(coeffs), 0, 1).astype(np.float32)

    def save(self, path):
        state = {"config": self.config, "shape": self.shape, "seq_len": self.seq_len,
                 "low": np.asarray(self.low).tolist(), "high": np.asarray(self.high).tolist()}
        if hasattr(self, "edges"):
            state["edges"] = self.edges.tolist()
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(state["config"])
        obj.shape, obj.seq_len = state["shape"], state["seq_len"]
        obj.low, obj.high = np.asarray(state["low"]), np.asarray(state["high"])
        obj.indices = ordering_indices(obj.config["ordering"], *obj.shape[-2:], **obj.config["ordering_params"])
        if "edges" in state:
            obj.edges = np.asarray(state["edges"])
        return obj

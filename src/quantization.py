from dataclasses import dataclass, field

import numpy as np

METHODS = ("uniform",)

@dataclass
class Quantizer:
    method: str = "uniform"
    num_bins: int = 256
    per_position: bool = False
    thresholds: np.ndarray | None = field(default=None, repr=False)
    centers: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown quantization {self.method!r}, expected one of {METHODS}")

    def fit(self, sequences: np.ndarray) -> "Quantizer":
        sequences = np.asarray(sequences, dtype=np.float64)
        if self.per_position:
            columns = sequences.T
        else:
            columns = [sequences.ravel()]
        fitted = [self._fit_column(column) for column in columns]
        self.thresholds = np.stack([thresholds for thresholds, _ in fitted])
        self.centers = np.stack([centers for _, centers in fitted])
        return self

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        self._check_fitted()
        sequences = np.asarray(sequences)
        if not self.per_position:
            return np.searchsorted(self.thresholds[0], sequences, side="right").astype(np.int64)
        tokens = np.empty(sequences.shape, dtype=np.int64)
        for position, thresholds in enumerate(self.thresholds):
            tokens[..., position] = np.searchsorted(thresholds, sequences[..., position], side="right")
        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._check_fitted()
        tokens = np.asarray(tokens)
        if not self.per_position:
            return self.centers[0][tokens]
        return np.take_along_axis(self.centers.T, tokens.reshape(-1, tokens.shape[-1]), axis=0).reshape(tokens.shape)

    def _fit_column(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(values.min(), values.max(), self.num_bins + 1)
        return edges[1:-1], (edges[:-1] + edges[1:]) / 2

    def _check_fitted(self):
        if self.thresholds is None:
            raise RuntimeError("call fit() on train data first")

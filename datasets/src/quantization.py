from dataclasses import dataclass, field

import numpy as np

METHODS = ("uniform",)

@dataclass
class Quantizer:
    method: str = "uniform"
    num_bins: int = 256
    per_position: bool = True
    thresholds: np.ndarray | None = field(default=None, repr=False)
    centers: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown quantization {self.method!r}, expected one of {METHODS}")

    def fit(self, sequences: np.ndarray) -> "Quantizer":
        sequences = np.asarray(sequences, dtype=np.float64)
        if self.per_position:
            columns = [sequences] if sequences.ndim == 1 else sequences.T
        else:
            columns = [sequences.ravel()]
        fitted = [self._fit_column(column) for column in columns]
        self.thresholds = np.stack([thresholds for thresholds, _ in fitted])
        self.centers = np.stack([centers for _, centers in fitted])
        return self

    def encode(self, sequences: np.ndarray) -> np.ndarray:
        self._check_fitted()
        sequences = np.asarray(sequences)
        if not self.per_position or sequences.ndim == 1:
            return np.searchsorted(self.thresholds[0], sequences, side="right").astype(np.int64)
        tokens = np.empty(sequences.shape, dtype=np.int64)
        for position, thresholds in enumerate(self.thresholds):
            tokens[..., position] = np.searchsorted(thresholds, sequences[..., position], side="right")
        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._check_fitted()
        tokens = np.asarray(tokens)
        if not self.per_position or tokens.ndim == 1:
            return self.centers[0][tokens]
        return np.take_along_axis(self.centers.T, tokens.reshape(-1, tokens.shape[-1]), axis=0).reshape(tokens.shape)

    def _fit_column(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(values.min(), values.max(), self.num_bins + 1)
        return edges[1:-1], (edges[:-1] + edges[1:]) / 2

    def state_dict(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        return {
            "thresholds": np.asarray(self.thresholds).copy(),
            "centers": np.asarray(self.centers).copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.thresholds = np.asarray(state["thresholds"], dtype=np.float64).copy()
        self.centers = np.asarray(state["centers"], dtype=np.float64).copy()

    def _check_fitted(self):
        if self.thresholds is None:
            raise RuntimeError("call fit() on train data first")

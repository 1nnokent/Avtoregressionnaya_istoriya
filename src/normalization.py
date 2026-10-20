
from dataclasses import dataclass, field

import numpy as np

TRANSFORMS = ("none", "signed_log")
SCALINGS = ("minmax", "standard")


def signed_log(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(scale * np.abs(x))


def inverse_signed_log(y: np.ndarray, scale: float = 1.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    return np.sign(y) * np.expm1(np.abs(y)) / scale


@dataclass
class Normalizer:
    transform: str = "signed_log"
    log_scale: float = 1.0
    scaling: str = "minmax"
    per_position: bool = False
    offset: np.ndarray | None = field(default=None, repr=False)
    spread: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}, expected one of {TRANSFORMS}")
        if self.scaling not in SCALINGS:
            raise ValueError(f"unknown scaling {self.scaling!r}, expected one of {SCALINGS}")

    def apply_transform(self, x: np.ndarray) -> np.ndarray:
        if self.transform == "signed_log":
            return signed_log(x, self.log_scale)
        return np.asarray(x, dtype=np.float64)

    def invert_transform(self, y: np.ndarray) -> np.ndarray:
        if self.transform == "signed_log":
            return inverse_signed_log(y, self.log_scale)
        return y

    def fit(self, sequences: np.ndarray) -> "Normalizer":
        y = self.apply_transform(sequences)
        axis = 0 if self.per_position else None
        if self.scaling == "minmax":
            offset, spread = y.min(axis=axis), y.max(axis=axis) - y.min(axis=axis)
        else:
            offset, spread = y.mean(axis=axis), y.std(axis=axis)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.spread = np.where(spread > 0, spread, 1.0)
        return self

    def __call__(self, sequences: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (self.apply_transform(sequences) - self.offset) / self.spread

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.invert_transform(normalized * self.spread + self.offset)

    def _check_fitted(self):
        if self.offset is None:
            raise RuntimeError("call fit() on train data first")

from dataclasses import dataclass, field

import numpy as np

from .kmeans import assign_codebook, decode_codebook, fit_codebook

METHODS = ("uniform",)
KMEANS_GROUPINGS = ("scalar_channels", "joint_channels")

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


@dataclass
class KMeansQuantizer:
    num_bins: int = 256
    grouping: str = "scalar_channels"
    sample_size: int = 200_000
    iterations: int = 20
    seed: int = 0
    centers: np.ndarray | None = field(default=None, repr=False)
    channels: int | None = field(default=None, repr=False)
    positions: int | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.grouping not in KMEANS_GROUPINGS:
            raise ValueError(
                f"unknown KMeans grouping {self.grouping!r}, "
                f"expected one of {KMEANS_GROUPINGS}"
            )

    def fit(self, coefficients: np.ndarray) -> "KMeansQuantizer":
        coefficients = _as_ncl(coefficients)
        self.channels = coefficients.shape[1]
        self.positions = coefficients.shape[2]
        match self.grouping:
            case "scalar_channels":
                self.centers = np.stack(
                    [
                        fit_codebook(
                            coefficients[:, channel].reshape(-1),
                            self.num_bins,
                            self.sample_size,
                            self.iterations,
                            self.seed + channel,
                        )
                        for channel in range(self.channels)
                    ]
                )
            case "joint_channels":
                observations = coefficients.transpose(0, 2, 1).reshape(
                    -1, self.channels
                )
                self.centers = fit_codebook(
                    observations,
                    self.num_bins,
                    self.sample_size,
                    self.iterations,
                    self.seed,
                )
        return self

    def encode(self, coefficients: np.ndarray) -> np.ndarray:
        self._check_fitted()
        coefficients = self._validate_coefficients(coefficients)
        batch_size = len(coefficients)
        match self.grouping:
            case "scalar_channels":
                tokens = np.stack(
                    [
                        assign_codebook(
                            coefficients[:, channel].reshape(-1),
                            self.centers[channel],
                        ).reshape(batch_size, self.positions)
                        for channel in range(self.channels)
                    ],
                    axis=1,
                )
                return tokens.reshape(batch_size, self.channels * self.positions)
            case "joint_channels":
                observations = coefficients.transpose(0, 2, 1).reshape(
                    -1, self.channels
                )
                return assign_codebook(observations, self.centers).reshape(
                    batch_size, self.positions
                )

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        self._check_fitted()
        tokens = np.asarray(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None]
        expected_length = (
            self.channels * self.positions
            if self.grouping == "scalar_channels"
            else self.positions
        )
        if tokens.ndim != 2 or tokens.shape[1] != expected_length:
            raise ValueError(
                f"expected tokens shaped (N, {expected_length}), got {tokens.shape}"
            )
        if tokens.size and (tokens.min() < 0 or tokens.max() >= self.num_bins):
            raise ValueError(f"tokens must be in [0, {self.num_bins})")
        match self.grouping:
            case "scalar_channels":
                channel_tokens = tokens.reshape(
                    -1, self.channels, self.positions
                )
                return np.stack(
                    [
                        decode_codebook(
                            channel_tokens[:, channel], self.centers[channel]
                        )[..., 0]
                        for channel in range(self.channels)
                    ],
                    axis=1,
                )
            case "joint_channels":
                vectors = decode_codebook(tokens, self.centers)
                return vectors.transpose(0, 2, 1)

    def state_dict(self) -> dict[str, np.ndarray]:
        self._check_fitted()
        return {
            "centers": np.asarray(self.centers).copy(),
            "channels": np.asarray(self.channels, dtype=np.int64),
            "positions": np.asarray(self.positions, dtype=np.int64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        centers = np.asarray(state["centers"], dtype=np.float64)
        channels = int(np.asarray(state["channels"]).item())
        positions = int(np.asarray(state["positions"]).item())
        expected_shape = (
            (channels, self.num_bins, 1)
            if self.grouping == "scalar_channels"
            else (self.num_bins, channels)
        )
        if centers.shape != expected_shape:
            raise ValueError(
                f"expected {self.grouping.removesuffix('_channels')} KMeans centers "
                f"shaped {expected_shape}, "
                f"got {centers.shape}"
            )
        self.centers = centers.copy()
        self.channels = channels
        self.positions = positions

    def _validate_coefficients(self, coefficients: np.ndarray) -> np.ndarray:
        coefficients = _as_ncl(coefficients)
        if coefficients.shape[1] != self.channels:
            raise ValueError(
                f"quantizer was fitted on {self.channels} channels, "
                f"got {coefficients.shape[1]}"
            )
        if coefficients.shape[2] != self.positions:
            raise ValueError(
                f"quantizer was fitted on {self.positions} positions, "
                f"got {coefficients.shape[2]}"
            )
        return coefficients

    def _check_fitted(self) -> None:
        if self.centers is None:
            raise RuntimeError("call fit() on train coefficients first")


def _as_ncl(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError(
            f"expected coefficients shaped (N, C, L), got {coefficients.shape}"
        )
    return coefficients

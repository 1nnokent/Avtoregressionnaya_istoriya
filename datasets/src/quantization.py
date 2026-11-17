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
        thresholds, _ = self._fitted_state()
        sequences = np.asarray(sequences)
        if not self.per_position or sequences.ndim == 1:
            return np.searchsorted(thresholds[0], sequences, side="right").astype(np.int64)
        tokens = np.empty(sequences.shape, dtype=np.int64)
        for position, position_thresholds in enumerate(thresholds):
            tokens[..., position] = np.searchsorted(
                position_thresholds, sequences[..., position], side="right"
            )
        return tokens

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        _, centers = self._fitted_state()
        tokens = np.asarray(tokens)
        if not self.per_position or tokens.ndim == 1:
            return centers[0][tokens]
        return np.take_along_axis(
            centers.T, tokens.reshape(-1, tokens.shape[-1]), axis=0
        ).reshape(tokens.shape)

    def _fit_column(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(values.min(), values.max(), self.num_bins + 1)
        return edges[1:-1], (edges[:-1] + edges[1:]) / 2

    def state_dict(self) -> dict[str, np.ndarray]:
        thresholds, centers = self._fitted_state()
        return {
            "thresholds": thresholds.copy(),
            "centers": centers.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.thresholds = np.asarray(state["thresholds"], dtype=np.float64).copy()
        self.centers = np.asarray(state["centers"], dtype=np.float64).copy()

    def _fitted_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.thresholds is None or self.centers is None:
            raise RuntimeError("call fit() on train data first")
        return self.thresholds, self.centers


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
        channels = coefficients.shape[1]
        positions = coefficients.shape[2]
        self.channels = channels
        self.positions = positions
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
                        for channel in range(channels)
                    ]
                )
            case "joint_channels":
                observations = coefficients.transpose(0, 2, 1).reshape(
                    -1, channels
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
        centers, channels, positions = self._fitted_state()
        coefficients = self._validate_coefficients(
            coefficients, channels, positions
        )
        batch_size = len(coefficients)
        match self.grouping:
            case "scalar_channels":
                tokens = np.stack(
                    [
                        assign_codebook(
                            coefficients[:, channel].reshape(-1),
                            centers[channel],
                        ).reshape(batch_size, positions)
                        for channel in range(channels)
                    ],
                    axis=1,
                )
                return tokens.reshape(batch_size, channels * positions)
            case "joint_channels":
                observations = coefficients.transpose(0, 2, 1).reshape(
                    -1, channels
                )
                return assign_codebook(observations, centers).reshape(
                    batch_size, positions
                )
        raise AssertionError(f"unsupported KMeans grouping {self.grouping!r}")

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        centers, channels, positions = self._fitted_state()
        tokens = np.asarray(tokens)
        if tokens.ndim == 1:
            tokens = tokens[None]
        expected_length = (
            channels * positions
            if self.grouping == "scalar_channels"
            else positions
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
                    -1, channels, positions
                )
                return np.stack(
                    [
                        decode_codebook(
                            channel_tokens[:, channel], centers[channel]
                        )[..., 0]
                        for channel in range(channels)
                    ],
                    axis=1,
                )
            case "joint_channels":
                vectors = decode_codebook(tokens, centers)
                return vectors.transpose(0, 2, 1)
        raise AssertionError(f"unsupported KMeans grouping {self.grouping!r}")

    @property
    def sequence_length(self) -> int:
        _, channels, positions = self._fitted_state()
        if self.grouping == "scalar_channels":
            return channels * positions
        return positions

    def state_dict(self) -> dict[str, np.ndarray]:
        centers, channels, positions = self._fitted_state()
        return {
            "centers": centers.copy(),
            "channels": np.asarray(channels, dtype=np.int64),
            "positions": np.asarray(positions, dtype=np.int64),
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

    def _validate_coefficients(
        self, coefficients: np.ndarray, channels: int, positions: int
    ) -> np.ndarray:
        coefficients = _as_ncl(coefficients)
        if coefficients.shape[1] != channels:
            raise ValueError(
                f"quantizer was fitted on {channels} channels, "
                f"got {coefficients.shape[1]}"
            )
        if coefficients.shape[2] != positions:
            raise ValueError(
                f"quantizer was fitted on {positions} positions, "
                f"got {coefficients.shape[2]}"
            )
        return coefficients

    def _fitted_state(self) -> tuple[np.ndarray, int, int]:
        if self.centers is None or self.channels is None or self.positions is None:
            raise RuntimeError("call fit() on train coefficients first")
        return self.centers, self.channels, self.positions


def _as_ncl(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError(
            f"expected coefficients shaped (N, C, L), got {coefficients.shape}"
        )
    return coefficients

import numpy as np

CHANNEL_TRANSFORMS = ("grayscale", "ycbcr", "rgb")
_RGB_TO_YCBCR = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)
_YCBCR_TO_RGB = np.linalg.inv(_RGB_TO_YCBCR)


class ChannelTransform:
    def __init__(self, name: str):
        if name not in CHANNEL_TRANSFORMS:
            raise ValueError(
                f"unknown channel transform {name!r}, expected one of {CHANNEL_TRANSFORMS}"
            )
        self.name = name

    def forward(self, images: np.ndarray) -> np.ndarray:
        images = self._validate(images)
        if self.name in ("grayscale", "rgb"):
            return images.copy()
        return _apply_channel_matrix(images, _RGB_TO_YCBCR)

    def inverse(self, images: np.ndarray) -> np.ndarray:
        images = self._validate(images)
        if self.name in ("grayscale", "rgb"):
            return images.copy()
        return _apply_channel_matrix(images, _YCBCR_TO_RGB)

    def _validate(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError(f"expected NCHW images, got shape {images.shape}")
        expected_channels = 1 if self.name == "grayscale" else 3
        if images.shape[1] != expected_channels:
            suffix = "" if expected_channels == 1 else "s"
            raise ValueError(
                f"{self.name} transform expects {expected_channels} channel{suffix}, "
                f"got {images.shape[1]}"
            )
        return images


def _apply_channel_matrix(images: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    dtype = np.result_type(images.dtype, np.float32)
    values = images.astype(dtype, copy=False)
    coefficients = matrix.astype(dtype, copy=False)
    return np.einsum("ij,njhw->nihw", coefficients, values)

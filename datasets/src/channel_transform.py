import numpy as np


CHANNEL_TRANSFORMS = ("grayscale", "ycbcr", "rgb")


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
        raise NotImplementedError("ycbcr channel transform is not implemented")

    def inverse(self, images: np.ndarray) -> np.ndarray:
        images = self._validate(images)
        if self.name in ("grayscale", "rgb"):
            return images.copy()
        raise NotImplementedError("ycbcr channel transform is not implemented")

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

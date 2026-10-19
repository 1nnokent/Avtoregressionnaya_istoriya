from pathlib import Path

import numpy as np
from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_mnist(train: bool, root: str | Path = DATA_DIR, transform=None) -> datasets.MNIST:
    return datasets.MNIST(root=root, train=train, transform=transform, download=True)


def load_normed_mnist_images(train: bool, root: str | Path = DATA_DIR) -> np.ndarray:
    return load_mnist(train, root).data.numpy().astype(np.float64) / 255.0


def train_val_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split of range(n) into train and validation indices"""
    if not (0 < val_fraction < 1):
        raise ValueError("val fraction must be in [0, 1]")

    permutation = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_fraction)
    return permutation[n_val:], permutation[:n_val]

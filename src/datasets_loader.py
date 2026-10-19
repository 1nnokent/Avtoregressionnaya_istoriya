from pathlib import Path

from torchvision import datasets

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_mnist(train: bool, root: str | Path = DATA_DIR, transform=None) -> datasets.MNIST:
    return datasets.MNIST(root=root, train=train, transform=transform, download=True)
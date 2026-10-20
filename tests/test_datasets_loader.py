import numpy as np
import pytest
from PIL import Image

from src.datasets_loader import DATA_DIR, load_images, train_val_indices

has_mnist = pytest.mark.skipif(
    not (DATA_DIR / "MNIST").exists(), reason="MNIST is not downloaded"
)
has_cifar10 = pytest.mark.skipif(
    not (DATA_DIR / "cifar-10-batches-py" / "test_batch").exists(),
    reason="CIFAR-10 is not downloaded",
)


def test_train_val_split_is_disjoint():
    train, val = train_val_indices(100, 0.2, seed=1)
    assert (
        len(val) == 20
        and set(train).isdisjoint(val)
        and len(set(train) | set(val)) == 100
    )


@has_mnist
def test_load_mnist_splits():
    for split, expected in [("train", 54000), ("val", 6000), ("test", 10000)]:
        images = load_images("mnist", split, download=False)
        assert images.shape == (expected, 1, 28, 28) and images.dtype == np.float32
        assert 0 <= images.min() and images.max() <= 1
    assert load_images("mnist", "train", limit=100, download=False).shape == (100, 1, 28, 28)


@has_cifar10
def test_load_cifar10():
    images = load_images("cifar10", "test", gray=False, limit=100, download=False)
    assert images.shape == (100, 3, 32, 32) and images.dtype == np.float32
    assert 0 <= images.min() and images.max() <= 1
    assert load_images("cifar10", "test", limit=100, download=False).shape == (100, 1, 32, 32)


def test_cifar_native_size_and_explicit_resize_without_download(monkeypatch, tmp_path):
    images = [(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)), 0)] * 4
    monkeypatch.setattr("src.datasets_loader.datasets.CIFAR10", lambda *args, **kwargs: images)
    native = load_images("cifar10", "test", gray=False, limit=2, root=tmp_path, download=False)
    resized = load_images("cifar10", "test", size=14, limit=2, root=tmp_path, download=False)
    assert native.shape == (2, 3, 32, 32)
    assert resized.shape == (2, 1, 14, 14)

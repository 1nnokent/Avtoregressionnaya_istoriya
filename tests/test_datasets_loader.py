import numpy as np
import pytest

from datasets.src import datasets_loader
from datasets.src.datasets_loader import DATA_DIR, load_images, train_val_indices

has_mnist = pytest.mark.skipif(
    not (DATA_DIR / "MNIST").exists(), reason="MNIST is not downloaded"
)
has_cifar10 = pytest.mark.skipif(
    not (DATA_DIR / "cifar10" / "test.parquet").exists(),
    reason="CIFAR-10 is not downloaded",
)
has_celeba = pytest.mark.skipif(
    not (DATA_DIR / "celeba" / "test.parquet").exists(),
    reason="CelebA is not downloaded",
)


def test_train_val_split_is_disjoint():
    train, val = train_val_indices(100, 0.2, seed=1)
    assert (
        len(val) == 20
        and set(train).isdisjoint(val)
        and len(set(train) | set(val)) == 100
    )


def test_celeba_parquet_paths_cover_all_shards_and_reuse_first_cache(
    monkeypatch, tmp_path
):
    downloads = []

    def fake_download(url, path):
        downloads.append((url, path))
        return path

    monkeypatch.setattr(datasets_loader, "_download", fake_download)

    paths = datasets_loader.celeba_parquet_paths("train", tmp_path)

    assert len(paths) == 19
    assert paths[0] == tmp_path / "celeba" / "train.parquet"
    assert paths[-1].name == "train-00018-of-00019.parquet"
    assert "train-00000-of-00019.parquet" in downloads[0][0]
    assert "train-00018-of-00019.parquet" in downloads[-1][0]


def test_celeba_parquet_paths_reject_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="unknown CelebA split"):
        datasets_loader.celeba_parquet_paths("validation", tmp_path)


@has_mnist
def test_load_mnist_splits():
    for split, expected in [("train", 54000), ("val", 6000), ("test", 10000)]:
        images = load_images("mnist", split)
        assert images.shape == (expected, 1, 28, 28) and images.dtype == np.float32
        assert 0 <= images.min() and images.max() <= 1
    assert load_images("mnist", "train", limit=100).shape == (100, 1, 28, 28)


@has_cifar10
def test_load_cifar10():
    images = load_images("cifar10", "test", gray=False, limit=100)
    assert images.shape == (100, 3, 32, 32) and images.dtype == np.float32
    assert 0 <= images.min() and images.max() <= 1
    assert load_images("cifar10", "test", limit=100).shape == (100, 1, 32, 32)
    assert not np.array_equal(images, load_images("cifar10", "test", gray=False, limit=100, seed=1))


@has_celeba
def test_load_celeba():
    images = load_images("celeba", "test", gray=False, limit=20)
    assert images.shape == (20, 3, 64, 64) and images.dtype == np.float32
    assert 0 <= images.min() and images.max() <= 1
    assert load_images("celeba", "test", limit=20).shape == (20, 1, 64, 64)
    assert not np.array_equal(images, load_images("celeba", "test", gray=False, limit=20, seed=1))

import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("mnist", "cifar10", "celeba", "imagenet")
SPLITS = ("train", "val", "test")
CIFAR10_HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"
CELEBA_HF_URL = "https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/img_align%2Bidentity%2Battr/{shard}.parquet"
CELEBA_HF_SHARDS = {"train": "train-00000-of-00019", "val": "valid-00000-of-00003", "test": "test-00000-of-00003"}
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def train_val_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random split of range(n) into train and validation indices"""
    if not (0 < val_fraction < 1):
        raise ValueError("val fraction must be in [0, 1]")

    permutation = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_fraction)
    return permutation[n_val:], permutation[:n_val]


def load_images(
    ds_name: str,
    split: str = "train",
    gray: bool = True,
    size: int = 64,
    limit: int | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
    root: str | Path = DATA_DIR,
) -> np.ndarray:
    if ds_name not in DATASETS:
        raise ValueError(f"unknown dataset {ds_name!r}, expected one of {DATASETS}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    dataset = _torchvision_dataset(ds_name, split, Path(root), size)
    indices = np.random.default_rng(seed).permutation(len(dataset))

    if split != "test" and ds_name != "celeba":
        train_idx, val_idx = train_val_indices(len(dataset), val_fraction, seed)
        indices = train_idx if split == "train" else val_idx
    if limit is not None:
        indices = indices[:limit]

    images = _to_array(ds_name, dataset, indices, size).astype(np.float32) / 255.0
    if gray and images.shape[1] == 3:
        images = np.einsum("c,nchw->nhw", LUMA, images)[:, None]
    return images


def _torchvision_dataset(ds_name: str, split: str, root: Path, size: int):
    if ds_name == "mnist":
        return datasets.MNIST(root, train=split != "test", download=True)
    if ds_name == "cifar10":
        return _load_cifar10("test" if split == "test" else "train", root)

    if ds_name == "celeba":
        return _load_celeba(split, root)

    folder = root / "imagenet" / ("val" if split == "test" else "train")
    if not folder.exists():
        raise FileNotFoundError(f"expected ImageNet in ImageFolder layout at {folder}/<class>/<image>")
    transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.PILToTensor()])

    return datasets.ImageFolder(folder, transform=transform)


def _to_array(ds_name: str, dataset, indices: np.ndarray, size: int) -> np.ndarray:
    if ds_name == "mnist":
        return dataset.data.numpy()[indices, None]
    if ds_name == "cifar10":
        return dataset[indices]
    if ds_name == "celeba":
        transform = T.Compose([T.CenterCrop(178), T.Resize(size), T.PILToTensor()])
        return torch.stack([transform(Image.open(io.BytesIO(dataset[i])).convert("RGB")) for i in indices]).numpy()
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=256)
    return torch.cat([images for images, _ in loader]).numpy()


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.rename(path)
    return path


def _load_celeba(split: str, root: Path) -> list[bytes]:
    url = CELEBA_HF_URL.format(shard=CELEBA_HF_SHARDS[split])
    path = _download(url, root / "celeba" / f"{split}.parquet")
    return [image["bytes"] for image in pq.read_table(path, columns=["image"]).column("image").to_pylist()]


def _load_cifar10(split: str, root: Path) -> np.ndarray:
    path = _download(CIFAR10_HF_URL.format(split=split), root / "cifar10" / f"{split}.parquet")
    pngs = pq.read_table(path, columns=["img"]).column("img").to_pylist()
    return np.stack([np.asarray(Image.open(io.BytesIO(png["bytes"]))) for png in pngs]).transpose(0, 3, 1, 2)

"""Deterministic train/validation split; official test data is never used for fitting."""
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms as T

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASETS = ("mnist", "cifar10", "celeba", "imagenet", "npz")
SPLITS = ("train", "val", "test")


def train_val_indices(n, val_fraction=0.1, seed=0):
    if n < 2 or not 0 < val_fraction < 1:
        raise ValueError("Split needs >=2 images and 0 < val_fraction < 1")
    permutation = np.random.default_rng(seed).permutation(n)
    n_val = max(1, min(n - 1, int(n * val_fraction)))
    return permutation[n_val:], permutation[:n_val]


def load_images(ds_name, split="train", gray=True, size=None, limit=None,
                val_fraction=0.1, seed=0, root=DATA_DIR, download=True, return_labels=False):
    if ds_name not in DATASETS or split not in SPLITS:
        raise ValueError(f"Expected dataset in {DATASETS} and split in {SPLITS}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if size is None:
        size = {"mnist": 28, "cifar10": 32, "celeba": 64, "imagenet": 64}.get(ds_name)
    root = Path(root)
    if ds_name == "npz":
        # Local dataset: uint8 NCHW/NHW arrays; labels are optional except for classifier training.
        with np.load(root, allow_pickle=False) as data:
            prefix = "test" if split == "test" else "train"
            source = data[f"{prefix}_images"]
            labels = data[f"{prefix}_labels"] if f"{prefix}_labels" in data else np.full(len(source), -1)
        if source.ndim == 3:
            source = source[:, None]
        if source.ndim != 4 or source.dtype != np.uint8 or source.shape[1] not in (1, 3):
            raise ValueError("NPZ images must be uint8 NHW or NCHW (1 or 3 channels)")
        dataset = None
        n = len(source)
    else:
        transform = T.Compose([T.CenterCrop(178), T.Resize((size, size))]) if ds_name == "celeba" else None
        if ds_name == "mnist":
            dataset = datasets.MNIST(root, train=split != "test", download=download)
        elif ds_name == "cifar10":
            dataset = datasets.CIFAR10(root, train=split != "test", download=download)
        elif ds_name == "celeba":
            dataset = datasets.CelebA(root, split={"train": "train", "val": "valid", "test": "test"}[split],
                                      download=False, target_type="identity", transform=transform)
        else:
            folder = root / "imagenet" / ("val" if split == "test" else "train")
            dataset = datasets.ImageFolder(folder, transform=T.Compose([T.Resize(size), T.CenterCrop(size)]))
        n = len(dataset)
    indices = np.arange(n)
    if split != "test" and ds_name != "celeba":
        train_idx, val_idx = train_val_indices(n, val_fraction, seed)
        indices = train_idx if split == "train" else val_idx
    indices = indices[:limit]
    if len(indices) == 0:
        raise ValueError(f"Empty {ds_name}/{split} split")
    images, targets = [], []
    for idx in indices:
        if dataset is None:
            array = source[idx]
            array = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
            image, label = Image.fromarray(array), int(labels[idx])
        else:
            image, label = dataset[int(idx)]
        if gray:
            image = image.convert("L")
        elif ds_name != "mnist":
            image = image.convert("RGB")
        if size is not None:
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        images.append(array[None] if array.ndim == 2 else array.transpose(2, 0, 1))
        targets.append(int(label))
    result = np.stack(images)
    return (result, np.asarray(targets)) if return_labels else result


def load_splits(config, seed):
    common = {k: v for k, v in config.items() if not k.endswith("_limit")}
    return {split: load_images(**common, split=split, limit=config[f"{split}_limit"],
                               seed=seed, return_labels=True) for split in SPLITS}

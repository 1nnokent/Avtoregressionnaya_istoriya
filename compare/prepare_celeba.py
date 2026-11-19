"""Build the common CelebA cache from the project's parquet downloads."""
import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from datasets.src.datasets_loader import (
    COMMON_BASE_SIZE,
    SPLITS,
    celeba_parquet_paths,
)

RGB_CHANNELS = 3
SHUFFLE_SEED = 42
DEFAULT_ROOT = "data"
DEFAULT_OUT_DIR = "data/celeba_common"


def fit_image(encoded: bytes, size: int = COMMON_BASE_SIZE) -> np.ndarray:
    with Image.open(io.BytesIO(encoded)) as source:
        img = ImageOps.fit(source.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def cache_path(out_dir: str | Path, split: str) -> Path:
    return Path(out_dir) / f"celeba_{split}_{COMMON_BASE_SIZE}px.npy"


def build_cache(
    split: str,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    limit: int | None = None,
    root: str | Path = DEFAULT_ROOT,
) -> Path:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    parquet_paths = celeba_parquet_paths(split, root)
    shard_sizes = [pq.ParquetFile(path).metadata.num_rows for path in parquet_paths]
    total_size = sum(shard_sizes)
    if limit is None or limit >= total_size:
        selected = np.arange(total_size)
    else:
        selected = np.sort(np.random.default_rng(SHUFFLE_SEED).choice(total_size, limit, replace=False))

    path = cache_path(out_dir, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".npy.part")
    array = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected), COMMON_BASE_SIZE, COMMON_BASE_SIZE, RGB_CHANNELS),
    )

    output_index = 0
    shard_start = 0
    with tqdm(total=len(selected), desc=split) as progress:
        for parquet_path, shard_size in zip(parquet_paths, shard_sizes, strict=True):
            shard_end = shard_start + shard_size
            first = np.searchsorted(selected, shard_start, side="left")
            last = np.searchsorted(selected, shard_end, side="left")
            local_indices = selected[first:last] - shard_start
            if len(local_indices):
                images = pq.read_table(parquet_path, columns=["image"]).column("image")
                for local_index in local_indices:
                    encoded = images[int(local_index)].as_py()["bytes"]
                    array[output_index] = fit_image(encoded)
                    output_index += 1
                    progress.update()
            shard_start = shard_end

    array.flush()
    del array
    partial.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    path = build_cache(args.split, args.out_dir, args.limit, args.root)
    print("Saved", path, np.load(path, mmap_mode="r").shape)


if __name__ == "__main__":
    main()

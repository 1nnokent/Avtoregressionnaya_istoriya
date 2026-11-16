"""Общий кеш CelebA для всех моделей: uint8 [N, H, W, 3], ImageOps.fit + LANCZOS."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

DATASET_ID = 'tpremoli/CelebA-attrs'
BASE_SIZE = 64
RGB_CHANNELS = 3
SHUFFLE_SEED = 42
HF_SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}
DEFAULT_OUT_DIR = 'data/celeba_common'


def fit_image(img, size=BASE_SIZE):
    img = ImageOps.fit(img.convert('RGB'), (size, size), method=Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def cache_path(out_dir, split):
    return Path(out_dir) / f'celeba_{split}_{BASE_SIZE}px.npy'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=list(HF_SPLITS), required=True)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, split=HF_SPLITS[args.split])
    if args.limit is not None:
        ds = ds.shuffle(seed=SHUFFLE_SEED).select(range(min(args.limit, len(ds))))

    path = cache_path(args.out_dir, args.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                      shape=(len(ds), BASE_SIZE, BASE_SIZE, RGB_CHANNELS))
    for index in tqdm(range(len(ds)), desc=args.split):
        array[index] = fit_image(ds[index]['image'])
    array.flush()
    print('Saved', path, array.shape)


if __name__ == '__main__':
    main()
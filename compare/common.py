import json
from pathlib import Path

import numpy as np

RGB_CHANNELS = 3
PROGRESS_SUFFIX = '.progress'
TIMING_SUFFIX = '.json'


def open_output(path, num_images, image_size):
    """.npy для дозаписи; файл .progress хранит число готовых картинок (докачка после обрыва Colab)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    progress = path.with_suffix(PROGRESS_SUFFIX)
    shape = (num_images, image_size, image_size, RGB_CHANNELS)
    if path.exists() and progress.exists():
        array = np.lib.format.open_memmap(path, mode='r+')
        if array.shape != shape:
            raise ValueError(f'{path}: shape {array.shape}, expected {shape}')
        done = int(progress.read_text())
    else:
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=shape)
        done = 0
        progress.write_text(str(done))
    return array, progress, done


def save_timing(path, **values):
    Path(path).with_suffix(TIMING_SUFFIX).write_text(json.dumps(values, indent=2))
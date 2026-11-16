"""FID / precision / recall для всех моделей одним кодом + потолки представлений."""
import argparse
import json
from pathlib import Path

import numpy as np

from datasets.src.datasets_loader import downsample_uint8
from datasets.src.tokenizer import Tokenizer
from metrics import scores
from metrics.encoders import ENCODERS
from metrics.visualization import save_grid

UINT8_MAX = 255.0
PIXEL_HALF_RANGE = 127.5
PROGRESS_SUFFIX = '.progress'
TIMING_SUFFIX = '.json'
PALETTE_CHUNK = 64
TOKENIZER_CHUNK = 512
DEFAULT_K = 3


def to_nchw_float(images_uint8):
    return np.asarray(images_uint8).transpose(0, 3, 1, 2).astype(np.float32) / UINT8_MAX


def to_nhwc_uint8(images_nchw_float):
    return np.rint(np.clip(images_nchw_float, 0, 1) * UINT8_MAX).astype(np.uint8).transpose(0, 2, 3, 1)


def load_samples(path, num_images):
    path = Path(path)
    progress = path.with_suffix(PROGRESS_SUFFIX)
    if progress.exists() and int(progress.read_text()) < num_images:
        raise ValueError(f'{path}: готово только {progress.read_text()} картинок')
    if path.suffix == '.npz':                          # generated.npz спектральной модели
        images = to_nhwc_uint8(np.load(path)['images'])
    else:
        images = np.load(path, mmap_mode='r')
    if len(images) < num_images:
        raise ValueError(f'{path}: {len(images)} < {num_images}')
    return np.asarray(images[:num_images])


def palette_roundtrip(images_uint8, palette):
    """Потолок iGPT: каждый пиксель -> ближайший из 512 цветов -> обратно."""
    palette_sq = (palette ** 2).sum(-1)
    result = np.empty_like(images_uint8)
    for start in range(0, len(images_uint8), PALETTE_CHUNK):
        chunk = images_uint8[start:start + PALETTE_CHUNK].astype(np.float32) / PIXEL_HALF_RANGE - 1.0
        ids = (2 * chunk.reshape(-1, 3) @ palette.T - palette_sq).argmax(-1)
        rgb = np.rint((palette[ids] + 1.0) * PIXEL_HALF_RANGE).clip(0, UINT8_MAX)
        result[start:start + PALETTE_CHUNK] = rgb.reshape(chunk.shape).astype(np.uint8)
    return result


def tokenizer_roundtrip(images_uint8, tokenizer):
    """Потолок спектральной модели: encode -> decode тем же токенизатором."""
    _, height, _ = tokenizer.image_shape
    images = to_nchw_float(downsample_uint8(images_uint8, height))
    decoded = np.concatenate([
        tokenizer.decode(tokenizer.encode(images[start:start + TOKENIZER_CHUNK]))
        for start in range(0, len(images), TOKENIZER_CHUNK)
    ])
    return to_nhwc_uint8(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', required=True, help='data/celeba_common/celeba_test_64px.npy')
    parser.add_argument('--sample', action='append', default=[], help='имя=путь, можно несколько раз')
    parser.add_argument('--size', type=int, default=32, help='общее разрешение сравнения')
    parser.add_argument('--num-images', type=int, default=5000)
    parser.add_argument('--encoders', nargs='+', default=['inception'])
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--palette', help='kmeans_centers.npy -> потолок iGPT')
    parser.add_argument('--tokenizer', help='tokenizer.npz из runs/... -> потолок спектральной модели')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default='runs/compare')
    args = parser.parse_args()

    real_all = np.load(args.real, mmap_mode='r')
    if len(real_all) < args.num_images:
        raise ValueError('В референсе меньше картинок, чем --num-images')
    real = np.asarray(real_all[:args.num_images])
    # потолки считаем на других test-картинках, если их хватает
    spare = real_all[args.num_images:2 * args.num_images]
    ceiling_source = np.asarray(spare) if len(spare) == args.num_images else real

    candidates = {}
    for item in args.sample:
        name, path = item.split('=', 1)
        candidates[name] = (load_samples(path, args.num_images), Path(path).with_suffix(TIMING_SUFFIX))
    if args.palette:
        candidates['ceiling_palette'] = (palette_roundtrip(ceiling_source, np.load(args.palette).astype(np.float32)), None)
    if args.tokenizer:
        candidates['ceiling_spectral'] = (tokenizer_roundtrip(ceiling_source, Tokenizer.load(args.tokenizer)), None)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_eval = to_nchw_float(downsample_uint8(real, args.size))
    save_grid(real_eval, out_dir / 'real.png')
    results = {name: {} for name in candidates}

    for encoder_name in args.encoders:
        encoder = ENCODERS[encoder_name](args.device)
        real_feats, real_probs = scores.get_features(real_eval, encoder, args.batch_size)
        results.setdefault('real', {})[encoder_name] = {'is': scores.compute_is(real_probs)}
        for name, (images, _) in candidates.items():
            fake_eval = to_nchw_float(downsample_uint8(images, args.size))
            fake_feats, fake_probs = scores.get_features(fake_eval, encoder, args.batch_size)
            precision, recall = scores.compute_precision_recall(real_feats, fake_feats, args.k)
            results[name][encoder_name] = {
                'fid': scores.compute_fid(real_feats, fake_feats),
                'is': scores.compute_is(fake_probs),
                'precision': precision,
                'recall': recall,
            }
            print(name, encoder_name, results[name][encoder_name])
        del encoder

    for name, (images, timing_path) in candidates.items():
        save_grid(to_nchw_float(downsample_uint8(images, args.size)), out_dir / f'{name}.png')
        if timing_path is not None and timing_path.exists():
            results[name]['timing'] = json.loads(timing_path.read_text())

    (out_dir / 'results.json').write_text(json.dumps(results, indent=2))
    print('Saved', out_dir / 'results.json')


if __name__ == '__main__':
    main()
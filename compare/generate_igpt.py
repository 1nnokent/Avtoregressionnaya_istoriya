import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from compare.common import open_output, save_timing
from compare.igpt_model import ImageGPTSmall

N_LAYER = 24
N_HEAD = 8
N_EMBD = 512
N_VOCAB = 512
IMG_SIZE = 64
SEQ_LEN = IMG_SIZE * IMG_SIZE
RGB_CHANNELS = 3
QUERY_CHUNK = 1024
PIXEL_HALF_RANGE = 127.5
UINT8_MAX = 255
PALETTE_URL = 'https://openaipublic.blob.core.windows.net/image-gpt/color-clusters/kmeans_centers.npy'
DEFAULT_PALETTE = 'data/kmeans_centers.npy'
DEFAULT_NUM_IMAGES = 5000
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 128


def load_palette(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PALETTE_URL, path)
    return np.load(path).astype(np.float32)


def tokens_to_rgb(tokens, palette):
    rgb = np.rint((palette[tokens] + 1.0) * PIXEL_HALF_RANGE).clip(0, UINT8_MAX).astype(np.uint8)
    return rgb.reshape(len(tokens), IMG_SIZE, IMG_SIZE, RGB_CHANNELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--palette', default=DEFAULT_PALETTE)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = ImageGPTSmall(n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD, n_vocab=N_VOCAB,
                          seq_len=SEQ_LEN, query_chunk=QUERY_CHUNK).cuda()
    model.load_state_dict(state['model'])          # strict: чужие веса упадут здесь
    model.eval()
    print('Loaded', checkpoint, 'step', state.get('step'))

    palette = load_palette(args.palette)
    out, progress, done = open_output(args.out, args.num_images, IMG_SIZE)
    torch.manual_seed(args.seed + done)
    generated_now, seconds = 0, 0.0
    while done < args.num_images:
        count = min(args.batch_size, args.num_images - done)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            tokens = model.generate(n_samples=count, temperature=args.temperature, top_k=args.top_k)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        out[done:done + count] = tokens_to_rgb(tokens.cpu().numpy(), palette)
        out.flush()
        done += count
        generated_now += count
        progress.write_text(str(done))
        print(f'{done}/{args.num_images}')

    if generated_now:
        save_timing(args.out, model='igpt', nfe=SEQ_LEN, sec_per_image=seconds / generated_now,
                    temperature=args.temperature, top_k=args.top_k)


if __name__ == '__main__':
    main()

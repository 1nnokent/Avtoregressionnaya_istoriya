"""iGPT: N картинок пачками в .npy uint8 [N, 64, 64, 3], с докачкой.

Генерация идёт своим циклом с заранее выделенным KV-кешем (веса и математика модели
те же, что в ImageGPTSmall.generate). Оригинальный generate склеивает кеш через torch.cat,
память на каждом шаге удваивается, и на T4 пачка из 32 картинок упирается в OOM и тормозит.
Перед генерацией автоматически сверяем новый цикл с обычным forward модели.

Запуск из корня проекта:
    python -m compare.generate_igpt --checkpoint content/igpt_celeba64_checkpoint_5000.pt --out samples/igpt.npy
"""
import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

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
DEFAULT_BATCH_SIZE = 48         # кеш fp16 для 48 картинок ≈ 9 GiB, влезает в T4 (15 GiB)
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 128
CACHE_DTYPE = torch.float16

SELF_CHECK_SAMPLES = 2
SELF_CHECK_LENGTH = 64
SELF_CHECK_TOLERANCE = 0.1      # допустимое расхождение логитов в fp16


def load_palette(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PALETTE_URL, path)
    return np.load(path).astype(np.float32)


def tokens_to_rgb(tokens, palette):
    rgb = np.rint((palette[tokens] + 1.0) * PIXEL_HALF_RANGE).clip(0, UINT8_MAX).astype(np.uint8)
    return rgb.reshape(len(tokens), IMG_SIZE, IMG_SIZE, RGB_CHANNELS)


class PreallocatedCache:
    """K/V всех слоёв для всех позиций: [layers, batch, heads, seq, head_dim], выделяется один раз."""

    def __init__(self, model, batch_size, length=SEQ_LEN):
        attn = model.blocks[0].attn
        shape = (len(model.blocks), batch_size, attn.n_head, length, attn.head_dim)
        self.keys = torch.empty(shape, device=model.wpe.device, dtype=CACHE_DTYPE)
        self.values = torch.empty_like(self.keys)


def decode_step(model, cache, position, prev_tokens, batch_size):
    """Один шаг: вход на позиции position (SOS или предыдущий токен) -> логиты следующего токена."""
    if position == 0:
        x = model.sos.view(1, 1, -1).expand(batch_size, 1, -1)
    else:
        x = model.wte(prev_tokens[:, None])
    x = x + model.wpe[position:position + 1][None]

    for layer, block in enumerate(model.blocks):
        attn = block.attn
        qkv = attn.qkv(block.norm1(x)).reshape(batch_size, 1, 3, attn.n_head, attn.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)        # [batch, heads, 1, head_dim]
        cache.keys[layer, :, :, position:position + 1] = k
        cache.values[layer, :, :, position:position + 1] = v
        y = F.scaled_dot_product_attention(
            q.to(CACHE_DTYPE),
            cache.keys[layer, :, :, :position + 1],
            cache.values[layer, :, :, :position + 1],
        )                                                     # один новый токен видит всё прошлое
        x = x + attn.proj(y.transpose(1, 2).reshape(batch_size, 1, -1))
        x = x + block.mlp(block.norm2(x))

    return model.lm_head(model.norm_f(x))[:, -1, :].float()


@torch.inference_mode()
def generate(model, batch_size, temperature, top_k):
    cache = PreallocatedCache(model, batch_size)
    generated = torch.empty((batch_size, SEQ_LEN), dtype=torch.long, device=model.wpe.device)
    prev = None
    for position in tqdm(range(SEQ_LEN), desc='iGPT', leave=False):
        logits = decode_step(model, cache, position, prev, batch_size) / temperature
        if top_k is not None and top_k < logits.shape[-1]:
            threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        prev = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        generated[:, position] = prev
    return generated


@torch.inference_mode()
def self_check(model):
    """Подаём одни и те же токены в новый цикл и в обычный forward — логиты должны совпасть."""
    device = model.wpe.device
    tokens = torch.randint(0, N_VOCAB, (SELF_CHECK_SAMPLES, SELF_CHECK_LENGTH), device=device)
    expected = model(tokens).float()                          # логиты позиции i предсказывают token[i]
    cache = PreallocatedCache(model, SELF_CHECK_SAMPLES, SELF_CHECK_LENGTH)
    steps = []
    for position in range(SELF_CHECK_LENGTH):
        prev = tokens[:, position - 1] if position > 0 else None
        steps.append(decode_step(model, cache, position, prev, SELF_CHECK_SAMPLES))
    max_diff = (torch.stack(steps, dim=1) - expected).abs().max().item()
    print(f'Self-check: max |logits difference| = {max_diff:.4f}')
    if max_diff > SELF_CHECK_TOLERANCE:
        raise RuntimeError('Новый цикл генерации расходится с forward модели — генерацию не запускаю')


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

    with torch.autocast('cuda', dtype=torch.float16):
        self_check(model)

    palette = load_palette(args.palette)
    out, progress, done = open_output(args.out, args.num_images, IMG_SIZE)
    torch.manual_seed(args.seed + done)
    generated_now, seconds = 0, 0.0
    while done < args.num_images:
        count = min(args.batch_size, args.num_images - done)
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.float16):
            tokens = generate(model, count, args.temperature, args.top_k)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        out[done:done + count] = tokens_to_rgb(tokens.cpu().numpy(), palette)
        out.flush()
        done += count
        generated_now += count
        progress.write_text(str(done))
        print(f'{done}/{args.num_images}  ({seconds / generated_now:.2f} s/image)')

    if generated_now:
        save_timing(args.out, model='igpt', nfe=SEQ_LEN, sec_per_image=seconds / generated_now,
                    temperature=args.temperature, top_k=args.top_k)


if __name__ == '__main__':
    main()

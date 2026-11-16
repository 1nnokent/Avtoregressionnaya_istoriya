import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from compare.common import open_output, save_timing
from compare.flow_model import IGPTRGBDenoiser

N_LAYER = 24
N_HEAD = 8
N_EMBD = 512
MLP_RATIO = 4
IMG_SIZE = 64
SEQ_LEN = IMG_SIZE * IMG_SIZE
RGB_CHANNELS = 3
QUERY_CHUNK = 1024
TIME_SCALE = 1000            # модель получала время как 1000 * t (flow) или шаг 0..999 (DDPM)
BETA_START = 1e-4
BETA_END = 0.02
PIXEL_HALF_RANGE = 127.5
CHECK_IMAGES = 32
DEFAULT_NUM_IMAGES = 5000
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEPS = 100


def to_model_space(images_uint8):
    x = torch.from_numpy(np.asarray(images_uint8)).cuda().float()
    return x.div(PIXEL_HALF_RANGE).sub(1.0).reshape(len(x), SEQ_LEN, RGB_CHANNELS)


def to_uint8(x):
    x = x.reshape(len(x), IMG_SIZE, IMG_SIZE, RGB_CHANNELS).clamp(-1, 1)
    return x.add(1.0).mul(PIXEL_HALF_RANGE).round().to(torch.uint8).cpu().numpy()


def ddpm_alpha_bars(device):
    betas = torch.linspace(BETA_START, BETA_END, TIME_SCALE, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


def predict(model, x, t, use_fp16):
    with torch.autocast('cuda', dtype=torch.float16, enabled=use_fp16):
        return model(x, t).float()


@torch.inference_mode()
def check_objective(model, real_path):
    x0 = to_model_space(np.load(real_path, mmap_mode='r')[:CHECK_IMAGES])
    gen = torch.Generator(device='cuda').manual_seed(0)
    noise = torch.randn(x0.shape, device='cuda', generator=gen)

    t = torch.rand(CHECK_IMAGES, device='cuda', generator=gen)
    xt = (1 - t)[:, None, None] * noise + t[:, None, None] * x0
    flow_mse = F.mse_loss(model(xt, TIME_SCALE * t), x0 - noise).item()

    steps = torch.randint(0, TIME_SCALE, (CHECK_IMAGES,), device='cuda', generator=gen)
    a = ddpm_alpha_bars('cuda')[steps][:, None, None]
    xk = a.sqrt() * x0 + (1 - a).sqrt() * noise
    ddpm_mse = F.mse_loss(model(xk, steps), noise).item()
    print(f'flow MSE = {flow_mse:.3f} | DDPM MSE = {ddpm_mse:.3f}  '
          '(верный вариант: меньше 1 и заметно меньше другого)')


@torch.inference_mode()
def sample_flow(model, noise, num_steps, use_fp16):
    """Эйлер по t от 0 (шум) до 1 (картинка): x_t = (1-t)*noise + t*x0, v = x0 - noise."""
    x = noise
    times = torch.linspace(0.0, 1.0, num_steps + 1, device=x.device)
    for t_now, t_next in zip(times[:-1], times[1:]):
        velocity = predict(model, x, (TIME_SCALE * t_now).expand(len(x)), use_fp16)
        x = x + (t_next - t_now) * velocity
    return x


@torch.inference_mode()
def sample_ddim(model, noise, num_steps, use_fp16):
    """Детерминированный DDIM (eta=0) для модели, предсказывающей шум."""
    alpha_bars = ddpm_alpha_bars(noise.device)
    steps = torch.linspace(TIME_SCALE - 1, 0, num_steps, device=noise.device).round().long()
    x = noise
    for index, step in enumerate(steps):
        a = alpha_bars[step]
        a_prev = alpha_bars[steps[index + 1]] if index + 1 < len(steps) else torch.ones((), device=x.device)
        eps = predict(model, x, step.expand(len(x)), use_fp16)
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
    return x


SAMPLERS = {'flow': sample_flow, 'ddpm': sample_ddim}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--objective', choices=list(SAMPLERS))
    parser.add_argument('--check-objective', metavar='REAL_NPY',
                        help='только проверить, flow это или DDPM')
    parser.add_argument('--out')
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fp32', action='store_true')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = IGPTRGBDenoiser(N_LAYER, N_EMBD, N_HEAD, SEQ_LEN, MLP_RATIO, QUERY_CHUNK).cuda()
    model.load_state_dict(state['model'])
    model.eval()
    print('Loaded', checkpoint, 'step', state.get('completed_steps'))

    if args.check_objective:
        check_objective(model, args.check_objective)
        return
    if args.objective is None or args.out is None:
        parser.error('для генерации нужны --objective и --out')

    out, progress, done = open_output(args.out, args.num_images, IMG_SIZE)
    gen = torch.Generator(device='cuda').manual_seed(args.seed + done)
    sampler = SAMPLERS[args.objective]
    generated_now, seconds = 0, 0.0
    while done < args.num_images:
        count = min(args.batch_size, args.num_images - done)
        noise = torch.randn((count, SEQ_LEN, RGB_CHANNELS), device='cuda', generator=gen)
        start = time.perf_counter()
        x = sampler(model, noise, args.steps, not args.fp32)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        out[done:done + count] = to_uint8(x)
        out.flush()
        done += count
        generated_now += count
        progress.write_text(str(done))
        print(f'{done}/{args.num_images}')

    if generated_now:
        save_timing(args.out, model=args.objective, nfe=args.steps, sec_per_image=seconds / generated_now)


if __name__ == '__main__':
    main()

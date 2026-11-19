
import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from compare.common import PROGRESS_SUFFIX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = ('flow', 'igpt')
IGPT_PATTERN = 'igpt_celeba64_checkpoint_{step}*.pt'
FLOW_PATTERN = 'igpt_celeba64_ddim_checkpoint_{step}*.pt'
DEFAULT_CHECKPOINT_DIR = '/kaggle/input'
DEFAULT_STEPS = [1000, 2000, 3000, 5000]
DEFAULT_NUM_IMAGES = 1000
DEFAULT_FLOW_SAMPLER_STEPS = 25
CHECK_EVERY_SEC = 600


@dataclass
class Task:
    name: str
    out: Path
    argv: list
    log: Path = field(init=False)

    def __post_init__(self):
        self.log = self.out.with_suffix('.log')


def find_checkpoint(root, pattern):
    found = sorted(glob.glob(str(Path(root) / '**' / pattern), recursive=True))
    return found[0] if found else None


def progress_of(path):
    progress = Path(path).with_suffix(PROGRESS_SUFFIX)
    return int(progress.read_text()) if progress.exists() else None


def is_done(out, num_images):
    done = progress_of(out)
    return out.exists() and done is not None and done >= num_images


def copy_existing(spec, out_dir):
    name, source = spec.split('=', 1)
    source = Path(source)
    target = out_dir / f'{name}.npy'
    if target.exists():
        print('Уже есть:', target)
        return
    shutil.copy(source, target)
    done = progress_of(source)
    if done is None:
        done = len(np.load(source, mmap_mode='r'))
    target.with_suffix(PROGRESS_SUFFIX).write_text(str(done))
    print(f'Скопировано: {source} -> {target} ({done} картинок)')


def batch_args(batch_size):
    return [] if batch_size is None else ['--batch-size', str(batch_size)]


def build_tasks(args, out_dir):
    tasks = []
    for step in args.steps:
        if 'flow' in args.models:
            name = f'flow{args.flow_sampler_steps}_s{step}'
            out = out_dir / f'{name}.npy'
            checkpoint = find_checkpoint(args.checkpoint_dir, FLOW_PATTERN.format(step=step))
            if is_done(out, args.num_images):
                print('Уже готово:', name)
            elif checkpoint is None:
                print(f'Пропуск {name}: нет чекпоинта диффузии для шага {step}')
            else:
                tasks.append(Task(name, out, ['-m', 'compare.generate_diffusion', '--checkpoint', checkpoint,
                                              '--objective', 'flow', '--steps', str(args.flow_sampler_steps),
                                              '--out', str(out), '--num-images', str(args.num_images)]
                                  + batch_args(args.flow_batch_size)))
        if 'igpt' in args.models:
            name = f'igpt_s{step}'
            out = out_dir / f'{name}.npy'
            checkpoint = find_checkpoint(args.checkpoint_dir, IGPT_PATTERN.format(step=step))
            if is_done(out, args.num_images):
                print('Уже готово:', name)
            elif checkpoint is None:
                print(f'Пропуск {name}: нет чекпоинта iGPT для шага {step}')
            elif args.palette is None:
                raise SystemExit('Для iGPT нужен --palette (kmeans_centers.npy)')
            else:
                tasks.append(Task(name, out, ['-m', 'compare.generate_igpt', '--checkpoint', checkpoint,
                                              '--out', str(out), '--num-images', str(args.num_images),
                                              '--palette', args.palette]
                                  + batch_args(args.igpt_batch_size)))
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--checkpoint-dir', default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--steps', type=int, nargs='+', default=DEFAULT_STEPS)
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--flow-sampler-steps', type=int, default=DEFAULT_FLOW_SAMPLER_STEPS)
    parser.add_argument('--palette', help='kmeans_centers.npy для iGPT')
    parser.add_argument('--existing', action='append', default=[], metavar='ИМЯ=ПУТЬ',
                        help='готовые сэмплы, например igpt_s5000=.../igpt.npy')
    parser.add_argument('--igpt-batch-size', type=int, default=None, help='по умолчанию 48; для 6 GB — 16')
    parser.add_argument('--flow-batch-size', type=int, default=None, help='по умолчанию 32; для 6 GB — 8')
    parser.add_argument('--gpus', type=int, default=None, help='по умолчанию все видимые')
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.palette:
        args.palette = str(Path(args.palette).resolve())
    for spec in args.existing:
        copy_existing(spec, out_dir)

    tasks = build_tasks(args, out_dir)
    if not tasks:
        print('Все точки уже готовы')
        return
    if args.gpus is None:
        import torch
        args.gpus = max(1, torch.cuda.device_count())
    print(f'Задач: {len(tasks)}, GPU: {args.gpus}:', ', '.join(task.name for task in tasks), flush=True)

    pending = queue.Queue()
    for task in tasks:
        pending.put(task)
    running, failed = {}, []
    lock = threading.Lock()

    def worker(gpu):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            with lock:
                running[task.name] = task
            print(time.strftime('%H:%M'), f'GPU{gpu} старт:', task.name, flush=True)
            with open(task.log, 'w') as log:
                code = subprocess.run(
                    [sys.executable, *task.argv],
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
            with lock:
                running.pop(task.name)
                if code != 0:
                    failed.append(task.name)
            print(time.strftime('%H:%M'), f'GPU{gpu} {"готово" if code == 0 else "ОШИБКА"}:',
                  task.name, f'(лог: {task.log})', flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in range(args.gpus)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(CHECK_EVERY_SEC)
        with lock:
            status = [f'{name}: {progress_of(task.out) or 0}/{args.num_images}' for name, task in running.items()]
        print(time.strftime('%H:%M'), ' | '.join(status) or 'ожидание', flush=True)
    for thread in threads:
        thread.join()

    print('Ошибки:', failed if failed else 'нет')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()

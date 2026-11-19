import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compare.common import PROGRESS_SUFFIX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_NAME = re.compile(r'^(?P<model>.+)_s(?P<step>\d+)$')
SAMPLE_SUFFIXES = ('.npy', '.npz')
METRICS = ('fid', 'is', 'precision', 'recall')
LOWER_IS_BETTER = {'fid'}
CEILING_PREFIX = 'ceiling_'
REAL_NAME = 'real'
DEFAULT_ENCODER = 'inception'
DEFAULT_NUM_IMAGES = 1000
DEFAULT_SIZE = 32
FIGURE_COLUMN_INCHES = 4.5
FIGURE_ROW_INCHES = 3.5
PLOT_DPI = 120


def collect_samples(directories, num_images):
    samples = {}
    for directory in directories:
        for path in sorted(Path(directory).rglob('*')):
            match = SAMPLE_NAME.match(path.stem)
            if path.suffix not in SAMPLE_SUFFIXES or match is None or path.stem in samples:
                continue
            progress = path.with_suffix(PROGRESS_SUFFIX)
            if path.suffix == '.npy' and progress.exists() and int(progress.read_text()) < num_images:
                print(f'Пропуск {path}: готово {progress.read_text()} < {num_images}')
                continue
            samples[path.stem] = path
    return samples


def run_evaluate(args, samples):
    command = [sys.executable, '-m', 'compare.evaluate', '--real', args.real,
               '--num-images', str(args.num_images), '--size', str(args.size),
               '--out', args.out, '--encoders', *args.encoders]
    for name, path in samples.items():
        command += ['--sample', f'{name}={path}']
    if args.palette:
        command += ['--palette', args.palette]
    if args.tokenizer:
        command += ['--tokenizer', args.tokenizer]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def write_table(results, encoder, out_dir):
    rows = []
    for name, values in results.items():
        metrics = values.get(encoder, {})
        match = SAMPLE_NAME.match(name)
        model, step = (match['model'], int(match['step'])) if match else (name, None)
        rows.append((model, step, metrics))
    rows.sort(key=lambda row: (row[0], row[1] or 0))
    present = [metric for metric in METRICS if any(metric in row[2] for row in rows)]
    lines = ['| модель | шаг | ' + ' | '.join(present) + ' |',
             '|' + '---|' * (len(present) + 2)]
    for model, step, metrics in rows:
        cells = [f'{metrics[m]:.4g}' if m in metrics else '' for m in present]
        lines.append(f'| {model} | {step if step is not None else ""} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    (out_dir / f'points_{encoder}.md').write_text(table + '\n')
    print(table)
    return rows, present


def plot_curves(rows, present, encoder, out_dir):
    figure, axes = plt.subplots(1, len(present), squeeze=False,
                                figsize=(FIGURE_COLUMN_INCHES * len(present), FIGURE_ROW_INCHES))
    models = sorted({model for model, step, _ in rows if step is not None})
    for axis, metric in zip(axes[0], present):
        for model in models:
            points = sorted((step, m[metric]) for mod, step, m in rows
                            if mod == model and step is not None and metric in m)
            if points:
                axis.plot(*zip(*points), marker='o', label=model)
        for model, step, metrics in rows:
            if step is None and metric in metrics:      # потолки и IS настоящих картинок
                axis.axhline(metrics[metric], linestyle='--', linewidth=1, label=model)
        arrow = '↓' if metric in LOWER_IS_BETTER else '↑'
        axis.set(title=f'{metric.upper()} {arrow}', xlabel='шаг обучения')
        axis.grid(alpha=0.3)
    axes[0][0].legend(fontsize='small')
    figure.suptitle(f'Метрики по шагам обучения ({encoder})')
    figure.tight_layout()
    path = out_dir / f'curves_{encoder}.png'
    figure.savefig(path, dpi=PLOT_DPI)
    plt.close(figure)
    print('Сохранено:', path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-dir', nargs='+', required=True, help='где искать <модель>_s<шаг>.npy|npz')
    parser.add_argument('--real', required=True)
    parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE)
    parser.add_argument('--encoders', nargs='+', default=[DEFAULT_ENCODER])
    parser.add_argument('--palette')
    parser.add_argument('--tokenizer')
    parser.add_argument('--out', default='runs/compare_points')
    parser.add_argument('--skip-evaluate', action='store_true', help='только перерисовать по готовому results.json')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_evaluate:
        samples = collect_samples(args.samples_dir, args.num_images)
        if not samples:
            raise SystemExit('Не найдено ни одного файла вида <модель>_s<шаг>.npy|npz')
        print('Точки:', ', '.join(samples))
        run_evaluate(args, samples)

    results = json.loads((out_dir / 'results.json').read_text())
    for encoder in args.encoders:
        rows, present = write_table(results, encoder, out_dir)
        if present:
            plot_curves(rows, present, encoder, out_dir)


if __name__ == '__main__':
    main()

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_grid(images, path, columns=8):
    images = images[:64]
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows), squeeze=False)
    for axis, image in zip(axes.flat, images):
        axis.imshow(image[0], cmap="gray", vmin=0, vmax=1) if image.shape[0] == 1 else axis.imshow(image.transpose(1, 2, 0))
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_loss_plot(output_dir):
    rows = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    fig, axis = plt.subplots(figsize=(7, 4))
    for key, label in [("train/nll", "Train"), ("val/nll", "Validation"), ("val_initial/nll", "Initial validation")]:
        points = [(row["step"], row[key]) for row in rows if key in row]
        if points:
            x, y = zip(*points)
            axis.plot(x, y, "o-", label=label)
    axis.set(xlabel="Optimizer step", ylabel="NLL per token (nats)")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=140)
    plt.close(fig)

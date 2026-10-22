"""Adapt the datasets' NCHW images to the unchanged grayscale metrics API."""
import numpy as np

from metrics.src.generation import PIXEL_MAX, PIXEL_MIN, generate_images
from metrics.src.metrics import scores

GRID_SIDE = 8
GRID_CELL_INCHES = 1.0
GRID_DPI = 120
GRAY_CHANNEL = 0
NCHW_NDIM = 4
GRAY_CHANNELS = 1
GENERATED_FILE = "generated.npz"
SAMPLES_GRID_FILE = "samples.png"
REAL_GRID_FILE = "real.png"


def save_grid(images, path, side=GRID_SIDE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = side * GRID_CELL_INCHES
    fig, axes = plt.subplots(side, side, figsize=(size, size))
    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap="gray", vmin=PIXEL_MIN, vmax=PIXEL_MAX)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=GRID_DPI)
    plt.close(fig)


def evaluate_images(model, tokenizer, real_images, config, device, output_dir=None):
    if real_images.ndim != NCHW_NDIM or real_images.shape[1] != GRAY_CHANNELS:
        raise ValueError("Original metrics support grayscale images; set data.params.gray: true")
    from metrics.src.metrics.encoders import ENCODERS

    real_gray = real_images[:, GRAY_CHANNEL]
    generation = config["generation"]
    fake_images, clipped_frac = generate_images(
        model, lambda tokens: tokenizer.decode(tokens)[0, GRAY_CHANNEL],
        len(real_images), generation["temperature"], device,
        batch_size=generation["batch_size"], top_k=generation.get("top_k"),
        seed=config["experiment"]["seed"],
    )
    if output_dir is not None:
        np.savez_compressed(output_dir / GENERATED_FILE, images=fake_images)
        save_grid(fake_images, output_dir / SAMPLES_GRID_FILE)
        save_grid(real_gray, output_dir / REAL_GRID_FILE)

    settings = config["metrics"]
    names = set(settings["names"])
    result = {"clipped_frac": clipped_frac}
    for name in settings["encoders"]:
        encoder = ENCODERS[name](device)
        real_feats, real_probs = scores.get_features(real_gray, encoder, settings["batch_size"])
        fake_feats, fake_probs = scores.get_features(fake_images, encoder, settings["batch_size"])
        del encoder

        values = {}
        if "fid" in names:
            values["fid"] = scores.compute_fid(real_feats, fake_feats)
        if "is" in names:
            values["is_fake"] = scores.compute_is(fake_probs)
            values["is_real"] = scores.compute_is(real_probs)
        if "precision" in names:
            values["precision"] = scores.fraction_inside(
                fake_feats, real_feats, scores.knn_radii(real_feats, settings["k"]))
        if "recall" in names:
            values["recall"] = scores.fraction_inside(
                real_feats, fake_feats, scores.knn_radii(fake_feats, settings["k"]))
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError(f"Non-finite metrics from {name}: {values}")
        result[name] = values
    return result
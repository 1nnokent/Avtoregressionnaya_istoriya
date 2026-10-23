"""Adapt the datasets' NCHW images to the unchanged grayscale metrics API."""
import numpy as np

from metrics.src.generation import PIXEL_MAX, PIXEL_MIN, generate_images
from metrics.src.metrics import scores

GRID_SIDE = 8
GRID_CELL_INCHES = 1.0
GRID_DPI = 120
NCHW_NDIM = 4
GENERATED_FILE = "generated.npz"
SAMPLES_GRID_FILE = "samples.png"
REAL_GRID_FILE = "real.png"
CLIP_PROMPT_TEMPLATE = "a photo of a {}"


def save_grid(images, path, side=GRID_SIDE):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = side * GRID_CELL_INCHES
    fig, axes = plt.subplots(side, side, figsize=(size, size))
    for ax, image in zip(axes.flat, images):  # image: (C, H, W)
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap="gray", vmin=PIXEL_MIN, vmax=PIXEL_MAX)
        else:
            ax.imshow(image.transpose(1, 2, 0).clip(PIXEL_MIN, PIXEL_MAX))
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=GRID_DPI)
    plt.close(fig)


def evaluate_images(model, tokenizer, real_images, config, device, output_dir=None):
    if real_images.ndim != NCHW_NDIM:
        raise ValueError("Expected images (N, C, H, W)")
    from metrics.src.metrics.encoders import ENCODERS

    generation = config["generation"]
    fake_images, clipped_frac = generate_images(
        model, lambda tokens: tokenizer.decode(tokens)[0],
        len(real_images), generation["temperature"], device,
        batch_size=generation["batch_size"], top_k=generation.get("top_k"),
        seed=config["experiment"]["seed"],
    )
    if output_dir is not None:
        np.savez_compressed(output_dir / GENERATED_FILE, images=fake_images)
        save_grid(fake_images, output_dir / SAMPLES_GRID_FILE)
        save_grid(real_images, output_dir / REAL_GRID_FILE)

    settings = config["metrics"]
    names = set(settings["names"])
    result = {"clipped_frac": clipped_frac}
    for name in settings["encoders"]:
        prompts = settings.get("clip_prompts")
        if name == "clip" and prompts:
            encoder = ENCODERS[name](device, class_prompts=[CLIP_PROMPT_TEMPLATE.format(c) for c in prompts])
        else:
            encoder = ENCODERS[name](device)
        real_feats, real_probs = scores.get_features(real_images, encoder, settings["batch_size"])
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
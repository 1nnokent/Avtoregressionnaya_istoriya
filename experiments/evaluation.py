"""Adapt the datasets' NCHW images to the unchanged grayscale metrics API."""
import numpy as np

from metrics.src.generation import generate_images
from metrics.src.metrics import scores


def evaluate_images(model, tokenizer, real_images, config, device):
    # The original encoders accept (N,H,W); keep their implementation unchanged.
    if real_images.ndim != 4 or real_images.shape[1] != 1:
        raise ValueError("Original metrics support grayscale images; set data.params.gray: true")
    from metrics.src.metrics.encoders import ENCODERS

    generation = config["generation"]
    fake_images = generate_images(
        model, lambda tokens: tokenizer.decode(tokens)[0, 0],
        len(real_images), generation["temperature"], device,
        batch_size=generation["batch_size"], top_k=generation.get("top_k"),
    )
    result = {}
    settings = config["metrics"]
    names = set(settings["names"])
    for name in settings["encoders"]:
        encoder = ENCODERS[name](device)
        real_feats, real_probs = scores.get_features(real_images[:, 0], encoder, settings["batch_size"])
        fake_feats, fake_probs = scores.get_features(fake_images, encoder, settings["batch_size"])
        values = {}
        if "fid" in names:
            values["fid"] = float(scores.compute_fid(real_feats, fake_feats))
        if "is" in names:
            values["is_fake"] = float(scores.compute_is(fake_probs))
            values["is_real"] = float(scores.compute_is(real_probs))
        if "precision" in names:
            values["precision"] = float(scores.fraction_inside(fake_feats, real_feats, scores.knn_radii(real_feats, settings["k"])))
        if "recall" in names:
            values["recall"] = float(scores.fraction_inside(real_feats, fake_feats, scores.knn_radii(fake_feats, settings["k"])))
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite metrics from {name}: {values}")
        result[name] = values
        del encoder
    return result

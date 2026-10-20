import torch

from src.metrics.encoders import ENCODERS, train_mnist_encoder
from src.metrics.scores import compute_fid, compute_is, fraction_inside, get_features, knn_radii


def evaluate_images(real_images, fake_images, splits, config, output_dir, device, seed, logger, fit_encoder):
    results = {}
    for name in config["encoders"]:
        if name == "mnist":
            path = output_dir / "mnist_encoder.pt"
            if fit_encoder:
                train_mnist_encoder(splits["train"], splits["val"], device, path,
                                    config["classifier_epochs"], seed, logger)
            encoder = ENCODERS[name](device, path)
        elif name == "clip":
            encoder = ENCODERS[name](device, config["clip_prompts"])
        else:
            encoder = ENCODERS[name](device)
        real, real_probs = get_features(real_images, encoder, config["batch_size"])
        fake, fake_probs = get_features(fake_images, encoder, config["batch_size"])
        scores = config["scores"]
        if "frechet" in scores:
            results[f"{name}/frechet"] = compute_fid(real, fake)
        if "classifier_score" in scores:
            results[f"{name}/classifier_score_fake"] = compute_is(fake_probs)
            results[f"{name}/classifier_score_real"] = compute_is(real_probs)
        if "precision" in scores:
            results[f"{name}/precision"] = fraction_inside(fake, real, knn_radii(real, config["knn_k"]))
        if "recall" in scores:
            results[f"{name}/recall"] = fraction_inside(real, fake, knn_radii(fake, config["knn_k"]))
        if name == "mnist":
            results["mnist/encoder_val_accuracy"] = encoder.val_accuracy
            results["mnist/encoder_test_accuracy"] = float((real_probs.argmax(1) == splits["test"][1][:len(real_probs)]).mean())
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results

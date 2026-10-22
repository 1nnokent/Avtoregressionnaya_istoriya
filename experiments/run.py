"""Configuration-driven glue for the three independently written components."""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import random
import subprocess
import sys
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from experiments.artifacts import RunLog, load_tokenizer, save_tokenizer
from experiments.config import ROOT, load_config, project_path, resolve, validate
from model.src.train import evaluate, load_checkpoint, train_model


def load_split(config, split, limit):
    params = dict(config["data"].get("params", {}))
    params["root"] = project_path(params.get("root", "data"))
    params.setdefault("seed", config["experiment"]["seed"])
    images = resolve(config["data"]["target"])(split=split, limit=limit, **params)
    if len(images) == 0:
        raise ValueError(f"Empty {split} split")
    return images


def token_loader(images, tokenizer, config, shuffle=False):
    tokens = torch.from_numpy(tokenizer.encode(images)).long()
    return DataLoader(
        TensorDataset(tokens), batch_size=config["data"]["batch_size"], shuffle=shuffle,
        num_workers=config["data"].get("num_workers", 0),
    )


def make_model(config, tokenizer):
    params = dict(config["model"].get("params", {}))
    for key, value in (("vocab_size", tokenizer.vocab_size), ("max_seq_len", tokenizer.sequence_length)):
        if key in params and params[key] != value:
            raise ValueError(f"model.params.{key} must match tokenizer ({value}) or be omitted")
        params[key] = value
    return resolve(config["model"]["target"])(resolve(config["model"]["config_target"])(**params))


@contextmanager
def preserved_rng(device):
    accelerator = {"cuda": torch.cuda, "mps": torch.mps}.get(torch.device(device).type)
    cpu_state = torch.get_rng_state()
    accelerator_state = accelerator.get_rng_state() if accelerator else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if accelerator:
            accelerator.set_rng_state(accelerator_state)


TRAIN_SAMPLES_DIR = "samples"


def make_metrics_hook(config, tokenizer, device, logger, directory):
    settings = config["metrics"]
    interval = settings.get("train_interval")
    if not settings["enabled"] or not interval:
        return None
    from experiments.evaluation import evaluate_images
    real_images = load_split(config, "val", settings["train_num_images"])

    def hook(model, step):
        if step % interval:
            return
        step_dir = directory / TRAIN_SAMPLES_DIR / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        with preserved_rng(device):
            result = evaluate_images(model, tokenizer, real_images, config, device, output_dir=step_dir)
        flat = {"clipped_frac": result.pop("clipped_frac")}
        for encoder, values in result.items():
            flat.update({f"{encoder}_{key}": value for key, value in values.items()})
        print(f"step={step} " + " ".join(f"{key}={value:.4f}" for key, value in flat.items()))
        logger.record("image_metrics", step=step, **flat)

    return hook


def run(config, mode, checkpoint=None):
    if mode == "evaluate" and not checkpoint:
        raise ValueError("Evaluation requires --checkpoint or experiment.checkpoint")
    checkpoint = project_path(checkpoint) if checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    seed = config["experiment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = config["experiment"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use device: cpu or auto")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = project_path(config["experiment"]["output_dir"]) / config["experiment"]["name"] / f"{stamp}-{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    metadata = {
        "mode": mode, "device": device, "seed": seed, "checkpoint": str(checkpoint) if checkpoint else None,
        "git_commit": revision.stdout.strip(), "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger = RunLog(directory, sys.stdout)
    try:
        with redirect_stdout(logger), redirect_stderr(logger):
            logger.record("started", mode=mode, device=device)
            print(f"Run: {directory}")
            if mode == "train" and checkpoint is None:
                train_images = load_split(config, "train", config["data"].get("train_limit"))
                tokenizer = resolve(config["tokenizer"]["target"])(config["tokenizer"].get("params", {})).fit(train_images)
            else:
                tokenizer = load_tokenizer(checkpoint.parent / "tokenizer.npz", config["tokenizer"]["target"])
                if mode == "train":
                    train_images = load_split(config, "train", config["data"].get("train_limit"))
            save_tokenizer(directory / "tokenizer.npz", tokenizer)
            model = make_model(config, tokenizer).to(device)
            val_images = load_split(config, "val", config["data"].get("val_limit"))
            val_loader = token_loader(val_images, tokenizer, config)
            if mode == "train":
                training = {**config["training"], "checkpoint_dir": str(directory), "resume_from": str(checkpoint) if checkpoint else None}
                on_step = make_metrics_hook(config, tokenizer, device, logger, directory)
                model = train_model(model, token_loader(train_images, tokenizer, config, shuffle=True), val_loader, training, device,
                                    on_step=on_step)
                checkpoint = directory / "latest.pt"
            else:
                load_checkpoint(checkpoint, model, device=device)
            summary = evaluate(model, val_loader, device, config["training"].get("max_val_batches"))
            if config["metrics"]["enabled"]:
                from experiments.evaluation import evaluate_images
                real_images = load_split(config, config["data"].get("evaluation_split", "test"), config["generation"]["num_images"])
                if len(real_images) < config["generation"]["num_images"]:
                    raise ValueError("Evaluation split is smaller than generation.num_images")
                summary["images"] = evaluate_images(model, tokenizer, real_images, config, device, output_dir=directory)
            (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
            logger.record("completed", checkpoint=str(checkpoint), **summary)
            print(json.dumps(summary, indent=2))
    except Exception as error:
        logger.record("failed", error=str(error))
        raise
    finally:
        logger.close()
    return directory


def main(default_mode="train"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "mnist.yaml"))
    parser.add_argument("--mode", choices=("train", "evaluate"), default=default_mode)
    parser.add_argument("--checkpoint", help="Existing checkpoint, used for evaluation or training resume")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print config without loading datasets or building a model")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        validate(config)
        checkpoint = args.checkpoint or config["experiment"].get("checkpoint")
        if args.mode == "evaluate" and not checkpoint:
            raise ValueError("Evaluation requires --checkpoint or experiment.checkpoint")
        if args.dry_run:
            print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            print("Configuration OK; no data loaded, no model training started.")
            return
        run(config, args.mode, checkpoint)
    except (ValueError, TypeError, KeyError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

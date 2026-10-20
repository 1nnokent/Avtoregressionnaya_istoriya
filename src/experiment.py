import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, build_model, load_config
from src.datasets_loader import load_splits
from src.evaluation import evaluate_images
from src.generation import generate_images
from src.logging_utils import ExperimentLogger
from src.tokenizer import ImageTokenizer
from src.train import evaluate, load_checkpoint, train_model


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_runtime(config):
    torch.set_num_threads(config["num_threads"])
    name = config["device"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install CUDA wheels or select cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "data/.cache/torch"))
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/.cache/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data/.cache/matplotlib"))
    return torch.device(name)


def fingerprint(splits):
    result = {}
    for name, (images, labels) in splits.items():
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(images).tobytes())
        digest.update(np.ascontiguousarray(labels).tobytes())
        result[name] = {"count": len(images), "shape": list(images.shape), "sha256": digest.hexdigest()}
    return result


def make_loaders(splits, tokenizer, config, seed):
    return {name: DataLoader(torch.from_numpy(tokenizer.encode(images)),
                             batch_size=config["training"]["batch_size"], shuffle=name == "train",
                             num_workers=config["runtime"]["num_workers"],
                             generator=torch.Generator().manual_seed(seed))
            for name, (images, _) in splits.items()}


def score_run(model, tokenizer, splits, loaders, config, output_dir, device, logger, fit_encoder):
    from src.artifacts import save_grid
    seed_everything(config["experiment"]["seed"] + 1)
    g = config["generation"]
    fake = generate_images(model, tokenizer.decode, g["num_samples"], g["temperature"], device,
                           g["batch_size"], g["top_k"])
    real = splits["test"][0][:g["num_samples"]]
    reconstructed = tokenizer.decode(tokenizer.encode(real))
    scores = evaluate_images(real, fake, splits, config["metrics"], output_dir, device,
                             config["experiment"]["seed"] + 2, logger, fit_encoder)
    scores["tokenizer/test_reconstruction_mse"] = float(np.mean((real - reconstructed) ** 2))
    scores["tokenizer/test_reconstruction_psnr"] = float(-10 * np.log10(max(scores["tokenizer/test_reconstruction_mse"], 1e-12)))
    scores.update({f"test/{k}": v for k, v in evaluate(model, loaders["test"], device).items()})
    np.savez_compressed(output_dir / "generated.npz", images=fake)
    save_grid(fake, output_dir / "samples.png")
    save_grid(real, output_dir / "real.png")
    save_grid(np.concatenate([real[:16], reconstructed[:16]]), output_dir / "reconstruction.png")
    return scores


def _check_eval_sizes(config, splits):
    n = config["generation"]["num_samples"]
    real_n = min(n, len(splits["test"][0]))
    if config["metrics"]["encoders"] and (min(n, real_n) < 2 or
            (set(config["metrics"]["scores"]) & {"precision", "recall"} and
             config["metrics"]["knn_k"] >= min(n, real_n))):
        raise ValueError("Evaluation needs at least 2 samples and knn_k < real/fake sample counts")


def run_experiment(config_path, overrides=(), run_id=None):
    config = load_config(config_path, overrides)
    device = setup_runtime(config["runtime"])
    seed = config["experiment"]["seed"]
    seed_everything(seed)
    name = config["experiment"]["name"]
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise ValueError("experiment.name may contain only letters, digits, _ and -")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6]
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"] / name / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    config["data"]["root"] = str((PROJECT_ROOT / config["data"]["root"]).resolve())
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    logger = ExperimentLogger(output_dir, config["logging"]["tensorboard"])
    started = time.perf_counter()
    metadata = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
                "packages": {p: importlib.metadata.version(p) for p in
                             ["torch", "torchvision", "numpy", "scipy", "PyWavelets", "PyYAML", "tensorboard"]}}
    try:
        metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        metadata["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
        print(f"RUN_DIR={output_dir}\nDEVICE={device}", flush=True)
        splits = load_splits(config["data"], seed)
        _check_eval_sizes(config, splits)
        metadata["data"] = fingerprint(splits)
        tokenizer = ImageTokenizer(config["tokenizer"]).fit(splits["train"][0])
        tokenizer.save(output_dir / "tokenizer.json")
        loaders = make_loaders(splits, tokenizer, config, seed)
        model = build_model(config["model"], tokenizer).to(device)
        metadata["parameters"] = sum(p.numel() for p in model.parameters())
        metadata["sequence_length"] = tokenizer.seq_len
        metadata["vocab_size"] = tokenizer.num_bins
        dump_json(output_dir / "metadata.json", metadata)
        training = train_model(model, loaders["train"], loaders["val"], config["training"], device, output_dir, logger)
        best = load_checkpoint(output_dir / "best.pt", model, device=device)
        scores = score_run(model, tokenizer, splits, loaders, config, output_dir, device, logger, True)
        logger.log(scores, training["steps"])
        elapsed = time.perf_counter() - started
        summary = {"training": training, "selected_checkpoint_step": best["step"],
                   "evaluation_samples": {"real": min(len(splits["test"][0]), config["generation"]["num_samples"]),
                                          "fake": config["generation"]["num_samples"]},
                   "metrics": scores, "elapsed_seconds": elapsed}
        dump_json(output_dir / "summary.json", summary)
        from src.artifacts import save_loss_plot
        save_loss_plot(output_dir)
        metadata.update(status="completed", elapsed_seconds=elapsed)
        print(json.dumps(summary, indent=2), flush=True)
        return output_dir
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        dump_json(output_dir / "metadata.json", metadata)
        logger.close()


def evaluate_run(run_dir, device_override=None):
    output_dir = Path(run_dir).resolve()
    config = load_config(output_dir / "config.yaml")
    if device_override:
        config["runtime"]["device"] = device_override
    device = setup_runtime(config["runtime"])
    seed = config["experiment"]["seed"]
    seed_everything(seed)
    tokenizer = ImageTokenizer.load(output_dir / "tokenizer.json")
    if tokenizer.config != config["tokenizer"]:
        raise ValueError("Saved tokenizer does not match the run config")
    splits = load_splits(config["data"], seed)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    if fingerprint(splits) != metadata["data"]:
        raise ValueError("Dataset contents differ from the saved run")
    loaders = make_loaders(splits, tokenizer, config, seed)
    model = build_model(config["model"], tokenizer).to(device)
    checkpoint = load_checkpoint(output_dir / "best.pt", model, device=device)
    logger = ExperimentLogger(output_dir, config["logging"]["tensorboard"])
    try:
        scores = score_run(model, tokenizer, splits, loaders, config, output_dir, device, logger, False)
        logger.log(scores, checkpoint["step"])
        dump_json(output_dir / "evaluation.json", scores)
        return scores
    finally:
        logger.close()

"""Small YAML loader with relative inheritance and dotted Python factories."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path, parents=()):
    path = Path(path).resolve()
    if path in parents:
        raise ValueError(f"Circular config inheritance: {path}")
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, (*parents, path)), config)
    return config


def resolve(target):
    module, name = target.split(":", 1)
    return getattr(import_module(module), name)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate(config):
    for key in ("experiment", "data", "tokenizer", "model", "training", "metrics", "generation"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing config section: {key}")
    name = config["experiment"]["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("experiment.name must contain only letters, digits, '_' or '-'")
    for section in ("data", "tokenizer", "model"):
        if not callable(resolve(config[section]["target"])):
            raise ValueError(f"{section}.target must be callable")
    resolve(config["model"]["config_target"])(**config["model"].get("params", {}))
    resolve(config["tokenizer"]["target"])(config["tokenizer"].get("params", {}))
    if config["data"]["batch_size"] < 1:
        raise ValueError("data.batch_size must be positive")
    if config["data"].get("evaluation_split", "test") not in ("val", "test"):
        raise ValueError("Evaluation must use val or test split")
    for key in ("train_limit", "val_limit"):
        value = config["data"].get(key)
        if value is not None and value < 1:
            raise ValueError(f"data.{key} must be positive or null")
    if not 0 < config["data"]["params"].get("val_fraction", 0.1) < 1:
        raise ValueError("data.params.val_fraction must be between zero and one")
    train = config["training"]
    for key in ("epochs", "eval_interval", "log_interval", "checkpoint_interval"):
        if train[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if train.get("max_steps") is not None and train["max_steps"] < 1:
        raise ValueError("training.max_steps must be positive or null")
    if train.get("max_val_batches") is not None and train["max_val_batches"] < 1:
        raise ValueError("training.max_val_batches must be positive or null")
    metrics = config["metrics"]

    unknown = set(metrics["names"]) - {"fid", "is", "precision", "recall"}
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    if not metrics["encoders"] or set(metrics["encoders"]) - {"inception", "clip"}:
        raise ValueError("metrics.encoders must select inception and/or clip")
    if metrics.get("train_interval") is not None:
        if metrics["train_interval"] < 1:
            raise ValueError("metrics.train_interval must be positive or null")
        if metrics["train_num_images"] <= metrics["k"]:
            raise ValueError("metrics.train_num_images must be greater than metrics.k")
    generation = config["generation"]
    if generation["num_images"] < 2 or generation["batch_size"] < 1 or metrics["batch_size"] < 1:
        raise ValueError("Need at least two evaluation images and positive batch sizes")
    if metrics["k"] < 1 or (set(metrics["names"]) & {"precision", "recall"} and generation["num_images"] <= metrics["k"]):
        raise ValueError("Precision/recall require num_images > k >= 1")
    if generation["temperature"] <= 0 or (generation.get("top_k") is not None and generation["top_k"] < 1):
        raise ValueError("temperature and top_k must be positive")

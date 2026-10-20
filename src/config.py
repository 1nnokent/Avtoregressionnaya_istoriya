"""YAML inheritance and validated dotted CLI overrides."""
import copy
import importlib
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def merge(base, update, path="", strict=True):
    if path == "model" and update.get("config_class", base.get("config_class")) != base.get("config_class"):
        base = {**base, "params": {}}
    result = copy.deepcopy(base)
    for key, value in update.items():
        current = f"{path}.{key}" if path else key
        if strict and key not in base:
            raise ValueError(f"Unknown config key: {current}")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            flexible = current.endswith(("_params", "model.params"))
            result[key] = merge(base[key], value, current, strict and not flexible)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent = data.pop("extends", None)
    return merge(_read(path.parent / parent, seen), data) if parent else data


def load_config(path, overrides=()):
    defaults = _read(PROJECT_ROOT / "configs/base.yaml")
    config = merge(defaults, _read(path))
    for override in overrides:
        if "=" not in override:
            raise ValueError("Override must be dotted.key=value")
        key, raw = override.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError(f"Unknown config key: {key}")
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError(f"Unknown config key: {key}")
        cursor[parts[-1]] = yaml.safe_load(raw)
    validate(config)
    return config


def validate(c):
    def positive(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be positive")

    for section, keys in {
        "runtime": ["num_threads"], "data": ["size"],
        "training": ["batch_size", "epochs", "learning_rate", "grad_clip", "log_interval", "eval_interval"],
        "generation": ["num_samples", "batch_size", "temperature"],
        "metrics": ["batch_size", "knn_k", "classifier_epochs"],
    }.items():
        for key in keys:
            positive(c[section][key], f"{section}.{key}")
    integer_fields = {"runtime": ["num_threads", "num_workers"], "experiment": ["seed"],
                      "data": ["size", "train_limit", "val_limit", "test_limit"],
                      "training": ["batch_size", "epochs", "max_steps", "log_interval", "eval_interval"],
                      "generation": ["num_samples", "batch_size", "top_k"],
                      "metrics": ["batch_size", "knn_k", "classifier_epochs"], "tokenizer": ["max_tokens"]}
    for section, keys in integer_fields.items():
        for key in keys:
            value = c[section][key]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{section}.{key} must be an integer")
    if c["experiment"]["seed"] < 0 or c["training"]["weight_decay"] < 0:
        raise ValueError("seed and weight_decay must be nonnegative")
    if not 0 < c["data"]["val_fraction"] < 1:
        raise ValueError("data.val_fraction must be between 0 and 1")
    for key in ["train_limit", "val_limit", "test_limit"]:
        if c["data"][key] is not None:
            positive(c["data"][key], f"data.{key}")
    for section, key in [("training", "max_steps"), ("tokenizer", "max_tokens"), ("generation", "top_k")]:
        if c[section][key] is not None:
            positive(c[section][key], f"{section}.{key}")
    if c["runtime"]["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("runtime.device must be auto, cpu, cuda or mps")
    if c["runtime"]["num_workers"] < 0:
        raise ValueError("runtime.num_workers must be nonnegative")
    t = c["tokenizer"]
    if t["decomposition"] not in ("dct", "block_dct", "dft", "wavelet"):
        raise ValueError("Unknown decomposition")
    if t["ordering"] not in ("zigzag", "dyadic", "radial", "block"):
        raise ValueError("Unknown ordering")
    if t["normalization"]["transform"] not in ("none", "signed_log"):
        raise ValueError("normalization.transform must be none or signed_log")
    positive(t["normalization"]["log_scale"], "log_scale")
    if t["quantization"]["method"] not in ("uniform", "quantile"):
        raise ValueError("quantization.method must be uniform or quantile")
    if not isinstance(t["quantization"]["num_bins"], int) or t["quantization"]["num_bins"] < 2:
        raise ValueError("num_bins must be an integer >= 2")
    if not set(c["metrics"]["encoders"]) <= {"mnist", "pixels", "inception", "clip"}:
        raise ValueError("Unknown metrics encoder")
    if not set(c["metrics"]["scores"]) <= {"frechet", "classifier_score", "precision", "recall"}:
        raise ValueError("Unknown metric score")
    if "clip" in c["metrics"]["encoders"] and not c["metrics"]["clip_prompts"]:
        raise ValueError("Set metrics.clip_prompts for the CLIP encoder")
    if "pixels" in c["metrics"]["encoders"] and "classifier_score" in c["metrics"]["scores"]:
        raise ValueError("Pixel encoder has no class probabilities; remove classifier_score")
    if "mnist" in c["metrics"]["encoders"] and c["data"]["ds_name"] != "mnist":
        raise ValueError("MNIST encoder requires the mnist dataset")


def import_object(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def build_model(config, tokenizer):
    params = dict(config["params"])
    # These two dimensions are always derived from the fitted tokenizer.
    params.update(vocab_size=tokenizer.num_bins, max_seq_len=tokenizer.seq_len)
    model_config = import_object(config["config_class"])(**params)
    return import_object(config["class_path"])(model_config)

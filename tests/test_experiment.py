import json

import numpy as np
import pytest
import torch

from src.config import PROJECT_ROOT, build_model, load_config
from src.datasets_loader import load_images, load_splits, train_val_indices
from src.experiment import evaluate_run, fingerprint, run_experiment
from src.model import ImageTokenTransformer, ModelConfig
from src.tokenizer import ImageTokenizer


def test_split_is_disjoint_deterministic_and_complete():
    a, b = train_val_indices(100, seed=5)
    assert not set(a) & set(b)
    assert sorted([*a, *b]) == list(range(100))
    np.testing.assert_array_equal(a, train_val_indices(100, seed=5)[0])


def test_causal_attention_cannot_see_future():
    torch.manual_seed(0)
    model = ImageTokenTransformer(ModelConfig(vocab_size=16, d_model=16, n_heads=2, n_layers=1,
                                               d_ff=32, max_seq_len=8, dropout=0)).eval()
    a = torch.randint(0, 16, (2, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 16
    with torch.no_grad():
        torch.testing.assert_close(model(a)[:, :4], model(b)[:, :4], atol=1e-6, rtol=1e-6)


def test_offline_full_run_and_checkpoint_reevaluation(tmp_path):
    rng = np.random.default_rng(10)
    source = tmp_path / "images.npz"
    np.savez(source, train_images=rng.integers(0, 256, (40, 8, 8), dtype=np.uint8),
             test_images=rng.integers(0, 256, (12, 8, 8), dtype=np.uint8))
    overrides = [f"data.root={source.as_posix()}", "data.ds_name=npz", "data.size=8", "data.val_limit=4",
                 "data.test_limit=8", f"experiment.output_dir={tmp_path.as_posix()}", "logging.tensorboard=false"]
    output = run_experiment(PROJECT_ROOT / "configs/smoke.yaml", overrides, "offline")
    summary = json.loads((output / "summary.json").read_text())
    scores = evaluate_run(output, "cpu")
    assert scores == pytest.approx(summary["metrics"], abs=1e-7)
    assert summary["training"]["steps"] == 2
    assert (output / "samples.png").exists()
    assert (output / "tokenizer.json").exists()
    assert (output / "loss.png").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["data"]["test"]["count"] == 8
    assert load_images("npz", root=source, size=8, split="train", limit=2).shape == (2, 1, 8, 8)
    # Replacing data after training must be detected before reevaluation.
    np.savez(source, train_images=np.zeros((40, 8, 8), dtype=np.uint8),
             test_images=np.zeros((12, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="Dataset contents differ"):
        evaluate_run(output, "cpu")


def test_evaluate_old_quantile_run_without_training(tmp_path, monkeypatch):
    import yaml

    def forbid_training(*args, **kwargs):
        pytest.fail("Evaluation must not train a model or fit a tokenizer")

    rng = np.random.default_rng(21)
    source = tmp_path / "images.npz"
    np.savez(source, train_images=rng.integers(0, 256, (40, 8, 8), dtype=np.uint8),
             test_images=rng.integers(0, 256, (12, 8, 8), dtype=np.uint8))
    config = load_config(PROJECT_ROOT / "configs/smoke.yaml", [f"data.root={source.as_posix()}",
                         "data.ds_name=npz", "data.size=8", "logging.tensorboard=false"])
    splits = load_splits(config["data"], config["experiment"]["seed"])
    tokenizer = ImageTokenizer(config["tokenizer"]).fit(splits["train"][0])
    # Recreate a previously saved quantile state without fitting or training a neural model.
    values = tokenizer.core.normalizer(tokenizer.core.sequence(splits["train"][0]))
    edges = np.quantile(values, np.linspace(0, 1, tokenizer.num_bins + 1), axis=0)
    tokenizer.core.quantizer.thresholds = edges[1:-1].T
    tokenizer.core.quantizer.centers = ((edges[:-1] + edges[1:]) / 2).T
    tokenizer.config["quantization"]["method"] = "quantile"
    config["tokenizer"]["quantization"]["method"] = "quantile"
    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    tokenizer.save(run_dir / "tokenizer.json")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps({"data": fingerprint(splits)}), encoding="utf-8")
    model = build_model(config["model"], tokenizer)
    torch.save({"model": model.state_dict(), "model_config": vars(model.config), "step": 0}, run_dir / "best.pt")
    saved_state = (run_dir / "tokenizer.json").read_bytes()
    saved_model = (run_dir / "best.pt").read_bytes()
    monkeypatch.setattr("src.experiment.train_model", forbid_training)
    monkeypatch.setattr("src.evaluation.train_mnist_encoder", forbid_training)
    monkeypatch.setattr(ImageTokenizer, "fit", forbid_training)
    metrics = evaluate_run(run_dir, "cpu")
    assert np.isfinite(metrics["pixels/frechet"])
    assert (run_dir / "evaluation.json").exists()
    assert (run_dir / "tokenizer.json").read_bytes() == saved_state
    assert (run_dir / "best.pt").read_bytes() == saved_model

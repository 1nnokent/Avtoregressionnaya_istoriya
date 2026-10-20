import json

import numpy as np
import pytest
import torch

from src.config import PROJECT_ROOT
from src.datasets_loader import load_images, train_val_indices
from src.experiment import evaluate_run, run_experiment
from src.model import ImageTokenTransformer, ModelConfig


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

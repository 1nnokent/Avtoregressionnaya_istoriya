"""Exercise component boundaries with synthetic data; never call train_model."""
import json
import sys

import numpy as np
import pytest
import torch

from datasets.src.tokenizer import Tokenizer
from experiments.artifacts import RunLog, load_tokenizer, save_tokenizer
from experiments.config import ROOT, load_config, validate
from experiments import run as runner
from model.src.train import prepare_batch


def config():
    return load_config(ROOT / "configs/mnist.yaml")


def test_all_example_configs_validate():
    for path in (ROOT / "configs").glob("*.yaml"):
        validate(load_config(path))
    block = load_config(ROOT / "configs/mnist_block_dct.yaml")
    assert block["tokenizer"]["params"]["decomposition_params"] == {"block_size": 7}
    assert block["model"]["params"]["d_model"] == 128


def test_circular_inheritance_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml")
    (tmp_path / "b.yaml").write_text("extends: a.yaml")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


@pytest.mark.parametrize("per_position", [False, True])
def test_saved_tokenizer_preserves_training_state(tmp_path, per_position):
    images = np.random.default_rng(1).random((8, 1, 4, 4)).astype(np.float32)
    tokenizer = Tokenizer({"normalization": {"per_position": per_position}, "quantization": {"per_position": per_position}}).fit(images)
    save_tokenizer(tmp_path / "tokenizer.npz", tokenizer)
    restored = load_tokenizer(tmp_path / "tokenizer.npz", "datasets.src.tokenizer:Tokenizer")
    np.testing.assert_array_equal(tokenizer.encode(images), restored.encode(images))
    np.testing.assert_array_equal(tokenizer.decode(tokenizer.encode(images)), restored.decode(restored.encode(images)))


def test_tokens_connect_to_original_model_without_training():
    images = np.random.default_rng(2).random((2, 1, 4, 4)).astype(np.float32)
    tokenizer = Tokenizer().fit(images)
    cfg = config()
    cfg["model"]["params"] = {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0}
    model = runner.make_model(cfg, tokenizer).eval()
    inputs, targets = prepare_batch(next(iter(runner.token_loader(images, tokenizer, cfg))), model, "cpu")
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (2, 16, 256)
    assert targets.shape == (2, 16)
    assert inputs[:, 0].tolist() == [256, 256]
    cfg["model"]["params"]["vocab_size"] = 16
    with pytest.raises(ValueError, match="vocab_size"):
        runner.make_model(cfg, tokenizer)


def test_dry_run_cannot_load_data_or_train(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run tried to load data or train")
    monkeypatch.setattr(runner, "load_split", forbidden)
    monkeypatch.setattr(runner, "run", forbidden)
    monkeypatch.setattr(runner, "train_model", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--dry-run"])
    runner.main()
    assert "Configuration OK" in capsys.readouterr().out


def test_evaluate_uses_saved_tokenizer_and_never_trains(tmp_path, monkeypatch):
    images = np.random.default_rng(3).random((4, 1, 4, 4)).astype(np.float32)
    cfg = config()
    cfg["experiment"]["output_dir"] = str(tmp_path / "runs")
    cfg["experiment"]["device"] = "cpu"
    cfg["model"]["params"] = {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0}
    cfg["metrics"]["enabled"] = False
    tokenizer = Tokenizer().fit(images)
    model = runner.make_model(cfg, tokenizer)
    checkpoint = tmp_path / "latest.pt"
    torch.save({"model": model.state_dict(), "model_config": vars(model.config)}, checkpoint)
    save_tokenizer(tmp_path / "tokenizer.npz", tokenizer)
    splits = []

    def synthetic_split(config, split, limit):
        splits.append(split)
        return images

    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation must not train or fit tokenizer")

    monkeypatch.setattr(runner, "load_split", synthetic_split)
    monkeypatch.setattr(runner, "train_model", forbidden)
    monkeypatch.setattr(Tokenizer, "fit", forbidden)
    directory = runner.run(cfg, "evaluate", str(checkpoint))
    summary = json.loads((directory / "summary.json").read_text())
    assert splits == ["val"]
    assert summary["val_loss"] > 0
    assert (directory / "tokenizer.npz").is_file()
    assert not (directory / "latest.pt").exists()
    events = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["started", "completed"]


def test_training_prints_are_logged_without_changing_trainer(tmp_path):
    import io
    log = RunLog(tmp_path, io.StringIO())
    log.write("epoch=1 step=20 train_")
    log.write("loss=4.1234\nstep=200 val_loss=3.4567 val_perplexity=31.71\n")
    log.close()
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert events[0]["train_loss"] == 4.1234
    assert events[1]["val_perplexity"] == 31.71


def test_training_wiring_with_trainer_replaced_by_stub(tmp_path, monkeypatch):
    """Verify orchestration only: the real trainer is never called."""
    images = np.random.default_rng(5).random((4, 1, 4, 4)).astype(np.float32)
    cfg = config()
    cfg["experiment"].update(output_dir=str(tmp_path), device="cpu")
    cfg["model"]["params"] = {"d_model": 8, "n_heads": 2, "n_layers": 1, "d_ff": 16, "dropout": 0}
    cfg["metrics"]["enabled"] = False
    seen = []

    def synthetic_split(config, split, limit):
        seen.append(split)
        return images if split == "train" else images[:2]

    def trainer_stub(model, train_loader, val_loader, training, device, on_step=None):
        from pathlib import Path
        assert next(iter(train_loader))[0].shape == (4, 16)
        assert next(iter(val_loader))[0].shape == (2, 16)
        assert training["resume_from"] is None
        assert (Path(training["checkpoint_dir"]) / "tokenizer.npz").exists()
        # Simulate checkpoint output without backward, optimizer or weight updates.
        torch.save({"model": model.state_dict(), "model_config": vars(model.config)}, Path(training["checkpoint_dir"]) / "latest.pt")
        return model

    monkeypatch.setattr(runner, "load_split", synthetic_split)
    monkeypatch.setattr(runner, "train_model", trainer_stub)
    directory = runner.run(cfg, "train")
    assert seen == ["train", "val"]
    assert (directory / "summary.json").exists()


def test_metrics_receive_grayscale_images_without_weight_downloads(monkeypatch):
    from experiments import evaluation
    from metrics.src.metrics import encoders

    images = np.random.default_rng(4).random((5, 1, 4, 4)).astype(np.float32)
    tokenizer = Tokenizer().fit(images)
    cfg = config()
    cfg["generation"]["num_images"] = 5
    seen = []

    class SyntheticEncoder:
        def __init__(self, device):
            self.device = device

        def preprocess(self, batch):
            seen.append(batch.shape)
            assert batch.ndim == 4
            return torch.tensor(batch.reshape(len(batch), -1)[:, :3])

        def __call__(self, batch):
            return batch, torch.softmax(batch, dim=-1)

    monkeypatch.setitem(encoders.ENCODERS, "inception", SyntheticEncoder)
    monkeypatch.setattr(evaluation, "generate_images", lambda *args, **kwargs: (images[:, 0].copy(), 0.0))
    summary = evaluation.evaluate_images(None, tokenizer, images, cfg, "cpu")
    assert seen == [(5, 4, 4), (5, 4, 4)]
    assert abs(summary["inception"]["fid"]) < 1e-6
    assert summary["inception"]["precision"] == 1.0
    assert summary["inception"]["recall"] == 1.0

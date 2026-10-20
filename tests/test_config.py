import pytest

from src.config import PROJECT_ROOT, load_config


def test_inheritance_and_cli_overrides():
    config = load_config(PROJECT_ROOT / "configs/mnist_block_dct.yaml", ["training.max_steps=7"])
    assert config["tokenizer"]["decomposition"] == "block_dct"
    assert config["tokenizer"]["decomposition_params"]["block_size"] == 7
    assert config["training"]["max_steps"] == 7
    assert config["model"]["params"]["d_model"] == 64


@pytest.mark.parametrize("override", ["training.typo=3", "training.batch_size=0", "tokenizer.quantization.num_bins=1",
                                     "runtime.device=magic", "metrics.encoders=[pixels]", "tokenizer.max_tokens=0"])
def test_rejects_bad_config(override):
    with pytest.raises(ValueError):
        load_config(PROJECT_ROOT / "configs/base.yaml", [override])


def test_rejects_unknown_inherited_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(f"extends: {str(PROJECT_ROOT / 'configs/base.yaml')}\ntraining:\n  typo: 1\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(path)


def test_rejects_inheritance_cycle(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("extends: bad.yaml\n")
    with pytest.raises(ValueError, match="Cyclic"):
        load_config(path)


def test_new_model_config_replaces_architecture_parameters(tmp_path):
    path = tmp_path / "new_model.yaml"
    path.write_text(f"extends: {str(PROJECT_ROOT / 'configs/base.yaml')}\nmodel:\n"
                    "  config_class: custom.ModelConfig\n  class_path: custom.Model\n  params:\n    hidden_size: 16\n")
    config = load_config(path)
    assert config["model"]["params"] == {"hidden_size": 16}

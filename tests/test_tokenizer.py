import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.tokenizer import ImageTokenizer, Tokenizer


@pytest.mark.parametrize("decomposition,params", [("dct", {}), ("dft", {}),
                                                   ("block_dct", {"block_size": 4}), ("wavelet", {})])
@pytest.mark.parametrize("ordering", ["zigzag", "dyadic", "radial", "block"])
@pytest.mark.parametrize("channels", [1, 3])
def test_color_transform_order_roundtrip(decomposition, params, ordering, channels, tmp_path):
    c = load_config(PROJECT_ROOT / "configs/base.yaml")["tokenizer"]
    c.update(decomposition=decomposition, decomposition_params=params, ordering=ordering, max_tokens=None)
    c["quantization"]["num_bins"] = 256
    images = np.random.default_rng(42).random((16, channels, 8, 8)).astype(np.float32)
    tokenizer = ImageTokenizer(c).fit(images)
    tokens = tokenizer.encode(images)
    restored = tokenizer.decode(tokens)
    assert tokens.shape == (16, channels * 64)
    assert np.mean((restored - images) ** 2) < 1e-3
    tokenizer.save(tmp_path / "tokenizer.json")
    loaded = ImageTokenizer.load(tmp_path / "tokenizer.json")
    np.testing.assert_array_equal(loaded.encode(images), tokens)
    np.testing.assert_array_equal(loaded.decode(tokens), restored)


@pytest.mark.parametrize("method", ["uniform", "quantile"])
@pytest.mark.parametrize("per_position", [True, False])
def test_quantizers_constant_images_and_truncation(method, per_position, tmp_path):
    c = load_config(PROJECT_ROOT / "configs/base.yaml")["tokenizer"]
    c["max_tokens"] = 16
    c["quantization"].update(method=method, per_position=per_position)
    images = np.zeros((4, 1, 8, 8), dtype=np.float32)
    tokenizer = ImageTokenizer(c).fit(images)
    tokens = tokenizer.encode(images)
    assert np.isfinite(tokenizer.decode(tokens)).all()
    tokenizer.save(tmp_path / "tokens.json")
    np.testing.assert_array_equal(ImageTokenizer.load(tmp_path / "tokens.json").decode(tokens), tokenizer.decode(tokens))


def test_validation_does_not_change_fit_and_clips_outliers():
    c = load_config(PROJECT_ROOT / "configs/base.yaml")["tokenizer"]
    images = np.random.default_rng(1).random((8, 1, 8, 8))
    tokenizer = ImageTokenizer(c).fit(images)
    offset = tokenizer.core.normalizer.offset.copy()
    spread = tokenizer.core.normalizer.spread.copy()
    thresholds = tokenizer.core.quantizer.thresholds.copy()
    tokens = tokenizer.encode(images * 100)
    np.testing.assert_array_equal(tokenizer.core.normalizer.offset, offset)
    np.testing.assert_array_equal(tokenizer.core.normalizer.spread, spread)
    np.testing.assert_array_equal(tokenizer.core.quantizer.thresholds, thresholds)
    assert tokens.min() >= 0 and tokens.max() < tokenizer.num_bins


@pytest.mark.parametrize("scaling", ["minmax", "standard"])
@pytest.mark.parametrize("normalization_per_position", [True, False])
@pytest.mark.parametrize("quantization_per_position", [True, False])
@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_runner_uses_data_branch_tokenizer(scaling, normalization_per_position, quantization_per_position, method):
    config = load_config(PROJECT_ROOT / "configs/base.yaml")["tokenizer"]
    config["max_tokens"] = None
    config["normalization"].update(scaling=scaling, per_position=normalization_per_position)
    config["quantization"].update(method=method, per_position=quantization_per_position)
    images = np.random.default_rng(17).random((12, 3, 8, 8)).astype(np.float32)
    runner = ImageTokenizer(config).fit(images)
    original = Tokenizer({key: value for key, value in config.items() if key != "max_tokens"}).fit(images)
    np.testing.assert_array_equal(runner.encode(images), original.encode(images))
    expected = np.clip(original.decode(original.encode(images)), 0, 1).astype(np.float32)
    np.testing.assert_array_equal(runner.decode(runner.encode(images)), expected)


@pytest.mark.parametrize("per_position", [True, False])
@pytest.mark.parametrize("method", ["uniform", "quantile"])
def test_reads_old_tokenizer_files(method, per_position, tmp_path):
    import json
    from src.decomposition import get_decomposition
    from src.normalization import inverse_signed_log
    from src.ordering import invert_ordering, ordering_indices

    config = load_config(PROJECT_ROOT / "configs/base.yaml")["tokenizer"]
    config["normalization"].pop("scaling")
    config["normalization"].pop("per_position")
    config["max_tokens"] = 16
    config["quantization"].update(method=method, per_position=per_position, num_bins=8)
    low = np.linspace(-3, -1, 16) if per_position else np.asarray(-3.0)
    high = low + 5
    state = {"config": config, "shape": [1, 8, 8], "seq_len": 16, "low": low.tolist(), "high": high.tolist()}
    if method == "quantile":
        edges = np.linspace(low, high, 9)
        state["edges"] = edges.tolist()
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    tokenizer = ImageTokenizer.load(path)
    tokens = np.random.default_rng(4).integers(0, 8, (3, 16))
    if method == "quantile":
        centers = (edges[:-1] + edges[1:]) / 2
        sequence = centers[tokens, np.arange(16)] if per_position else centers[tokens]
    else:
        sequence = low + (tokens + 0.5) * (high - low) / 8
    full = np.zeros((3, 1, 64))
    full[:, 0, :16] = inverse_signed_log(sequence, config["normalization"]["log_scale"])
    coeffs = invert_ordering(full, ordering_indices("zigzag", 8, 8), 8, 8)
    expected = np.clip(get_decomposition("dct").inverse(coeffs), 0, 1).astype(np.float32)
    np.testing.assert_allclose(tokenizer.decode(tokens), expected, atol=1e-7)
    tokenizer.save(tmp_path / "current.json")
    np.testing.assert_array_equal(ImageTokenizer.load(tmp_path / "current.json").decode(tokens), tokenizer.decode(tokens))

import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.tokenizer import ImageTokenizer


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
    low, high = tokenizer.low.copy(), tokenizer.high.copy()
    tokens = tokenizer.encode(images * 100)
    np.testing.assert_array_equal(tokenizer.low, low)
    np.testing.assert_array_equal(tokenizer.high, high)
    assert tokens.min() >= 0 and tokens.max() < tokenizer.num_bins

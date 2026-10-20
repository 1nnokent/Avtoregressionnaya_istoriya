import numpy as np
import pytest

from datasets.src.decomposition import DECOMPOSITIONS
from datasets.src.ordering import ORDERINGS
from datasets.src.tokenizer import Tokenizer


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    images = np.zeros((40, 3, 16, 16), dtype=np.float32)
    images[:, :, 4:12, 5:11] = rng.random((40, 3, 8, 6))
    return images


def fine_config(**overrides):
    config = {"quantization": {"method": "uniform", "num_bins": 4096}}
    return {**config, **overrides}


@pytest.mark.parametrize("decomposition", DECOMPOSITIONS)
@pytest.mark.parametrize("ordering", ORDERINGS)
def test_encode_decode_every_decomposition_and_ordering(
    images, decomposition, ordering
):
    tokenizer = Tokenizer(
        fine_config(decomposition=decomposition, ordering=ordering)
    ).fit(images)
    tokens = tokenizer.encode(images)
    assert tokens.shape == (40, 3 * 16 * 16) and tokens.dtype == np.int64
    assert tokens.min() >= 0 and tokens.max() < tokenizer.vocab_size
    restored = tokenizer.decode(tokens)
    assert restored.shape == images.shape
    assert np.mean((restored - images) ** 2) < 1e-4


def test_sequence_roundtrip_is_exact(images):
    tokenizer = Tokenizer().fit(images)
    np.testing.assert_allclose(
        tokenizer.sequence_to_images(tokenizer.sequence(images)), images, atol=1e-5
    )


def test_gray_images_without_channel_axis(images):
    gray = images[:, 0]
    tokenizer = Tokenizer().fit(gray)
    assert tokenizer.encode(gray).shape == (40, 256)
    assert tokenizer.decode(tokenizer.encode(gray)).shape == (40, 1, 16, 16)


def test_wrong_image_shape_is_rejected(images):
    tokenizer = Tokenizer().fit(images)
    with pytest.raises(ValueError):
        tokenizer.encode(images[:, :1])


def test_block_dct_with_block_ordering(images):
    config = fine_config(
        decomposition="block_dct",
        decomposition_params={"block_size": 8},
        ordering="block",
        ordering_params={"block_size": 8},
    )
    tokenizer = Tokenizer(config).fit(images)
    assert np.mean((tokenizer.decode(tokenizer.encode(images)) - images) ** 2) < 1e-4


def test_partial_config_keeps_defaults():
    tokenizer = Tokenizer(
        {"normalization": {"scaling": "standard"}, "quantization": {"num_bins": 64}}
    )
    assert (
        tokenizer.normalizer.log_scale == 255.0
        and tokenizer.normalizer.scaling == "standard"
    )
    assert (
        tokenizer.quantizer.method == "uniform" and tokenizer.quantizer.num_bins == 64
    )

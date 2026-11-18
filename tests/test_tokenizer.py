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


def scalar_kmeans_config(num_bins=8, **overrides):
    quantization = {
        "method": "kmeans",
        "grouping": "scalar_channels",
        "num_bins": num_bins,
        "sample_size": 512,
        "iterations": 5,
        "seed": 13,
    }
    quantization.update(overrides)
    return {"quantization": quantization}


def joint_kmeans_config(num_bins=8, **overrides):
    quantization = {
        "method": "kmeans",
        "grouping": "joint_channels",
        "num_bins": num_bins,
        "sample_size": 512,
        "iterations": 5,
        "seed": 17,
    }
    quantization.update(overrides)
    return {
        "normalization": {"per_position": False},
        "quantization": quantization,
    }


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


def test_default_rgb_transform_preserves_existing_tokens(images):
    default = Tokenizer().fit(images)
    explicit_rgb = Tokenizer({"channel_transform": "rgb"}).fit(images)

    np.testing.assert_array_equal(explicit_rgb.encode(images), default.encode(images))


def test_default_grayscale_transform_preserves_existing_tokens(images):
    gray = images[:, :1]
    default = Tokenizer().fit(gray)
    explicit_grayscale = Tokenizer({"channel_transform": "grayscale"}).fit(gray)

    np.testing.assert_array_equal(
        explicit_grayscale.encode(gray), default.encode(gray)
    )


def test_unknown_state_format_is_rejected(images):
    state = Tokenizer().fit(images).state_dict()
    state["state_format"] = np.asarray(99, dtype=np.int64)

    with pytest.raises(ValueError, match="unsupported tokenizer state format 99"):
        Tokenizer().load_state_dict(state)


def test_scalar_kmeans_cifar_shape_uses_3072_tokens_and_256_ids():
    images = np.random.default_rng(10).random((4, 3, 32, 32)).astype(np.float32)
    tokenizer = Tokenizer(
        scalar_kmeans_config(
            num_bins=256, sample_size=768, iterations=2, seed=4
        )
    ).fit(images)

    tokens = tokenizer.encode(images)
    restored = tokenizer.decode(tokens)

    assert tokens.shape == (4, 3072)
    assert tokens.dtype == np.int64
    assert tokens.min() >= 0 and tokens.max() < 256
    assert tokenizer.sequence_length == 3072
    assert tokenizer.vocab_size == 256
    assert restored.shape == images.shape


def test_scalar_kmeans_validation_encoding_does_not_refit(images):
    tokenizer = Tokenizer(scalar_kmeans_config()).fit(images[:20])
    state_before = {
        key: value.copy() for key, value in tokenizer.state_dict().items()
    }

    tokenizer.encode(images[20:])

    state_after = tokenizer.state_dict()
    assert state_before.keys() == state_after.keys()
    for key, value in state_before.items():
        np.testing.assert_array_equal(state_after[key], value)


def test_scalar_kmeans_configuration_rejects_unknown_grouping():
    with pytest.raises(ValueError, match="unknown KMeans grouping"):
        Tokenizer(scalar_kmeans_config(grouping="spatial"))


def test_scalar_kmeans_integration_keeps_uniform_tokens_unchanged(images):
    default = Tokenizer().fit(images)
    explicit_uniform = Tokenizer({"quantization": {"method": "uniform"}}).fit(
        images
    )

    np.testing.assert_array_equal(
        explicit_uniform.encode(images), default.encode(images)
    )


def test_joint_kmeans_cifar_shape_uses_1024_tokens_and_decodes_rgb():
    images = np.random.default_rng(18).random((2, 3, 32, 32)).astype(np.float32)
    tokenizer = Tokenizer(
        joint_kmeans_config(num_bins=16, sample_size=1024, iterations=3)
    ).fit(images)

    tokens = tokenizer.encode(images)
    restored = tokenizer.decode(tokens)

    assert tokens.shape == (2, 1024)
    assert tokenizer.sequence_length == 1024
    assert tokenizer.vocab_size == 16
    assert restored.shape == (2, 3, 32, 32)


def test_joint_kmeans_rejects_channel_count_after_fitting():
    images = np.random.default_rng(19).random((8, 3, 8, 8)).astype(np.float32)
    tokenizer = Tokenizer(joint_kmeans_config()).fit(images)

    with pytest.raises(ValueError, match="fitted on images"):
        tokenizer.encode(images[:, :1])

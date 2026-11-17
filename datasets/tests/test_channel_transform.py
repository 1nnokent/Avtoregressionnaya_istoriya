import numpy as np
import pytest

from datasets.src.channel_transform import CHANNEL_TRANSFORMS, ChannelTransform


def test_available_channel_transforms_are_explicit():
    assert set(CHANNEL_TRANSFORMS) == {"grayscale", "ycbcr", "rgb"}


@pytest.mark.parametrize("name, channels", [("grayscale", 1), ("rgb", 3)])
def test_identity_transform_returns_independent_copy(name, channels):
    images = np.arange(2 * channels * 3 * 4, dtype=np.float32).reshape(
        2, channels, 3, 4
    )

    transformed = ChannelTransform(name).forward(images)

    np.testing.assert_array_equal(transformed, images)
    assert not np.shares_memory(transformed, images)


@pytest.mark.parametrize("name, channels", [("grayscale", 1), ("rgb", 3)])
def test_identity_transform_roundtrip_is_exact(name, channels):
    images = np.random.default_rng(0).random((2, channels, 4, 5))
    transform = ChannelTransform(name)

    restored = transform.inverse(transform.forward(images))

    np.testing.assert_array_equal(restored, images)


@pytest.mark.parametrize("method", ["forward", "inverse"])
@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 3, 4, 5, 1)])
def test_rgb_rejects_non_nchw_input(method, shape):
    transform = ChannelTransform("rgb")

    with pytest.raises(ValueError, match="NCHW"):
        getattr(transform, method)(np.zeros(shape))


@pytest.mark.parametrize(
    "name, channels, message",
    [("grayscale", 3, "1 channel"), ("rgb", 1, "3 channels")],
)
def test_identity_transform_rejects_wrong_channel_count(name, channels, message):
    with pytest.raises(ValueError, match=message):
        ChannelTransform(name).forward(np.zeros((2, channels, 4, 5)))


def test_unknown_channel_transform_is_rejected():
    with pytest.raises(ValueError, match="unknown channel transform"):
        ChannelTransform("lab")

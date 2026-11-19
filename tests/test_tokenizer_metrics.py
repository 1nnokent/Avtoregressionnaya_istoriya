import numpy as np
import pytest

from metrics import scores
from metrics.evaluation import evaluate_tokenizer


def test_reconstruction_metrics_report_mse_and_psnr():
    original = np.array([[[[0.0, 1.0]]]])
    reconstructed = np.zeros_like(original)

    result = scores.reconstruction_metrics(original, reconstructed)

    assert result["mse"] == pytest.approx(0.5)
    assert result["psnr"] == pytest.approx(10 * np.log10(2))


def test_reconstruction_metrics_report_infinite_psnr_for_exact_roundtrip():
    images = np.array([[[[0.25, 0.75]]]])

    result = scores.reconstruction_metrics(images, images.copy())

    assert result == {"mse": 0.0, "psnr": float("inf")}


@pytest.mark.parametrize(
    ("original", "reconstructed", "message"),
    [
        (np.zeros((1, 1, 2, 2)), np.zeros((1, 1, 2, 3)), "matching shapes"),
        (np.empty((0, 1, 2, 2)), np.empty((0, 1, 2, 2)), "non-empty"),
        (np.array([np.nan]), np.array([0.0]), "finite"),
    ],
)
def test_reconstruction_metrics_reject_invalid_images(original, reconstructed, message):
    with pytest.raises(ValueError, match=message):
        scores.reconstruction_metrics(original, reconstructed)


@pytest.mark.parametrize("data_range", [0.0, -1.0, np.inf])
def test_reconstruction_metrics_reject_invalid_data_range(data_range):
    with pytest.raises(ValueError, match="finite and positive"):
        scores.reconstruction_metrics(np.array([0.0]), np.array([0.0]), data_range)


def test_token_usage_metrics_describe_vocabulary_distribution():
    tokens = np.array([[0, 0, 1, 2], [0, 1, 2, 2]])

    result = scores.token_usage_metrics(tokens, vocab_size=4)

    assert result["sequence_length"] == 4
    assert result["vocab_size"] == 4
    assert result["token_count"] == 8
    assert result["used_tokens"] == 3
    assert result["unused_tokens"] == 1
    assert result["utilization"] == pytest.approx(0.75)
    assert result["entropy_bits"] == pytest.approx(1.561278124459133)
    assert result["effective_vocab_size"] == pytest.approx(2**1.561278124459133)
    assert result["frequencies"] == pytest.approx([3 / 8, 2 / 8, 3 / 8, 0.0])


@pytest.mark.parametrize(
    ("tokens", "vocab_size", "message"),
    [
        (np.array([0, 1]), 2, "two-dimensional"),
        (np.empty((0, 2), dtype=np.int64), 2, "non-empty"),
        (np.array([[0.0, 1.0]]), 2, "integer"),
        (np.array([[0, 2]]), 2, "outside"),
        (np.array([[0, 1]]), 0, "positive"),
    ],
)
def test_token_usage_metrics_reject_invalid_tokens(tokens, vocab_size, message):
    with pytest.raises(ValueError, match=message):
        scores.token_usage_metrics(tokens, vocab_size)


def test_token_usage_metrics_reject_non_integer_vocab_size():
    with pytest.raises(TypeError, match="integer"):
        scores.token_usage_metrics(np.array([[0, 1]]), 2.5)


def test_evaluate_tokenizer_reports_roundtrip_and_token_usage():
    images = np.array([[[[0.0, 1.0]]]])

    class SyntheticTokenizer:
        vocab_size = 4

        def encode(self, encoded_images):
            np.testing.assert_array_equal(encoded_images, images)
            return np.array([[0, 1]])

        def decode(self, tokens):
            np.testing.assert_array_equal(tokens, np.array([[0, 1]]))
            return np.zeros_like(images)

    result = evaluate_tokenizer(SyntheticTokenizer(), images)

    assert result["num_images"] == 1
    assert result["reconstruction"]["mse"] == pytest.approx(0.5)
    assert result["reconstruction"]["psnr"] == pytest.approx(10 * np.log10(2))
    assert result["token_usage"]["token_count"] == 2
    assert result["token_usage"]["frequencies"] == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_evaluate_tokenizer_serializes_exact_roundtrip_psnr_as_null():
    images = np.array([[[[0.25, 0.75]]]])

    class ExactTokenizer:
        vocab_size = 2

        def encode(self, encoded_images):
            return np.array([[0, 1]])

        def decode(self, tokens):
            return images.copy()

    result = evaluate_tokenizer(ExactTokenizer(), images)

    assert result["reconstruction"] == {"mse": 0.0, "psnr": None}

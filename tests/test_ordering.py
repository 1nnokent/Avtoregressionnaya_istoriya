import numpy as np

from src.ordering import inverse_zigzag, zigzag, zigzag_indices


def test_matches_jpeg_order():
    # beginning of the standard JPEG zigzag for an 8x8 block, as flat indices
    assert zigzag_indices(8, 8)[:10].tolist() == [0, 1, 8, 16, 9, 2, 3, 10, 17, 24]


def test_is_permutation():
    for height, width in [(28, 28), (4, 7), (7, 4), (1, 5)]:
        assert sorted(zigzag_indices(height, width).tolist()) == list(range(height * width))


def test_roundtrip_is_exact():
    matrix = np.arange(28 * 28).reshape(28, 28)
    assert np.array_equal(inverse_zigzag(zigzag(matrix), 28, 28), matrix)


def test_starts_at_low_frequencies():
    matrix = np.arange(9).reshape(3, 3)
    assert zigzag(matrix).tolist() == [0, 1, 3, 6, 4, 2, 5, 7, 8]

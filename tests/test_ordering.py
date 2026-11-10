import numpy as np
import pytest

from datasets.src.ordering import ORDERINGS, forward_ordering, invert_ordering, ordering_indices


def test_matches_jpeg_order():
    # beginning of the standard JPEG zigzag for an 8x8 block, as flat indices
    assert ordering_indices("zigzag", 8, 8)[:10].tolist() == [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
    ]


def test_is_permutation():
    for height, width in [(28, 28), (4, 7), (7, 4), (1, 5)]:
        indices = ordering_indices("zigzag", height, width)
        assert sorted(indices.tolist()) == list(range(height * width))


def test_roundtrip_is_exact():
    matrix = np.arange(28 * 28).reshape(28, 28)
    indices = ordering_indices("zigzag", 28, 28)
    sequence = forward_ordering(matrix, indices)
    assert np.array_equal(invert_ordering(sequence, indices, 28, 28), matrix)


def test_starts_at_low_frequencies():
    matrix = np.arange(9).reshape(3, 3)
    sequence = forward_ordering(matrix, ordering_indices("zigzag", 3, 3))
    assert sequence.tolist() == [0, 1, 3, 6, 4, 2, 5, 7, 8]


@pytest.mark.parametrize("name", ORDERINGS)
@pytest.mark.parametrize("shape", [(28, 28), (4, 7), (7, 4), (1, 5)])
def test_all_orderings_are_permutations_starting_at_corner(name, shape):
    indices = ordering_indices(name, *shape)
    assert sorted(indices.tolist()) == list(range(shape[0] * shape[1]))
    assert indices[0] == 0


@pytest.mark.parametrize("name", ORDERINGS)
def test_batched_roundtrip(name):
    coeffs = np.random.default_rng(0).random((2, 3, 8, 8))
    indices = ordering_indices(name, 8, 8)
    sequence = forward_ordering(coeffs, indices)
    assert sequence.shape == (2, 3, 64)
    assert np.array_equal(invert_ordering(sequence, indices, 8, 8), coeffs)


def test_dyadic_squares():
    rows, cols = np.divmod(ordering_indices("dyadic", 8, 8), 8)
    # the first 4 positions are the 2x2 corner, the first 16 are the 4x4 corner
    assert set(zip(rows[:4], cols[:4])) == {(i, j) for i in range(2) for j in range(2)}
    assert set(zip(rows[:16], cols[:16])) == {
        (i, j) for i in range(4) for j in range(4)
    }


def test_block_ordering():
    rows, cols = np.divmod(ordering_indices("block", 8, 8, block_size=4), 8)
    # block-major: the whole first 4x4 block comes first, in zigzag order
    assert np.all(rows[:16] < 4) and np.all(cols[:16] < 4)
    assert (rows * 8 + cols)[:6].tolist() == [0, 1, 8, 16, 9, 2]

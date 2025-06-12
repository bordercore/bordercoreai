"""
Pytest test suite for the `embeddings` module.
"""

import pytest
from modules.embeddings import batched


def test_batched_regular():
    """Batches of three from a seven-item iterable."""
    result = list(batched("ABCDEFG", 3))
    assert result == [
        ("A", "B", "C"),
        ("D", "E", "F"),
        ("G",),
    ]


def test_batched_exact_multiple():
    """Iterable length is an exact multiple of n."""
    result = list(batched([1, 2, 3, 4], 2))
    assert result == [(1, 2), (3, 4)]


def test_batched_n_greater_than_length():
    """n larger than iterable length should yield a single tuple."""
    result = list(batched([1, 2], 5))
    assert result == [(1, 2)]


def test_batched_n_equals_one():
    """n = 1 yields each element in its own one-item tuple."""
    result = list(batched([1, 2, 3], 1))
    assert result == [(1,), (2,), (3,)]


def test_batched_invalid_n():
    """n < 1 should raise ValueError."""
    with pytest.raises(ValueError):
        _ = list(batched([1, 2], 0))

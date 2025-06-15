"""
Pytest test suite for the `audio` module.
"""

import pytest
from modules.audio import Audio


@pytest.fixture()
def audio():
    """Provide a fresh ``Audio`` instance for each test case."""
    yield Audio()


def test_empty_list(audio):
    """fix_timestamps should return an empty list unchanged."""
    assert audio.fix_timestamps([]) == []


def test_single_timestamp(audio):
    """A single chunk should be returned verbatim."""
    input_data = [{"text": "Hello", "timestamp": (0.0, 1.5)}]
    expected = [{"text": "Hello", "timestamp": (0.0, 1.5)}]
    assert audio.fix_timestamps(input_data) == expected


def test_continuous_timestamps(audio):
    """Already-contiguous chunks must remain unmodified."""
    input_data = [
        {"text": "Hello", "timestamp": (0.0, 1.5)},
        {"text": "world", "timestamp": (1.5, 2.5)},
        {"text": "!", "timestamp": (2.5, 3.0)},
    ]
    expected = input_data.copy()
    assert audio.fix_timestamps(input_data) == expected


def test_discontinuous_timestamps(audio):
    """Non-monotonic chunks should be shifted to form a continuous sequence."""
    input_data = [
        {"text": "Hello", "timestamp": (0.0, 1.5)},
        {"text": "world", "timestamp": (0.0, 1.0)},   # overlaps previous
        {"text": "!", "timestamp": (0.0, 0.5)},       # overlaps previous
    ]
    expected = [
        {"text": "Hello", "timestamp": (0.0, 1.5)},
        {"text": "world", "timestamp": (1.5, 2.5)},
        {"text": "!", "timestamp": (2.5, 3.0)},
    ]
    assert audio.fix_timestamps(input_data) == expected


def test_mixed_timestamps(audio):
    """Mixed overlapping and gap segments should be packed and re-timed."""
    input_data = [
        {"text": "Hello", "timestamp": (0.0, 1.5)},
        {"text": "beautiful", "timestamp": (0.0, 1.0)},  # overlaps
        {"text": "world", "timestamp": (2.5, 3.0)},    # gap before this
        {"text": "!", "timestamp": (0.0, 0.5)},          # overlaps
    ]
    expected = [
        {"text": "Hello", "timestamp": (0.0, 1.5)},
        {"text": "beautiful", "timestamp": (1.5, 2.5)},
        {"text": "world", "timestamp": (4.0, 4.5)},
        {"text": "!", "timestamp": (4.5, 5.0)},
    ]
    assert audio.fix_timestamps(input_data) == expected


def test_rounding(audio):
    """Output timestamps should be rounded to one decimal place."""
    input_data = [
        {"text": "Hello", "timestamp": (0.0, 1.33)},
        {"text": "world", "timestamp": (0.0, 1.67)},
    ]
    expected = [
        {"text": "Hello", "timestamp": (0.0, 1.3)},
        {"text": "world", "timestamp": (1.3, 3.0)},
    ]
    assert audio.fix_timestamps(input_data) == expected

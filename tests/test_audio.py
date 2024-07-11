from ..audio import Audio


def test_empty_list():
    audio = Audio()
    assert audio.fix_timestamps([]) == []


def test_single_timestamp():
    audio = Audio()
    input = [{"text": "Hello", "timestamp": [0, 1.5]}]
    expected = [{"text": "Hello", "timestamp": [0, 1.5]}]
    assert audio.fix_timestamps(input) == expected


def test_continuous_timestamps():
    audio = Audio()
    input = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "world", "timestamp": [1.5, 2.5]},
        {"text": "!", "timestamp": [2.5, 3.0]}
    ]
    expected = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "world", "timestamp": [1.5, 2.5]},
        {"text": "!", "timestamp": [2.5, 3.0]}
    ]
    audio.fix_timestamps(input), expected


def test_discontinuous_timestamps():
    audio = Audio()
    input = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "world", "timestamp": [0, 1.0]},
        {"text": "!", "timestamp": [0, 0.5]}
    ]
    expected = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "world", "timestamp": [1.5, 2.5]},
        {"text": "!", "timestamp": [2.5, 3.0]}
    ]
    audio.fix_timestamps(input), expected


def test_mixed_timestamps():
    audio = Audio()
    input = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "beautiful", "timestamp": [0, 1.0]},
        {"text": "world", "timestamp": [2.5, 3.0]},
        {"text": "!", "timestamp": [0, 0.5]}
    ]
    expected = [
        {"text": "Hello", "timestamp": [0, 1.5]},
        {"text": "beautiful", "timestamp": [1.5, 2.5]},
        {"text": "world", "timestamp": [2.5, 3.0]},
        {"text": "!", "timestamp": [3.0, 3.5]}
    ]
    audio.fix_timestamps(input), expected


def test_rounding():
    audio = Audio()
    input = [
        {"text": "Hello", "timestamp": [0, 1.33]},
        {"text": "world", "timestamp": [0, 1.67]}
    ]
    expected = [
        {"text": "Hello", "timestamp": [0, 1.3]},
        {"text": "world", "timestamp": [1.3, 3.0]}
    ]
    audio.fix_timestamps(input), expected

from ..app import map_speech_rate_value


def test_map_speech_value():
    assert map_speech_rate_value(0) == 1.5
    assert map_speech_rate_value(2) == 0.5
    assert map_speech_rate_value(1) == 1.0
    assert map_speech_rate_value(1.1) == 0.95

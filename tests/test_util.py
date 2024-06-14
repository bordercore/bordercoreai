from ..util import sort_models


def test_sort_models():

    assert sort_models(
        [{"name": "b"}, {"name": "d"}, {"name": "e"}, {"name": "a"}],
        ["a", "d"]
    ) == [{"name": "a"}, {"name": "d"}, {"name": "b"}, {"name": "e"}]

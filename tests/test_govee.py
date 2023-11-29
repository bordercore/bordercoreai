from ..govee import balance_braces


def test_balance_braces():

    assert balance_braces("example_string_with{{}}}}}}}") == "example_string_with{{}}"
    assert balance_braces("example_string_with{}") == "example_string_with{}"

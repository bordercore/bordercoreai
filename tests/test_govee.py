"""
Unit tests for the `balance_braces` function in the `modules.govee` module.
"""

from modules.govee import balance_braces


def test_balance_braces():
    """
    Test that `balance_braces` removes extra closing braces and preserves valid ones.

    Verifies that unbalanced trailing braces are stripped, and properly balanced
    strings are left unchanged.
    """
    assert balance_braces("example_string_with{{}}}}}}}") == "example_string_with{{}}"
    assert balance_braces("example_string_with{}") == "example_string_with{}"

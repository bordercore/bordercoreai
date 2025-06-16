
"""
Unit tests for the FunctionCall class in `modules.function_calling`.
"""

import re

import pytest
from modules.function_calling import FunctionCall


@pytest.fixture()
def test_instance():
    """
    Fixture providing a FunctionCall instance with a mock model name.
    """
    yield FunctionCall("model_name")


def test_rename_existing_key(test_instance):
    """
    Test renaming a key that exists in the dictionary.
    """
    data = {"old_key": "value"}
    test_instance.rename_key(data, "old_key", "new_key")
    assert "new_key" in data
    assert "old_key" not in data
    assert data["new_key"] == "value"


def test_rename_nonexistent_key(test_instance):
    """
    Test attempting to rename a key that does not exist; data should be unchanged.
    """
    data = {"existing_key": "value"}
    test_instance.rename_key(data, "nonexistent_key", "new_key")
    assert data == {"existing_key": "value"}


def test_rename_to_existing_key(test_instance):
    """
    Test renaming a key to an existing key; target value should be overwritten.
    """
    data = {"old_key": "old_value", "new_key": "existing_value"}
    test_instance.rename_key(data, "old_key", "new_key")
    assert "old_key" not in data
    assert data["new_key"] == "old_value"


def test_rename_with_empty_dict(test_instance):
    """
    Test renaming a key in an empty dictionary; should remain empty.
    """
    data = {}
    test_instance.rename_key(data, "old_key", "new_key")
    assert data == {}


def test_rename_key_to_itself(test_instance):
    """
    Test renaming a key to itself; dictionary should remain unchanged.
    """
    data = {"key": "value"}
    test_instance.rename_key(data, "key", "key")
    assert data == {"key": "value"}


def test_default_length(test_instance):
    """
    Test that the default random ID length is 6.
    """
    random_id = test_instance.generate_random_id()
    assert len(random_id) == 6


def test_custom_length(test_instance):
    """
    Test that specifying a custom random ID length works.
    """
    random_id = test_instance.generate_random_id(length=10)
    assert len(random_id) == 10


def test_zero_length(test_instance):
    """
    Test generating a random ID of length 0 returns an empty string.
    """
    random_id = test_instance.generate_random_id(length=0)
    assert random_id == ""


def test_randomness(test_instance):
    """
    Test that two generated IDs are not the same (likely).
    """
    id1 = test_instance.generate_random_id()
    id2 = test_instance.generate_random_id()
    assert id1 != id2


def test_id_format(test_instance):
    """
    Test that the generated ID contains only alphanumeric characters.
    """
    random_id = test_instance.generate_random_id()
    assert re.match(r'^[a-zA-Z0-9]{6}$', random_id)

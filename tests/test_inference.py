"""
Unit tests for the `Inference` class configuration behavior.
"""

from unittest.mock import patch

import pytest
from modules.inference import Inference


@pytest.fixture
def config(monkeypatch):
    """
    Fixture that sets up a mock Inference instance with predefined model_info.

    This fixture patches the get_tokenizer method and the get_model_info function
    to simulate two fake models for testing configuration options.

    Returns:
        An Inference instance with mock configuration data.
    """
    model_info = {
        "model1": {
            "option1": "value1",
            "option2": "value2"
        },
        "model2": {
            "option1": "value3",
            "option3": "value4"
        }
    }

    def mocked_method(self):
        return None

    monkeypatch.setattr(Inference, "get_tokenizer", mocked_method)

    with patch("modules.inference.get_model_info", return_value=model_info):
        return Inference("/path/to/model1")


def test_get_config_option_existing_key(config):
    """
    Test that get_config_option returns the correct value for existing keys.

    Args:
        config: The mocked Inference instance fixture.
    """
    assert config.get_config_option("option1") == "value1"
    assert config.get_config_option("option2") == "value2"


def test_get_config_option_non_existing_key_with_default(config):
    """
    Test that get_config_option returns the provided default for missing keys.

    Args:
        config: The mocked Inference instance fixture.
    """
    assert config.get_config_option("option3", "default_value") == "default_value"
    assert config.get_config_option("non_existing_option", "default") == "default"


def test_get_config_option_non_existing_key_without_default(config):
    """
    Test that get_config_option returns None when no default is provided.

    Args:
        config: The mocked Inference instance fixture.
    """
    assert config.get_config_option("option3") is None
    assert config.get_config_option("non_existing_option") is None

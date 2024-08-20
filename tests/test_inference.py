from unittest.mock import patch

import pytest
from modules.inference import Inference


@pytest.fixture
def config():
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
    with patch("modules.inference.get_model_info", return_value=model_info), \
         patch("modules.inference.get_tokenizer", return_value=None):
        return Inference("/path/to/model1")


def test_get_config_option_existing_key(config):
    assert config.get_config_option("option1") == "value1"
    assert config.get_config_option("option2") == "value2"


def test_get_config_option_non_existing_key_with_default(config):
    assert config.get_config_option("option3", "default_value") == "default_value"
    assert config.get_config_option("non_existing_option", "default") == "default"


def test_get_config_option_non_existing_key_without_default(config):
    assert config.get_config_option("option3") is None
    assert config.get_config_option("non_existing_option") is None

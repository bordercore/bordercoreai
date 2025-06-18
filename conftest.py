import sys
from pathlib import Path

import pytest

from modules.chatbot import ChatBot


def pytest_configure(config):
    """Add the project’s root directory to sys.path so test imports resolve."""
    project_dir = str(Path(__file__).resolve().parent)
    sys.path.insert(0, project_dir)

@pytest.fixture
def chatbot():
    """Provide a ChatBot instance preset with a test model and empty args."""
    bot = ChatBot()
    bot.model_name = "test-model"
    bot.args = {}
    return bot

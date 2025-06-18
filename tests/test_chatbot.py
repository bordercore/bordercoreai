"""
Unit tests for the `sanitize_string` method of the ChatBot class.
"""
from unittest.mock import MagicMock, patch

import pytest
from modules.chatbot import ChatBot, Context


def test_sanitize_string():
    """Test that `sanitize_string` correctly strips trailing punctuation and whitespace."""
    chatbot = ChatBot()
    assert chatbot.sanitize_string("foobar.") == "foobar"
    assert chatbot.sanitize_string("foobar ") == "foobar"


def test_init_stt_if_enabled_enabled():
    """Returns WhisperMic instance when STT is enabled."""
    instance = ChatBot()
    instance.args = {"stt": True}
    with patch("modules.chatbot.WhisperMic") as mock_mic:
        mic = instance.init_stt_if_enabled()
        mock_mic.assert_called_once_with(model="small", energy=100)
        assert mic == mock_mic()


def test_init_stt_if_enabled_disabled():
    """Returns WhisperMic instance when STT is enabled."""
    instance = ChatBot()
    instance.args = {"stt": False}
    mic = instance.init_stt_if_enabled()
    assert mic is None


def test_get_user_input_stt_inactive_skips_on_wrong_wake_word():
    """Skips input when wake-word is incorrect and assistant mode is active."""
    instance = ChatBot()
    instance.args = {"stt": True, "assistant": True, "debug": False}
    instance.get_wake_word = MagicMock(return_value="hello")
    instance.sanitize_string = lambda s: s
    mic = MagicMock()
    mic.listen.return_value = "not hello"
    result = instance.get_user_input(mic, active=False)
    assert result is None


def test_get_user_input_keyboard_interrupt():
    """Exits the program when user sends a keyboard interrupt."""
    instance = ChatBot()
    instance.args = {"stt": False}
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit):
            instance.get_user_input(mic=None, active=False)


def test_handle_response_inference_enabled():
    """Uses inference engine to process and print assistant response."""
    instance = ChatBot()
    instance.args = {"tts": False}
    inference = MagicMock()
    inference.context.get.return_value = ["context"]
    inference.generate.return_value = ["Hello", " world!"]
    instance.send_message_to_model = MagicMock()
    instance.speak = MagicMock()
    instance.handle_response("Hi", inference)
    inference.context.add.assert_called_once_with("Hi", True)

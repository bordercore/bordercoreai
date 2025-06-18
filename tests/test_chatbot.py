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


def test_handle_message_lights(chatbot):
    """Ensure a lights request calls control_lights and returns its response."""
    chatbot.get_request_type = MagicMock(return_value={"category": "lights"})
    with patch("modules.chatbot.control_lights", return_value="light-response") as mock_control:
        messages = [{"role": "user", "content": "turn on the lights"}]
        result = chatbot.handle_message(messages)
        assert result == "light-response"
        mock_control.assert_called_once()


def test_handle_message_music(chatbot):
    """Ensure a music request calls play_music and returns its response."""
    chatbot.get_request_type = MagicMock(return_value={"category": "music"})
    with patch("modules.chatbot.play_music", return_value="music-response") as mock_play:
        messages = [{"role": "user", "content": "play music"}]
        result = chatbot.handle_message(messages)
        assert result == "music-response"
        mock_play.assert_called_once()


def test_handle_message_math_with_wolfram(chatbot):
    """Ensure math requests use Wolfram Alpha when wolfram_alpha is enabled."""
    chatbot.args["wolfram_alpha"] = True
    messages = [{"role": "user", "content": "what is 2+2"}]
    with patch("modules.chatbot.WolframAlphaFunctionCall") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.run.return_value = "4"
        result = chatbot.handle_message(messages)
        assert result == "4"
        mock_instance.run.assert_called_once_with("what is 2+2")


def test_handle_message_math_with_thinking_enabled(chatbot):
    """Ensure math requests fall back to the model when enable_thinking is True."""
    chatbot.args["enable_thinking"] = True
    chatbot.get_request_type = MagicMock(return_value={"category": "math"})
    messages = [{"role": "user", "content": "what is 2+2"}]
    chatbot.send_message_to_model = MagicMock(return_value="llm-response")
    result = chatbot.handle_message(messages)
    assert result == "llm-response"


def test_handle_message_default(chatbot):
    """Ensure unrecognized categories route to send_message_to_model."""
    chatbot.get_request_type = MagicMock(return_value={"category": "unknown"})
    messages = [{"role": "user", "content": "tell me a joke"}]
    chatbot.send_message_to_model = MagicMock(return_value="default-response")
    result = chatbot.handle_message(messages)
    assert result == "default-response"


def test_handle_message_with_url(chatbot):
    """Ensure URL content is appended and the message is routed to the model."""
    chatbot.args["url"] = "http://example.com"
    with patch("modules.chatbot.get_webpage_contents", return_value="Example site") as mock_get:
        chatbot.get_request_type = MagicMock(return_value={"category": "other"})
        messages = [{"role": "user", "content": "summarize"}]
        chatbot.send_message_to_model = MagicMock(return_value="summarized")
        result = chatbot.handle_message(messages)
        assert "Example site" in messages[-1]["content"]
        assert result == "summarized"

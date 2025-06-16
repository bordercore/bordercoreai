"""
Unit tests for the `sanitize_string` method of the ChatBot class.
"""

from modules.chatbot import ChatBot, Context


def test_sanitize_string():
    """Test that `sanitize_string` correctly strips trailing punctuation and whitespace."""
    chatbot = ChatBot()

    assert chatbot.sanitize_string("foobar.") == "foobar"
    assert chatbot.sanitize_string("foobar ") == "foobar"

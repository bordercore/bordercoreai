from modules.chatbot import ChatBot, Context


def test_sanitize_string():

    chatbot = ChatBot()

    assert chatbot.sanitize_string("foobar.") == "foobar"
    assert chatbot.sanitize_string("foobar ") == "foobar"

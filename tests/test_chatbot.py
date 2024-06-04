from ..chatbot import ChatBot, Context


def test_sanitize_string():

    context = Context()
    chatbot = ChatBot(context)

    assert chatbot.sanitize_string("foobar.") == "foobar"
    assert chatbot.sanitize_string("foobar ") == "foobar"

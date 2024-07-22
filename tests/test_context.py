import pytest
from modules.context import Context


@pytest.fixture()
def context():
    yield Context([{"role": "system", "content": "System message"}])


def test_context_initialization():
    context = Context()
    assert context.context == []
    assert context.context_limit == 4096


def test_add_single_message(context):
    context.add("Hello", prune=False)
    assert context.context == [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Hello"}
    ]


def test_add_list_of_messages(context):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    context.add(messages, prune=False)
    assert context.context == messages


def test_get_context(context):
    context.add([{"role": "user", "content": "Test"}], prune=False)
    assert context.get() == [{"role": "user", "content": "Test"}]


def test_size_calculation(context):
    context = Context([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ])
    assert context.size() == 4 + 5 + 9 + 9


def test_clear_context(context):
    context.add([{"role": "user", "content": "Test"}], prune=False)
    context.clear()
    assert context.context == []


def test_prune_context(context):
    context.context_limit = 30
    context.add("This is a long message", prune=True)
    context.add("Short", prune=True)
    assert len(context.context) == 2
    assert context.context[1]["content"] == "Short"


def test_prune_preserves_first_message(context):
    context.context_limit = 30
    context.add("This is a long message", prune=True)
    context.add("Another long message", prune=True)
    assert len(context.context) == 1
    assert context.context[0]["role"] == "system"


def test_add_without_pruning(context):
    context.context_limit = 10
    context.add("Long message 1", prune=False)
    context.add("Long message 2", prune=False)
    assert len(context.context) == 3


def test_add_with_pruning(context):
    context.context_limit = 40
    context.add("Long message 1", prune=True)
    context.add("Long message 2", prune=True)
    assert len(context.context) == 2
    assert context.context[1]["content"] == "Long message 2"

"""
Pytest test suite for the `context` module.
"""

import pytest
from modules.context import Context


@pytest.fixture()
def system_message():
    """Fixture that provides a single system message."""
    yield [{"role": "system", "content": "System message"}]


@pytest.fixture()
def context(system_message):
    """Fixture that returns a Context instance initialized with a system message."""
    system_message_copy = list(system_message)
    yield Context(system_message_copy)


def test_context_initialization():
    """Test that a new Context initializes with an empty list and default context limit."""
    context = Context()
    assert context.context == []
    assert context.context_limit == 4096


def test_add_single_message(context, system_message):
    """Test that a single string message is added correctly to the context."""
    context.add("Hello", prune=False)
    assert context.context == system_message + [
        {"role": "user", "content": "Hello"}
    ]


def test_add_list_of_messages(context, system_message):
    """Test that a list of messages is appended correctly to the context."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    context.add(messages, prune=False)
    assert context.context == system_message + messages


def test_get_context(context, system_message):
    """Test that get() returns the current context with all added messages."""
    messages = [{"role": "user", "content": "Test"}]
    context.add(messages, prune=False)
    assert context.get() == system_message + messages


def test_size_calculation(context):
    """Test that size() returns the correct total character count of all messages."""
    context = Context([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ])
    assert context.size() == 4 + 5 + 9 + 9


def test_clear_context(context):
    """Test that clear() removes all messages from the context."""
    context.add([{"role": "user", "content": "Test"}], prune=False)
    context.clear()
    assert context.context == []


def test_prune_context(context):
    """Test that prune() removes messages to stay within the context limit."""
    context.context_limit = 30
    context.add("This is a long message", prune=True)
    context.add("Short", prune=True)
    assert len(context.context) == 2
    assert context.context[1]["content"] == "Short"


def test_prune_preserves_first_message(context):
    """Test that prune() does not remove the first system message."""
    context.context_limit = 30
    context.add("This is a long message", prune=True)
    context.add("Another long message", prune=True)
    assert len(context.context) == 2
    assert context.context[0]["role"] == "system"


def test_add_without_pruning(context):
    """Test that messages can be added without triggering pruning when prune=False."""
    context.context_limit = 10
    context.add("Long message 1", prune=False)
    context.add("Long message 2", prune=False)
    assert len(context.context) == 3


def test_add_with_pruning(context):
    """Test that adding messages with prune=True enforces the context limit."""
    context.context_limit = 40
    context.add("Long message 1", prune=True)
    context.add("Long message 2", prune=True)
    assert len(context.context) == 2
    assert context.context[1]["content"] == "Long message 2"

"""
This module defines the `Context` class, which manages a sequence of chat messages
for use in language model conversations. It provides utility methods for adding
messages, pruning old entries to respect a size limit, and accessing or resetting
the current context state.

Each message is a dictionary with `role` and `content` keys, following a typical
chat-based API structure (e.g., OpenAI's ChatCompletion format).

Classes:
    Context: Maintains an ordered list of chat messages with optional pruning logic
             to enforce a maximum character size limit.
"""

from typing import Dict, List, Union


class Context:
    """
    A helper class to manage chat history context for a conversation.

    Attributes:
        context_limit: Maximum size of the context in characters.
        context: The chat history.
    """

    context_limit: int = 4096

    def __init__(self, context: List[Dict[str, str]] | None = None) -> None:
        """
        Initialize the Context instance.

        Args:
            context: Initial chat history. If None, an empty list is used.
        """
        if context is None:
            context = []
        self.context: List[Dict[str, str]] = context

    def add(
        self,
        message: Union[str, List[Dict[str, str]]],
        prune: bool = True,
        role: str = "user",
        replace_context: bool = False
    ) -> None:
        """
        Add a message or a list of messages to the context.

        Args:
            message: The message to add or a list of chat messages.
            prune: Whether to prune the context after adding. Defaults to True.
            role: Role of the sender (e.g., "user", "assistant"). Used only for single string messages.
            replace_context: Whether to replace the existing context with the new messages. Only applies if `message` is a list.
        """
        if isinstance(message, list):
            if replace_context:
                self.context = message
            else:
                self.context.extend(message)
        else:
            self.context.append(
                {
                    "role": role,
                    "content": message
                }
            )
        if prune:
            self.prune()

    def get(self) -> List[Dict[str, str]]:
        """
        Get the current chat context.

        Returns:
            The current context.
        """
        return self.context

    def size(self) -> int:
        """
        Calculate the size of the context in characters.

        Returns:
            Total number of characters in all context entries.
        """
        return sum(len(str(value)) for d in self.context for value in d.values())

    def clear(self) -> None:
        """
        Clear the current context.
        """
        self.context = []

    def prune(self) -> None:
        """
        Prune the context to fit within the context limit by removing messages
        from the beginning of the context, excluding the first two messages.
        """
        while self.size() > self.context_limit and len(self.context) > 2:
            self.context.pop(1)

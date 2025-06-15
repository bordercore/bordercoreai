"""
This module defines a `Vision` class that integrates an image into a prompt
and uses a language model to generate a response. The final message includes
both the image and associated text, and is sent to a chatbot model for processing.
"""

from typing import Any


class Vision():
    """
    A wrapper class for sending a multimodal prompt (text + image) to a chatbot model.
    """

    def __init__(self, model_name: str, message: list[dict[str, Any]], image: Any):
        """
        Initialize the Vision instance.

        Args:
            model_name: The name of the language model to use.
            message: A list of message dicts, with the last item being the user prompt.
            image: Image content to attach to the final prompt, in a format expected by the model.
        """
        self.model_name = model_name
        self.message = message
        self.image = image

    def __call__(self) -> str:
        """
        Format the message to include both image and text, then send it to the chatbot model.

        Returns:
            The streamed response from the chatbot model.
        """
        prompt = self.message[-1]["content"]
        self.message[-1]["content"] = [
            {
                "type": "image",
                "image": self.image,
            },
            {"type": "text", "text": prompt},
        ]

        from modules.chatbot import ChatBot
        chatbot = ChatBot(self.model_name)
        response = chatbot.send_message_to_model(self.message, {})
        return ChatBot.get_streaming_message(response)

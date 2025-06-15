"""
This module defines a Discord bot that listens for messages in a specific channel,
responds to bot mentions using a language model, and supports commands like "info"
and "reset" to interact with the conversation context.
"""

import os
import re
import sys
from typing import Any

import discord
import openai
from api import settings

from .chatbot import ChatBot

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = settings.discord_channel_id


class DiscordBot(discord.Client, ChatBot):
    """
    A Discord bot that responds to mentions using a language model.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the DiscordBot with appropriate intents and ChatBot setup.
        """
        if not DISCORD_TOKEN:
            print("Error: DISCORD_TOKEN not found.")
            sys.exit(1)

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        ChatBot.__init__(self, **kwargs)

    def get_message_content(self, message: str) -> str:
        """
        Remove the Discord mention prefix from a message.

        Args:
            message: The raw message content from Discord.

        Returns:
            The cleaned message content without bot mentions.
        """
        return re.sub(r"<@\d+> ", "", message)

    async def on_ready(self) -> None:
        """
        Called when the bot has successfully connected to Discord.
        """
        print(f"{self.user} has connected to Discord!")

    async def on_message(self, message: discord.Message) -> None:
        """
        Handle incoming Discord messages, respond to commands and mentions.

        Args:
            message: The message received from a Discord channel.
        """
        content = self.get_message_content(message.content)

        if content == "info":
            await message.channel.send("Model: " + ChatBot.get_model_info())
            return

        if content == "reset":
            await message.channel.send("Deleting current context...")
            self.context.clear()
            return

        if message.author == self.user:
            return

        if self.user.name in [x.name for x in message.mentions]:
            async with message.channel.typing():
                response = self.send_message_to_model(content)
            await message.channel.send(ChatBot.get_streaming_message(response))

    def run_bot(self) -> None:
        """
        Start the Discord bot using the provided DISCORD_TOKEN.
        """
        self.run(DISCORD_TOKEN)

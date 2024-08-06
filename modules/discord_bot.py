import os
import re
import sys

import openai
from api import settings

import discord

from .chatbot import ChatBot

DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
DISCORD_CHANNEL_ID = settings.discord_channel_id

openai.api_key = settings.openai_api_key


class DiscordBot(discord.Client, ChatBot):

    def __init__(self, **kwargs):
        if not DISCORD_TOKEN:
            print("Error: DISCORD_TOKEN not found.")
            sys.exit(1)

        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)
        ChatBot.__init__(self, **kwargs)

    def get_message_content(self, message):
        return re.sub(r"<@\d+> ", "", message)

    async def on_ready(self):
        print(f"{self.user} has connected to Discord!")

    async def on_message(self, message):

        content = self.get_message_content(message.content)
        if content == "info":
            await message.channel.send("Model: " + ChatBot.get_model_info())
            return
        elif content == "reset":
            await message.channel.send("Deleting current context...")
            self.context.clear()
            return

        if message.author == self.user:
            return

        if self.user.name in [x.name for x in message.mentions]:
            # Remove the message ID from the start of the message first
            async with message.channel.typing():
                response = self.send_message_to_model(re.sub(r"<@\d+> ", "", message.content))
            await message.channel.send(response["content"])

    def run_bot(self):
        self.run(DISCORD_TOKEN)

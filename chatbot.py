import argparse
import asyncio
import json
import logging
import os
import re
import string
import sys
import warnings
import wave

import discord
import openai
import piper
import pysbd
import requests
import simpleaudio
import sounddevice  # Adding this eliminates an annoying warning
import websockets
from pydub import AudioSegment
from pydub.playback import play

from govee import control_lights

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from whisper_mic.whisper_mic import WhisperMic

URI = "http://10.3.2.5:5000/api/v1/generate"
URI_INFO = "http://10.3.2.5:5000/api/v1/model"
URI_STREAM = "ws://10.3.2.5:5005/api/v1/stream"

DISCORD_TOKEN_CHAD = ""
DISCORD_TOKEN_FLOYD = ""
DISCORD_CHANNEL_ID = ""

MODE = "chat"

CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"
RED = "\033[91m"
END = "\033[0m"

openai.api_key = ""

seg = pysbd.Segmenter(language="en", clean=False)

logger = logging.getLogger("whisper_mic")
# Set the logger level to a higher level than any log messages you want to silence
logger.setLevel(logging.WARNING)
# Create a NullHandler to suppress the log messages
null_handler = logging.NullHandler()
logger.addHandler(null_handler)


class Context():

    context = []
    context_limit = 4096

    def add(self, role, message):
        self.context.append(
            {
                "role": role,
                "message": message
            }
        )
        self.prune()

    def get(self):
        output = ""
        for message in self.context:
            output = output + f"{message['message']}\n"

        return output

    def size(self):
        return len(self.get())

    def clear(self):
        self.context = []

    def prune(self):
        for message in list(self.context):
            if self.size() < self.context_limit:
                break
            self.context.pop(0)


class ChatBot():

    DISCORD_BOT_NAME = "FloydBot"
    ASSISTANT_NAME = "Luna"
    TEMPERATURE = 0.7

    def __init__(self, context, **args):
        self.context = context
        self.args = args
        if "temperature" in self.args:
            self.TEMPERATURE = self.args["temperature"]

    def get_prompt(self, prompt):

        return f"""
A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input
USER: {prompt}
ASSISTANT:
        """
#         return f"""
# You are a helpful AI assistant.

# USER: {prompt}
# ASSISTANT:
#     """

    # Remove punctuation and whitespace from the end of the string.
    def sanitize_string(self, input_string):
        while input_string and input_string[-1] in string.punctuation:
            input_string = input_string[:-1]
        return input_string.strip()

    def speak(self, message):
        voice = piper.PiperVoice.load(model_path="en_US-amy-medium.onnx", config_path="en_US-amy-medium.onnx.json")

        filename = "message.wav"
        with open(filename, "wb") as f:
            with wave.Wave_write(f) as wav:
                voice.synthesize(message, wav)

        # Play back the audio at a slighter faster speed
        audio = AudioSegment.from_file(filename, format="wav")
        faster_audio = audio.speedup(playback_speed=1.2)
        play(faster_audio)

        # wave_obj = simpleaudio.WaveObject.from_wave_file(filename)
        # play_obj = wave_obj.play()
        # play_obj.wait_done()

        os.remove(filename)

    def get_wake_word(self):
        return f"{self.ASSISTANT_NAME}".lower()

    def interactive(self):

        mic = WhisperMic(model="small", energy=100)
        active = False

        while True:
            if self.args["voice"]:
                print("Listening...")
                user_input = self.sanitize_string(mic.listen())
                if self.args["debug"]:
                    print(user_input)
                if not active:
                    if self.args["assistant"] and user_input.lower() != self.get_wake_word():
                        continue
                    else:
                        active = True
                        self.speak("I'm listening")
                        continue
                else:
                    if user_input.lower() == "goodbye":
                        self.speak("Be seeing you")
                        sys.exit(0)
                print(f"\b\b\b\b\b\b\b\b\b\b\b\b{user_input}")
            else:
                user_input = input(f"{MAGENTA}Prompt:{END} ")

            if self.args["assistant"]:
                print("Processing...")

            _ = self.send_message_to_model_stream(user_input)

            # if self.args["speak"]:
            #     self.speak(result)

    def get_chatbot_params(self):

        # Note: the selected defaults change from time to time.
        return {
            "prompt": self.context.get(),
            "mode": self.args["chat_mode"],
            "max_tokens_second": 0,
            "auto_max_new_tokens": True,
            "new_conversation": self.args["new_conversation"] or False,

            # Generation params. If "preset" is set to different than "None", the values
            # in presets/preset-name.yaml are used instead of the individual numbers.
            "preset": "None",
            "do_sample": True,
            "temperature": self.TEMPERATURE,
            "top_p": 0.1,
            "typical_p": 1,
            "epsilon_cutoff": 0,  # In units of 1e-4
            "eta_cutoff": 0,  # In units of 1e-4
            "tfs": 1,
            "top_a": 0,
            "repetition_penalty": 1.18,
            "repetition_penalty_range": 0,
            "top_k": 40,
            "min_length": 0,
            "no_repeat_ngram_size": 0,

            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "mirostat_mode": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "guidance_scale": 1,
            "negative_prompt": "",

            "seed": -1,
            "add_bos_token": True,
            "truncation_length": 2048,
            "ban_eos_token": False,
            "skip_special_tokens": True,
            "stopping_strings": []
        }

    async def run(self, context):
        request = self.get_chatbot_params()

        async with websockets.connect(URI_STREAM, ping_interval=None) as websocket:
            await websocket.send(json.dumps(request))

            # yield context  # Remove this if you just want to see the reply

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)

                if incoming_data["event"] == "text_stream":
                    yield incoming_data["text"]
                elif incoming_data["event"] == "stream_end":
                    return

    def check_if_sentence(self):
        sentences = seg.segment(self.fragment)
        if len(sentences) > 1:
            if self.args["speak"]:
                self.speak(sentences[0])
            self.fragment = " ".join(sentences[1:])

    async def print_response_stream(self, prompt):
        self.fragment = ""
        self.result = ""
        print(f"{CYAN}")

        async for response in self.run(prompt):
            if response.strip() == "":
                continue
            self.result += response
            self.fragment += response
            print(response, end="")
            sys.stdout.flush()  # If we don't flush, we won't see tokens in realtime.
            self.check_if_sentence()

        print(f"{END}\n")
        self.context.add("system", self.result)

    def handle_prompt(self, prompt_raw):
        if prompt_raw.strip() == "info":
            return ChatBot.get_model_info()

        prompt = self.get_prompt(prompt_raw)
        self.context.add("user", prompt)

    def send_message_to_model_stream(self, prompt_raw):
        self.handle_prompt(prompt_raw)
        asyncio.run(self.print_response_stream(prompt_raw))
        return self.result

    def send_message_to_model(self, prompt_raw):

        if self.args["control_lights"] == "true":
            return control_lights(prompt_raw)

        self.handle_prompt(prompt_raw)
        request = self.get_chatbot_params()

        response_model = requests.post(URI, json=request)
        content = response_model.json()["results"][0]["text"].strip()

        self.context.add("system", content)
        if response_model.status_code == 200:
            if self.args["debug"]:
                print(f"\n{CYAN}{content}\n")
        else:
            print(f"Error: {response_model}")
            sys.exit(1)

        return content

    @staticmethod
    def get_model_info():
        response = requests.post(URI_INFO, json={"action": "info"})
        return response.json()["result"]

    @staticmethod
    def get_model_list():
        response = requests.post(URI_INFO, json={"action": "list"})
        return ChatBot.get_personal_model_names([x for x in response.json()["result"] if x != "None"])

    @staticmethod
    def get_personal_model_names(model_list):
        mapping = {
            "TheBloke_CodeLlama-7B-Instruct-AWQ": "CodeLlama 7B Instruct",
            "TheBloke_CodeLlama-13B-Instruct-AWQ": "CodeLlama 13B Instruct",
            "TheBloke_Karen_theEditor_13B-AWQ": "Karen 13B",
            "TheBloke_WizardLM-13B-V1-0-Uncensored-SuperHOT-8K-GPTQ": "Floyd -- WizardLM 13B",
            "TheBloke_dolphin-2.1-mistral-7B-AWQ": "Daisy -- Mistral 7B"
        }
        models = [
            {
                "model": x,
                "name": mapping.get(x, x)
            }
            for x in
            model_list
        ]
        return models

    @staticmethod
    def load_model(model):
        response = requests.post(URI_INFO, json={"action": "load", "model_name": model})
        return response.json()["result"]


class DiscordBot(discord.Client, ChatBot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_message_content(self, message):
        return re.sub(r"<@\d+> ", "", message)

    async def on_ready(self):
        print(f"{self.user} has connected to Discord!")

    async def on_message(self, message):

        if message.author == client.user:
            return

        if self.discord_bot_name in [x.name for x in message.mentions]:
            # Remove the message ID from the start of the message first
            async with message.channel.typing():
                response = self.send_message_to_model(re.sub(r"<@\d+> ", "", message.content))
            await message.channel.send(response)


class FloydBot(DiscordBot):

    def __init__(self, **kwargs):
        self.discord_bot_name = "FloydBot"
        super().__init__(**kwargs)

    async def on_message(self, message):
        content = self.get_message_content(message.content)
        if content == "info":
            await message.channel.send("Model: " + ChatBot.get_model_info()["model_name"])
        elif content == "reset":
            await message.channel.send("Deleting current context...")
            self.context.clear()
        else:
            await super().on_message(message)


class ChatGTPDiscordBot(DiscordBot):

    def __init__(self, **kwargs):
        self.discord_bot_name = "ChadBot"
        super().__init__(**kwargs)

    def send_message_to_model(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", "--assistant", help="Assistant mode", action="store_true")
    parser.add_argument("-c", "--chat-mode", choices=["instruct", "chat"], default="chat", help="The chat mode: intruct or chat")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-m", "--mode", choices=["chatgpt", "floyd", "interactive"], default="interactive", help="The mode: interactive, floyd on discord, chad on discord")
    parser.add_argument("-s", "--speak", help="Voice output", action="store_true")
    parser.add_argument("-v", "--voice", help="Voice input", action="store_true")
    args = parser.parse_args()

    assistant = args.assistant
    mode = args.mode
    chat_mode = args.chat_mode
    speak = args.speak
    voice = args.voice

    context = Context()

    if mode == "interactive":
        chatbot = ChatBot(context, assistant=args.assistant, debug=args.debug, chat_mode=chat_mode, voice=voice, speak=speak)
        chatbot.interactive()
    elif mode == "chatgpt":
        intents = discord.Intents.default()
        intents.message_content = True
        client = ChatGTPDiscordBot(intents=intents)
        client.context = context
        client.args = {"debug": args.debug, "chat_mode": chat_mode}
        client.run(DISCORD_TOKEN_CHAD)
    elif mode == "floyd":
        intents = discord.Intents.default()
        intents.message_content = True
        client = FloydBot(intents=intents)
        client.context = context
        client.args = {"debug": args.debug, "chat_mode": chat_mode}
        client.run(DISCORD_TOKEN_FLOYD)

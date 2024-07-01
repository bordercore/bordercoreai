import argparse
import json
import logging
import os
import re
import string
import sys
import time
import urllib.parse
import warnings
import wave

import discord
import openai
import piper
import pyaudio
import pysbd
import requests
import sounddevice  # Adding this eliminates an annoying warning
import sseclient
from pydub import AudioSegment
from pydub.playback import play
from requests.exceptions import ConnectionError

from api import settings
from context import Context
from modules.calendar import get_schedule
from modules.govee import run_command
from modules.music import play_music
from modules.weather import get_weather_info
from util import get_model_info, sort_models

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

try:
    from whisper_mic.whisper_mic import WhisperMic
except ImportError:
    # WhisperMic will complain if imported without X. This is fine, since
    #  sometimes I want to run this code as a daemon using supervisor
    pass

HOST = settings.api_host
URI_CHAT = f"{HOST}/v1/chat/completions"
URI_MODEL_INFO = f"{HOST}/v1/internal/model/info"
URI_MODEL_LIST = f"{HOST}/v1/internal/model/list"
URI_MODEL_LOAD = f"{HOST}/v1/internal/model/load"

DISCORD_TOKEN_CHAD = os.environ.get("DISCORD_TOKEN_CHAD")
DISCORD_TOKEN_FLOYD = os.environ.get("DISCORD_TOKEN_FLOYD")
DISCORD_CHANNEL_ID = settings.discord_channel_id

MODE = "chat"

CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"
RED = "\033[91m"
END = "\033[0m"

openai.api_key = settings.openai_api_key

seg = pysbd.Segmenter(language="en", clean=False)

logger = logging.getLogger("whisper_mic")
# Set the logger level to a higher level than any log messages you want to silence
logger.setLevel(logging.WARNING)
# Create a NullHandler to suppress the log messages
null_handler = logging.NullHandler()
logger.addHandler(null_handler)

model_info = get_model_info()

if not DISCORD_TOKEN_CHAD:
    print("Error: DISCORD_TOKEN_CHAD not found.")
    sys.exit(1)
if not DISCORD_TOKEN_FLOYD:
    print("Error: DISCORD_TOKEN_FLOYD not found.")
    sys.exit(1)


class ChatBot():

    ASSISTANT_NAME = "Luna"
    TEMPERATURE = 0.7

    def __init__(self, model_name=None, **args):
        self.context = Context()
        self.args = args
        self.model_name = model_name
        if "temperature" in self.args:
            self.TEMPERATURE = self.args["temperature"]

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

        os.remove(filename)

    def get_wake_word(self):
        return f"{self.ASSISTANT_NAME}".lower()

    def play_response(self, response):

        text = urllib.parse.quote(response)
        host = settings.tts_host
        voice = settings.tts_voice
        output_file = "output.wav"
        url = f"http://{host}/api/tts-generate-streaming?text={text}&voice={voice}&language=en&output_file={output_file}"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            p = pyaudio.PyAudio()
            stream = p.open(format=8, channels=1, rate=24000, output=True)
            for chunk in response.iter_content(chunk_size=1024):
                stream.write(chunk)
        else:
            print(f"Failed to get audio: status_code = {response.status_code}")


    def interactive(self):

        if self.args["voice"]:
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
                user_input = input(f"\n{MAGENTA}You{END} ")

            if self.args["assistant"]:
                print("Processing...")

            try:
                response = self.send_message_to_model(user_input)
                print(f"\n{MAGENTA}AI{CYAN} {response['content']}")
                if self.args["speak"]:
                    self.play_response(response["content"])

            except ConnectionError:
                print("Error: API refusing connections.")

    def get_chatbot_params(self):

        # Note: the selected defaults change from time to time.
        return {
            "messages": self.context.get(),
            "mode": self.args["chat_mode"],
            "max_tokens_second": 0,
            "auto_max_new_tokens": True,
            "new_conversation": self.args.get("new_conversation", False),

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

    def handle_prompt(self, prompt_raw):
        # If we're passing in a list, assume this is the complete
        # chat history and replace it with the new
        if type(prompt_raw) is list:
            self.context.set(prompt_raw)
        else:
            if prompt_raw.strip() == "info":
                return ChatBot.get_model_info()

            self.context.add("user", prompt_raw)

    def send_message_to_model_stream(self, prompt_raw):
        self.handle_prompt(prompt_raw)

        data = {
            "mode": "instruct",
            "stream": True,
            "messages": self.context.get()
        }

        headers = {
            "Content-Type": "application/json"
        }

        stream_response = requests.post(URI_CHAT, headers=headers, json=data, verify=False, stream=True)
        client = sseclient.SSEClient(stream_response)

        assistant_message = ""
        print(f"\n{MAGENTA}AI{CYAN} ", end="")
        for event in client.events():
            payload = json.loads(event.data)
            chunk = payload["choices"][0]["message"]["content"]
            assistant_message += chunk
            print(chunk, end="")

        self.context.add("assistant", assistant_message)

    def handle_message(self, prompt):

        request_type = self.get_request_type(prompt)

        if request_type["category"] == "lights":
            return run_command(self.model_name, prompt)
        elif request_type["category"] == "music":
            return play_music(self.model_name, prompt)
        elif request_type["category"] == "weather":
            return get_weather_info(self.model_name, prompt)
        elif request_type["category"] == "calendar":
            return get_schedule(self.model_name, prompt)
        else:
            return self.send_message_to_model(prompt)

    def get_request_type(self, message):
        prompt = """
        I want you to put this instruction into one of multiple categories. If the instruction is to play some music, the category is "music". If the instruction is to control lights, the category is "lights". If the instruction is asking about the weather or the moon's phase, the category is "weather". If the instruction is asking about today's calendar, or is something like 'What's happening today' or 'What is my schedule', the category is "calendar". For everything else, the category is "other". Give me the category in JSON format with the field name "category". Do not format the JSON by including newlines. Give only the JSON and no additional characters, text, or comments. Here is the instruction:
        """
        prompt = prompt + message

        args = {"temperature": 0.1}
        response = self.send_message_to_model(prompt, args)
        print(f"{response=}")
        return json.loads(response["content"])

    def send_message_to_model(self, prompt, args={}):
        messages = [{"role": "user", "content": prompt}]
        model_type = ChatBot.get_model_type(self.model_name)
        if model_type == "openai":
            return self.send_message_to_model_openai(messages, args)
        else:
            return self.send_message_to_model_local_llm(messages, args)

    def send_message_to_model_openai(self, messages, args={}):
        start = time.time()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            **args
        )
        speed = int(response["usage"]["completion_tokens"] / (time.time() - start))
        return {
            "content": response["choices"][0]["message"]["content"],
            "speed": speed
        }

    def send_message_to_model_local_llm(self, messages, args):
        # params = self.get_chatbot_params()
        # request = {
        #     **params,
        #     **args
        # }
        request = {
            "mode": "instruct",
            "messages": messages,
            **args
        }
        response = requests.post(URI_CHAT, json=request)
        payload = response.json()
        if response.status_code != 200:
            raise Exception(f"Error from local LLM: {response}")
        speed = payload["choices"][0]["message"]["speed"]
        return {
            "content": payload["choices"][0]["message"]["content"],
            "speed": speed
        }

    @staticmethod
    def get_model_type(model_name):
        if model_name and \
           model_name in model_info and \
           "type" in model_info[model_name]:
            return model_info[model_name]["type"]

    @staticmethod
    def get_model_info():
        response = requests.get(URI_MODEL_INFO)
        return response.json()["model_name"]

    @staticmethod
    def get_model_list():
        response = requests.get(URI_MODEL_LIST)

        model_names = response.json()["model_names"]

        # Add proprietary models
        model_names.extend(["gpt-4o", "gpt-4-turbo"])

        model_list = ChatBot.get_personal_model_names(model_names)

        return sort_models(
            model_list,
            [v.get("name", None) for k, v in model_info.items()]
        )

    @staticmethod
    def get_personal_model_names(model_list):
        models = [
            {
                "model": x,
                "name": model_info.get(x, {"name": x}).get("name", x),
                "type": model_info.get(x, {"type": x}).get("type", None),
            }
            for x in
            model_list
        ]
        return models

    @staticmethod
    def load_model(model):
        response = requests.post(URI_MODEL_LOAD, json={"model_name": model})
        return response.json()


class DiscordBot(discord.Client, ChatBot):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ChatBot.__init__(self, **kwargs)

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
            await message.channel.send(response["content"])


class FloydBot(DiscordBot):

    def __init__(self, **kwargs):
        self.discord_bot_name = "FloydBot"
        super().__init__(**kwargs)

    async def on_message(self, message):
        content = self.get_message_content(message.content)
        if content == "info":
            await message.channel.send("Model: " + ChatBot.get_model_info())
        elif content == "reset":
            await message.channel.send("Deleting current context...")
            self.context.clear()
        else:
            await super().on_message(message)


class ChatGPTDiscordBot(DiscordBot):

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

    if mode == "interactive":
        chatbot = ChatBot(assistant=args.assistant, debug=args.debug, chat_mode=chat_mode, voice=voice, speak=speak)
        chatbot.interactive()
    elif mode == "chatgpt":
        intents = discord.Intents.default()
        intents.message_content = True
        client = ChatGPTDiscordBot(intents=intents)
        client.args = {"debug": args.debug, "chat_mode": chat_mode}
        client.run(DISCORD_TOKEN_CHAD)
    elif mode == "floyd":
        intents = discord.Intents.default()
        intents.message_content = True
        client = FloydBot(intents=intents)
        client.args = {"debug": args.debug, "chat_mode": chat_mode}
        client.run(DISCORD_TOKEN_FLOYD)

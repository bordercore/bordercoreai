import argparse
import json
import logging
import os
import string
import sys
import time
import urllib.parse
import warnings
import wave

import anthropic
import openai
import piper
import pyaudio
import pysbd
import requests
import sounddevice  # Adding this eliminates an annoying warning
import sseclient
from api import settings
from http_constants.status import HttpStatus
from pydub import AudioSegment
from pydub.playback import play
from requests.exceptions import ConnectionError

from modules.calendar import get_schedule
from modules.context import Context
from modules.govee import run_command
from modules.music import play_music
from modules.util import get_model_info, sort_models
from modules.weather import get_weather_info

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
        if response.status_code == HttpStatus.OK:
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

            try:
                response = self.send_message_to_model(user_input)
                print(f"\n{MAGENTA}AI{CYAN} {response['content']}")
                if self.args["speak"]:
                    self.play_response(response["content"])

            except ConnectionError:
                print("Error: API refusing connections.")

    def send_message_to_model_stream(self, prompt_raw):
        self.context.add(prompt_raw)

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
        request_type = self.get_request_type(prompt[-1]["content"])

        if request_type["category"] == "lights":
            return run_command(self.model_name, prompt[-1]["content"])
        elif request_type["category"] == "music":
            return play_music(self.model_name, prompt[-1]["content"])
        elif request_type["category"] == "weather":
            return get_weather_info(self.model_name, prompt[-1]["content"])
        elif request_type["category"] == "calendar":
            return get_schedule(self.model_name, prompt[-1]["content"])
        elif request_type["category"] == "summary":
            return self.get_summary()
        else:
            self.context.clear()
            return self.send_message_to_model(prompt, replace_context=True)

    def get_summary(self):
        response = get_weather_info(self.model_name, "What's the weather today?")
        content = response["content"]
        speed = response["speed"]

        response = get_schedule(self.model_name, "What's on my calendar today?")
        content += "\n\n" + response["content"]

        return {
            "content": content,
            "speed": int((speed + response["speed"]) / 2)
        }

    def get_request_type(self, message):
        prompt = """
        I want you to put this instruction into one of multiple categories. If the instruction is to play some music, the category is "music". If the instruction is to control lights, the category is "lights". If the instruction is asking about the weather or the moon's phase, the category is "weather". If the instruction is asking about today's calendar, or is something like 'What's happening today' or 'What is my schedule', the category is "calendar". If the instruction is asking about today's agenda or summary, the category is "summary". For everything else, the category is "other". Give me the category in JSON format with the field name "category". Do not format the JSON by including newlines. Give only the JSON and no additional characters, text, or comments. Here is the instruction:
        """
        prompt = prompt + message

        args = {"temperature": 0.1}
        response = self.send_message_to_model(prompt, args, prune=False)

        if settings.debug:
            print(f"{response=}")

        return json.loads(response["content"])

    def send_message_to_model(self, messages, args={}, prune=True, replace_context=False):
        if type(messages) is not list:
            messages = [{"role": "user", "content": messages}]
        self.context.add(messages, prune, replace_context)

        model_vendor = ChatBot.get_model_attribute(self.model_name, "vendor")
        if model_vendor == "openai":
            response = self.send_message_to_model_openai(args)
        elif model_vendor == "anthropic":
            response = self.send_message_to_model_anthropic(args)
        else:
            response = self.send_message_to_model_local_llm(args)

        self.context.add(response["content"], True, role="assistant")
        return response

    def send_message_to_model_openai(self, args):
        start = time.time()
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.context.get(),
            **args
        )
        speed = int(response["usage"]["completion_tokens"] / (time.time() - start))
        return {
            "content": response["choices"][0]["message"]["content"],
            "speed": speed
        }

    def send_message_to_model_anthropic(self, args):
        start = time.time()

        messages = self.context.get()

        # Anthropic will reject messages with extraneous attributes
        [x.pop("id", None) for x in messages]

        # Anthropic requires any system messages to be provided
        #  as a separate parameter and not be present in the
        #  list of user messages.
        system = []
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            messages.pop(0)

        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key
        )
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=messages,
            system=system,
            **args
        )
        speed = int(response.usage.output_tokens / (time.time() - start))
        return {
            "content": response.content[0].text,
            "speed": speed
        }

    def send_message_to_model_local_llm(self, args):
        request = {
            "mode": "instruct",
            "messages": self.context.get(),
            **args
        }

        response = requests.post(URI_CHAT, json=request)
        if response.status_code != HttpStatus.OK:
            raise Exception(f"Error from local LLM: {str(HttpStatus(response.status_code))}")
        payload = response.json()
        speed = payload["choices"][0]["message"]["speed"]
        return {
            "content": payload["choices"][0]["message"]["content"],
            "speed": speed
        }

    @staticmethod
    def get_model_attribute(model_name, attribute):
        if model_name and \
           model_name in model_info and \
           attribute in model_info[model_name]:
            return model_info[model_name][attribute]

    @staticmethod
    def get_model_info():
        response = requests.get(URI_MODEL_INFO)
        return response.json()["model_name"]

    @staticmethod
    def get_model_list():
        response = requests.get(URI_MODEL_LIST)

        model_names = response.json()["model_names"]

        # Add API-based models
        model_names.extend(
            [
                k
                for k, v
                in model_info.items()
                if "type" in v and v["type"] == "api"]
        )

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
        current_model = ChatBot.get_model_info()
        if current_model == model:
            return {"status": "OK"}
        else:
            return requests.post(URI_MODEL_LOAD, json={"model_name": model}).json()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", "--assistant", help="Assistant mode", action="store_true")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true")
    parser.add_argument("-m", "--mode", choices=["chatgpt", "localllm", "interactive"], default="interactive", help="The mode: interactive, localllm on discord, chatgpt on discord")
    parser.add_argument("-s", "--speak", help="Voice output", action="store_true")
    parser.add_argument("-v", "--voice", help="Voice input", action="store_true")
    args = parser.parse_args()

    assistant = args.assistant
    mode = args.mode
    speak = args.speak
    voice = args.voice

    if mode == "interactive":
        chatbot = ChatBot(assistant=args.assistant, debug=args.debug, voice=voice, speak=speak)
        chatbot.interactive()
    elif mode == "chatgpt":
        from modules.discord_bot import DiscordBot
        client = DiscordBot(model_name="gpt-4o-mini")
        client.run_bot()
    elif mode == "localllm":
        from modules.discord_bot import DiscordBot
        client = DiscordBot()
        client.run_bot()

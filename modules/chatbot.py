import argparse
import json
import logging
import string
import sys
import tempfile
import urllib.parse
import warnings

import anthropic
import openai
import pysbd
import requests
import sounddevice  # Adding this eliminates an annoying warning
from api import settings
from http_constants.status import HttpStatus
from requests.exceptions import ConnectionError

from modules.context import Context
from modules.google_calendar import get_schedule
from modules.govee import control_lights
from modules.music import play_music
from modules.util import get_model_info, get_webpage_contents, sort_models
from modules.weather import get_weather_info
from modules.wolfram_alpha import WolframAlphaFunctionCall

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

RED = "\033[91m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"

CONTROL_VALUE = "9574724975"

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
        self.model_name = model_name
        self.args = args
        if "temperature" in self.args:
            self.TEMPERATURE = self.args["temperature"]

    # Remove punctuation and whitespace from the end of the string.
    def sanitize_string(self, input_string):
        while input_string and input_string[-1] in string.punctuation:
            input_string = input_string[:-1]
        return input_string.strip()

    def get_wake_word(self):
        return f"{self.ASSISTANT_NAME}".lower()

    def speak(self, text):

        text = urllib.parse.quote(text)
        host = settings.tts_host
        voice = settings.tts_voice
        output_file = "stream_output.wav"
        url = f"http://{host}/api/tts-generate-streaming?text={text}&voice={voice}&language=en&output_file={output_file}"
        response = requests.get(url, stream=True)

        if response.status_code == HttpStatus.OK:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(response.raw.read())
                # Set playsounds' logger level to ERROR to suppress this warning:
                #  "playsound is relying on another python subprocess..."
                logging.getLogger("playsound").setLevel(logging.ERROR)
                from playsound import playsound
                playsound(temp_file.name)
        else:
            print(f"Failed to get audio: status_code = {response.status_code}")


    def interactive(self, inference=None):

        if self.args["stt"]:
            print("Loading STT package...")
            mic = WhisperMic(model="small", energy=100)
            active = False

        while True:
            if self.args["stt"]:
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
                try:
                    user_input = input(f"\n{COLOR_GREEN}You:{COLOR_RESET} ")
                except KeyboardInterrupt:
                    sys.exit(0)

            try:
                if inference:
                    inference.context.add(user_input, True)
                    response = inference.generate(inference.context.get())
                else:
                    response = self.send_message_to_model(user_input)
                print(f"\n{COLOR_BLUE}AI{COLOR_RESET} ", end="")
                content = ""
                for x in response:
                    content += x
                    print(x, end="", flush=True)
                print()
                if self.args["tts"]:
                    self.speak(content)

            except ConnectionError:
                print("Error: API refusing connections.")

    def handle_message(self, messages):
        if self.args.get("wolfram_alpha", False):
            request_type = {"category": "math"}
        elif self.args.get("url", None):
            request_type = {"category": "other"}
            contents = get_webpage_contents(self.args.get("url"))
            messages[-1]["content"] += f": {contents}"
        else:
            request_type = self.get_request_type(messages[-1]["content"])

        if request_type["category"] == "lights":
            return control_lights(self.model_name, messages[-1]["content"])
        elif request_type["category"] == "music":
            return play_music(self.model_name, messages[-1]["content"])
        elif request_type["category"] == "weather":
            return get_weather_info(self.model_name, messages[-1]["content"])
        elif request_type["category"] == "calendar":
            return get_schedule(self.model_name, messages[-1]["content"])
        elif request_type["category"] == "agenda":
            return self.get_agenda()
        elif request_type["category"] == "math":
            func_call = WolframAlphaFunctionCall(self.model_name)
            return func_call.run(messages[-1]["content"])
        else:
            return self.send_message_to_model(messages, replace_context=True)

    def send_message_to_model(self, messages, args={}, prune=True, replace_context=False, tool_name=None, tool_list=None):
        if type(messages) is not list:
            messages = [{"role": "user", "content": messages}]
        self.context.add(messages, prune, replace_context)

        model_vendor = ChatBot.get_model_attribute(self.model_name, "vendor")
        if model_vendor == "openai":
            response = self.send_message_to_model_openai(args)
        elif model_vendor == "anthropic":
            response = self.send_message_to_model_anthropic(args)
        else:
            response = self.send_message_to_model_local_llm(args, tool_name, tool_list)

        return response

    def send_message_to_model_openai(self, args):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.context.get(),
            stream=True,
            **args
        )
        for chunk in response:
            content = chunk["choices"][0]["delta"].get("content", "")
            yield content

    def send_message_to_model_anthropic(self, args):
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
            stream=True,
            **args
        )
        for chunk in response:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text

    def send_message_to_model_local_llm(self, args, tool_name, tool_list):
        request = {
            "mode": "instruct",
            "messages": self.context.get(),
            "tool_name": tool_name,
            "tool_list": tool_list,
            "temperature": self.TEMPERATURE,
            **args
        }

        response = requests.post(URI_CHAT, json=request, stream=True)
        content = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # Filter out keep-alive new chunks
                content += chunk.decode("utf-8")
                yield chunk.decode("utf-8")
        self.context.add(content, role="assistant")

    def get_agenda(self):
        response = get_weather_info(self.model_name, "What's the weather today?")
        weather_content = ChatBot.get_streaming_message(response)

        response = get_schedule(self.model_name, "What's on my calendar today?")
        calendar_content = ChatBot.get_streaming_message(response)

        return f"{weather_content}\n\n{calendar_content}"

    def get_request_type(self, message):
        prompt = """
        I want you to put this instruction into one of multiple categories. If the instruction is to play some music, the category is "music". If the instruction is to control lights, the category is "lights". If the instruction is asking about the weather or the moon's phase, the category is "weather". If the instruction is asking about today's calendar, or is something like 'What's happening today' or 'What is my schedule', the category is "calendar". If the instruction is asking about today's agenda, or something like 'What's my update?', the category is "agenda". If the instruction is asking for mathematical calculation, the category is "math". For everything else, the category is "other". Give me the category in JSON format with the field name "category". Do not format the JSON by including newlines. Give only the JSON and no additional characters, text, or comments. Here is the instruction:
        """
        prompt += message

        chatbot = ChatBot(self.model_name, temperature=0.1)
        response = chatbot.send_message_to_model(prompt)
        content = ChatBot.get_streaming_message(response)

        if settings.debug:
            print(f"{content=}")

        return json.loads(content)

    @staticmethod
    def get_streaming_message(streamer):
        response = ""
        for x in streamer:
            response += x
        return response

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
    parser.add_argument(
        "-a",
        "--assistant",
        help="Assistant mode",
        action="store_true"
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Debug mode",
        action="store_true"
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["chatgpt", "localllm", "interactive"],
        default="interactive",
        help="The mode: interactive, localllm on discord, chatgpt on discord"
    )
    parser.add_argument(
        "--tts",
        help="TTS (Text to Speech)",
        action="store_true"
    )
    parser.add_argument(
        "--stt",
        help="STT (Speech to Text)",
        action="store_true"
    )
    args = parser.parse_args()
    assistant = args.assistant
    mode = args.mode
    tts = args.tts
    stt = args.stt

    if mode == "interactive":
        chatbot = ChatBot(assistant=args.assistant, debug=args.debug, stt=stt, tts=tts)
        chatbot.interactive()
    elif mode == "chatgpt":
        from modules.discord_bot import DiscordBot
        client = DiscordBot(model_name="gpt-4o-mini")
        client.run_bot()
    elif mode == "localllm":
        from modules.discord_bot import DiscordBot
        client = DiscordBot()
        client.run_bot()

"""
This module defines the `ChatBot` class, which provides an interactive interface for
communicating with language models via local or remote APIs (e.g., OpenAI, Anthropic).
It supports multiple capabilities including:

- Message routing based on intent classification
- Integration with tools like music playback, smart lighting, calendar, and weather
- Support for voice interaction (STT and TTS)
- Streaming output from model completions
- Model management (listing, loading, metadata retrieval)

The chatbot can operate in different modes including interactive CLI or as a backend
for services like Discord bots.

Configuration is handled via command-line arguments and the `api.settings` module.
"""

import argparse
import json
import logging
import string
import sys
import tempfile
import urllib.parse
import warnings
from typing import (Any, Dict, Generator, Iterable, Iterator, List, Optional,
                    Union)

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
from modules.util import (get_model_info, get_webpage_contents, sort_models,
                          strip_code_fences)
from modules.weather import get_weather_info
from modules.wolfram_alpha import WolframAlphaFunctionCall

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

try:
    from whisper_mic.whisper_mic import WhisperMic
except ImportError:
    # WhisperMic will complain if imported without X. This is fine, since
    #  sometimes I want to run this code as a daemon using supervisor
    pass


RED = "\033[91m"
COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"

CONTROL_VALUE = "9574724975"

seg = pysbd.Segmenter(language="en", clean=False)

logger = logging.getLogger("whisper_mic")
# Set the logger level to a higher level than any log messages you want to silence
logger.setLevel(logging.WARNING)
# Create a NullHandler to suppress the log messages
null_handler = logging.NullHandler()
logger.addHandler(null_handler)

model_info = get_model_info()


class ChatBot():
    """
    ChatBot provides an interactive command-line interface to Luna, supporting
    local LLMs, OpenAI, and Anthropic APIs, as well as TTS, STT, and various tools.
    """

    ASSISTANT_NAME = "Luna"
    temperature = 0.7

    def __init__(self, model_name: Optional[str] = None, **args: Any) -> None:
        """
        Initialize a ChatBot instance.

        Args:
            model_name: Name of the model to use (API or local).
            **args: Arbitrary keyword arguments to configure behavior (e.g., temperature, stt, tts).
        """
        self.context = Context()
        self.model_name = model_name
        self.args = args

        if "temperature" in self.args:
            self.temperature = self.args["temperature"]

    @staticmethod
    def get_api_endpoints() -> dict[str, str]:
        """
        Return the endpoints for local LLM HTTP API interactions.

        Returns:
            Mapping of endpoint keys to full URL strings.
        """
        host = settings.api_host
        return {
            "CHAT": f"{host}/v1/chat/completions",
            "MODEL_INFO": f"{host}/v1/internal/model/info",
            "MODEL_LIST": f"{host}/v1/internal/model/list",
            "MODEL_LOAD": f"{host}/v1/internal/model/load",
        }

    # Remove punctuation and whitespace from the end of the string.
    def sanitize_string(self, input_string: str) -> str:
        """
        Remove trailing punctuation and whitespace from a string.

        Args:
            input_string: The raw string to sanitize.
        Returns:
            A trimmed string without trailing punctuation.
        """
        while input_string and input_string[-1] in string.punctuation:
            input_string = input_string[:-1]
        return input_string.strip()

    def get_wake_word(self) -> str:
        """
        Get the lowercase wake word for activating voice mode.

        Returns:
            The wake word string.
        """
        return f"{self.ASSISTANT_NAME}".lower()

    def speak(self, text: str) -> None:
        """
        Perform text-to-speech for the given text and play audio.

        Args:
            text: The text string to vocalize.
        """
        text = urllib.parse.quote(text)
        host = settings.tts_host
        voice = settings.tts_voice
        output_file = "stream_output.wav"
        url = f"http://{host}/api/tts-generate-streaming?text={text}&voice={voice}&language=en&output_file={output_file}"
        response = requests.get(url, stream=True, timeout=20)

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

    def interactive(self, inference: Optional[Any] = None) -> None:
        """
        Enter an interactive loop reading user input and printing AI responses.
        """
        mic = self.init_stt_if_enabled()
        active = False

        while True:
            user_input = self.get_user_input(mic, active)
            if user_input is None:
                continue
            if self.args["stt"] and not active:
                active = True
                self.speak("I'm listening")
                continue
            if user_input.lower() == "goodbye":
                self.speak("Be seeing you")
                sys.exit(0)

            self.handle_response(user_input, inference)

    def init_stt_if_enabled(self) -> Optional["WhisperMic"]:
        """Initialise the WhisperMic when STT is turned on.

        Returns:
            WhisperMic instance when ``self.args["stt"]`` is truthy; otherwise
            ``None``.
        """
        if self.args["stt"]:
            print("Loading STT package...")
            return WhisperMic(model="small", energy=100)
        return None

    def get_user_input(
        self,
        mic: Optional["WhisperMic"],
        active: bool,
    ) -> Optional[str]:
        """Retrieve a single line of user input (voice or keyboard).

        Args:
            mic: Active ``WhisperMic`` instance if STT is enabled, else ``None``.
            active: ``True`` once the wake-word has been detected; determines
                whether normal utterances are processed or ignored.

        Returns:
            A sanitised input string, or ``None`` when:
            * the wake-word has not yet been spoken, or
            * no actionable input was captured.
        """
        if self.args["stt"]:
            print("Listening...")
            user_input = self.sanitize_string(mic.listen())
            if self.args["debug"]:
                print(user_input)
            if self.args["assistant"] and not active and user_input.lower() != self.get_wake_word():
                return None
            print(f"\b\b\b\b\b\b\b\b\b\b\b\b{user_input}")
            return user_input
        try:
            return input(f"\n{COLOR_GREEN}You:{COLOR_RESET} ")
        except KeyboardInterrupt:
            sys.exit(0)

    def handle_response(self, user_input: str, inference: Optional[Any]) -> None:
        """Generate the assistant’s reply and (optionally) speak it aloud.

        Args:
            user_input: The final, cleaned user utterance.
            inference: External inference engine providing ``context`` and
                ``generate``; if ``None``, calls
                :py:meth:`self.send_message_to_model` directly.
        """
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

    def handle_message(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Route the last message to the appropriate tool or model.

        Args:
            messages: List of message dicts with keys 'role' and 'content'.
        Returns:
            The result from the selected tool or model streaming output.
        """
        if self.args.get("wolfram_alpha", False):
            request_type = {"category": "math"}
        elif self.args.get("url", None):
            request_type = {"category": "other"}
            contents = get_webpage_contents(self.args["url"])
            messages[-1]["content"] += f": {contents}"
        else:
            request_type = self.get_request_type(messages[-1]["content"])

        category = request_type["category"]
        content = messages[-1]["content"]
        handlers = {
            "lights": lambda: control_lights(self.model_name, content),
            "music": lambda: play_music(self.model_name, content),
            "weather": lambda: get_weather_info(self.model_name, content),
            "calendar": lambda: get_schedule(self.model_name, content),
            "agenda": self.get_agenda,
            "math": lambda: WolframAlphaFunctionCall(self.model_name).run(content)
            if not self.args.get("enable_thinking", False) else None
        }

        result = handlers.get(category)
        if result:
            output = result()
            if output is not None:
                return output

        return self.send_message_to_model(messages, replace_context=True)

    def send_message_to_model(self,
                              messages: Union[str, List[Dict[str, Any]]],
                              args: Optional[Dict[str, Any]] = None,
                              prune: bool = True,
                              replace_context: bool = False,
                              tool_name: Optional[str] = None,
                              tool_list: Optional[List[str]] = None) -> Iterator[str]:
        """
        Send messages to the configured model or tool, updating the conversation context.

        Args:
            messages: A string or a list of message dicts (each with 'role' and 'content').
            args: Optional dict of additional parameters for the model call.
            prune: Whether to prune old messages from context before adding new ones.
            replace_context: Whether to replace the entire context with these messages.
            tool_name: Name of a specific tool to invoke for a local LLM, if applicable.
            tool_list: Optional list of tools available for local LLM invocation.

        Returns:
            An iterator yielding streamed response chunks from the selected model or tool.
        """
        args = args or {}
        if not isinstance(messages, list):
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

    def send_message_to_model_openai(self, args: Dict[str, Any]) -> Iterator[str]:
        """
        Send the current conversation context to OpenAI's ChatCompletion API and stream the response.

        Args:
            args: Additional keyword arguments for openai.ChatCompletion.create (e.g., temperature).
        Yields:
            Streamed content chunks from the OpenAI API response.
        """
        openai.api_key = settings.openai_api_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.context.get(),
            stream=True,
            **args
        )
        for chunk in response:
            content = chunk["choices"][0]["delta"].get("content", "")
            yield content

    def send_message_to_model_anthropic(self, args: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Sends a message to an Anthropic language model and yields streamed response chunks.

        This method prepares a message list according to Anthropic's API requirements,
        removing unsupported attributes and separating out the system prompt. It then
        sends the request with streaming enabled and yields the text content of each
        streamed chunk as it arrives.

        Args:
            args: Additional keyword arguments to be passed to the Anthropic `messages.create()` method.

        Yields:
            The text content of each streamed response chunk from the Anthropic model.
        """
        messages = self.context.get()

        # Anthropic will reject messages with extraneous attributes
        for x in messages:
            x.pop("id", None)

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

    def send_message_to_model_local_llm(
        self,
        args: Dict[str, Any],
        tool_name: Optional[str],
        tool_list: Optional[List[str]]
    ) -> Generator[str, None, None]:
        """
        Sends a request to a locally hosted LLM API endpoint and yields streamed response chunks.

        Constructs a request payload using the current context and additional parameters,
        sends it to the local model's `/chat` endpoint, and streams the response back.
        The full decoded content is also appended to the conversation context.

        Args:
            args: Additional arguments to be merged into the request JSON.
            tool_name: The name of the tool to include in the payload.
            tool_list: The tool's function list or identifier string.

        Yields:
            The text content of each streamed response chunk as UTF-8 decoded strings.
        """
        request = {
            "mode": "instruct",
            "messages": self.context.get(),
            "tool_name": tool_name,
            "tool_list": tool_list,
            "temperature": self.temperature,
            "enable_thinking": self.args.get("enable_thinking", False),
            **args
        }

        endpoints = ChatBot.get_api_endpoints()

        response = requests.post(
            endpoints["CHAT"],
            json=request,
            stream=True,
            timeout=20
        )
        content = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # Filter out keep-alive new chunks
                content += chunk.decode("utf-8")
                yield chunk.decode("utf-8")
        self.context.add(content, role="assistant")

    def get_agenda(self) -> str:
        """
        Retrieves a combined daily agenda consisting of weather and calendar information.

        This method sends queries to obtain the current weather and the day's calendar schedule
        using the model associated with this instance. It processes the responses using
        ChatBot's streaming message handler and combines them into a single formatted string.

        Returns:
            A string containing the weather information followed by the calendar schedule,
            separated by two newlines.
        """
        response = get_weather_info(self.model_name, "What's the weather today?")
        weather_content = ChatBot.get_streaming_message(response)

        response = get_schedule(self.model_name, "What's on my calendar today?")
        calendar_content = ChatBot.get_streaming_message(response)

        return f"{weather_content}\n\n{calendar_content}"

    def get_request_type(self, message: str) -> Dict[str, Any]:
        """
        Classifies a user instruction into a predefined request type category.

        This method constructs a prompt to classify the given instruction into one of several
        categories such as "music", "lights", "weather", "calendar", "agenda", "math", or "other".
        It sends the prompt to a chatbot and expects a single-line JSON response with a "category" field.

        Args:
            message: The user's instruction to classify.

        Returns:
            A dictionary with a single key "category" indicating the classified request type.
        """
        prompt = """
        I want you to put this instruction into one of multiple categories. If the instruction is to play some music, the category is "music". If the instruction is to control lights, the category is "lights". If the instruction is asking about the weather or the moon's phase, the category is "weather". If the instruction is asking about today's calendar, or is something like 'What's happening today' or 'What is my schedule', the category is "calendar". If the instruction is asking about today's agenda, or something like 'What's my update?', the category is "agenda". If the instruction is asking for mathematical calculation, the category is "math". For everything else, the category is "other". Give me the category in JSON format with the field name "category". Do not format the JSON by including newlines. Give only the JSON and no additional characters, text, or comments. Here is the instruction:
        """
        prompt += message

        chatbot = ChatBot(self.model_name, temperature=0.1)
        response = chatbot.send_message_to_model(prompt)
        content = ChatBot.get_streaming_message(response)

        if settings.debug:
            print(f"{content=}")

        response_json = None
        try:
            response_json = json.loads(strip_code_fences(content))
        except ValueError as e:
            print(f"Content generating invalid JSON: {content}")
            raise ValueError("Request type response is not proper JSON.") from e

        return response_json

    @staticmethod
    def get_streaming_message(streamer: Iterable[str]) -> str:
        """
        Joins and returns a complete string from a stream of text chunks.

        Args:
            streamer: An iterable of string chunks (e.g., from a streaming LLM response).

        Returns:
            A single concatenated string formed by joining all elements of the stream.
        """
        return "".join(streamer)

    @staticmethod
    def get_model_attribute(model_name: Optional[str], attribute: str) -> Optional[Any]:
        """
        Retrieves a specific attribute for a given model from the model_info dictionary.

        Args:
            model_name: The name of the model to look up.
            attribute: The attribute key to retrieve for the given model.

        Returns:
            The value of the attribute if it exists, otherwise None.
        """
        if model_name and \
           model_name in model_info and \
           attribute in model_info[model_name]:
            return model_info[model_name][attribute]
        return None

    @staticmethod
    def get_model_info() -> str:
        """
        Retrieves the current model name from the model info API.

        Sends a GET request to the local `/model_info` endpoint and extracts
        the model name from the JSON response.

        Returns:
            A string representing the current model's name.
        """
        endpoints = ChatBot.get_api_endpoints()
        response = requests.get(endpoints["MODEL_INFO"], timeout=10)
        return response.json()["model_name"]

    @staticmethod
    def get_model_list() -> List[Dict[str, Any]]:
        """
        Retrieves and returns a sorted list of available models, including both local and API-based ones.

        Fetches the model list from the server endpoint, appends additional models defined via API config,
        transforms them into a standardized list of dictionaries, and returns the sorted result.

        Returns:
            A sorted list of dictionaries, where each dictionary contains metadata about a model,
            such as "model", "name", "type", and optional "qwen_vision".
        """
        endpoints = ChatBot.get_api_endpoints()
        response = requests.get(endpoints["MODEL_LIST"], timeout=10)

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
    def get_personal_model_names(model_list: List[str]) -> List[Dict[str, Any]]:
        """
        Maps a list of model names to detailed model metadata dictionaries.

        For each model in the input list, looks up details in the `model_info` dictionary
        and constructs a standardized representation.

        Args:
            model_list: A list of model identifier strings.

        Returns:
            A list of dictionaries, each containing keys:
                - "model": the model identifier
                - "name": the display name
                - "type": the model's type (e.g., "local", "api")
                - "qwen_vision": vision support flag (optional)
        """
        models = [
            {
                "model": x,
                "name": model_info.get(x, {"name": x}).get("name", x),
                "type": model_info.get(x, {"type": x}).get("type", None),
                "qwen_vision": model_info.get(x, {"qwen_vision": x}).get("qwen_vision", None),
            }
            for x in
            model_list
        ]
        return models

    @staticmethod
    def load_model(model: str) -> Dict[str, Any]:
        """
        Loads the specified model if it is not already active.

        Compares the requested model name to the currently active model.
        If they differ, sends a request to the backend to load the specified model.
        Otherwise, returns a success status without reloading.

        Args:
            model: The name of the model to load.

        Returns:
            A dictionary representing the JSON response from the backend.
            If the model is already loaded, returns {"status": "OK"}.
        """
        current_model = ChatBot.get_model_info()
        if current_model == model:
            return {"status": "OK"}
        endpoints = ChatBot.get_api_endpoints()
        return requests.post(
            endpoints["MODEL_LOAD"],
            json={"model_name": model},
            timeout=120
        ).json()


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
    config = parser.parse_args()
    arg_assistant = config.assistant
    arg_mode = config.mode
    arg_tts = config.tts
    arg_stt = config.stt

    if arg_mode == "interactive":
        chatbot = ChatBot(assistant=arg_assistant, debug=config.debug, stt=arg_stt, tts=arg_tts)
        chatbot.interactive()
    elif arg_mode == "chatgpt":
        from modules.discord_bot import DiscordBot
        bot = DiscordBot(model_name="gpt-4o-mini")
        bot.run_bot()
    elif arg_mode == "localllm":
        from modules.discord_bot import DiscordBot
        bot = DiscordBot()
        bot.run_bot()

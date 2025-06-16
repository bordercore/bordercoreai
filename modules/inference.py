"""
This module defines an `Inference` class that loads, configures, and interacts with
language and vision models (e.g., Qwen, LLaMA, Gemma). It supports quantization,
custom templates, image-based prompts, and streaming text generation.
The module can be executed as a script to run inference interactively or with an image.
"""

import argparse
import base64
import importlib
import json
import os
from pathlib import Path
from threading import Thread
from typing import Any, Generator, Optional

import torch
import transformers
from qwen_vl_utils import process_vision_info

try:
    from awq import AutoAWQForCausalLM
except ModuleNotFoundError:
    # Useful during testing the webapp, which does not have the awq package
    pass
from api import settings
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration,
                          TextIteratorStreamer, pipeline)

from modules.context import Context
from modules.util import get_model_info

# This stifles the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"


class Inference:
    """Encapsulates model loading, image handling, prompt templating, and inference."""

    max_new_tokens = 4096
    temperature_default = 0.7
    top_p = 0.95
    top_k = 40

    def __init__(
        self,
        model_path: str,
        temperature: Optional[float] = None,
        quantize: bool = False,
        tool_name: Optional[str] = None,
        tool_list: Optional[str] = None,
        enable_thinking: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initialize the Inference class with model configuration and options.

        Args:
            model_path: Path to the model directory.
            temperature: Sampling temperature for generation.
            quantize: Whether to use 4-bit quantization.
            tool_name: Optional tool module to invoke.
            tool_list: Function name to use from the tool module.
            enable_thinking: Whether to enable tool reasoning mode.
            debug: Enable debug mode for verbose output.
        """
        self.context = Context()
        self.model_path = model_path
        self.model_name = Path(model_path).parts[-1]
        self.quantize = quantize
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default
        self.tokenizer = self.get_tokenizer()
        self.tool_name = tool_name
        self.tool_list = tool_list
        self.enable_thinking = enable_thinking
        self.debug = debug
        self.model: Optional[Any] = None

    def prepare_image(self, image_path: str) -> list[dict[str, Any]]:
        """
        Read and encode an image into a base64 string and format it into a prompt.

        Args:
            image_path: Path to the image file.

        Returns:
            A list containing a single message dict formatted for vision models.
        """

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"The file {image_path} does not exist.")

        # Read the image file and convert to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{encoded_string}",
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]

        return messages

    def get_template_type(self) -> str:
        """
        Determine which chat template type to use for message formatting.

        Returns:
            A string indicating the template type, such as 'llama2' or 'chatml'.
        """
        if self.model_name in self.model_info and "template" in self.model_info[self.model_name]:
            return self.model_info[self.model_name]["template"]

        print("No chat template found in models.yaml. Using llama2.")
        return "llama2"

    def get_prompt_template(self, tokenizer: Any, messages: list[dict[str, Any]]) -> str:
        """
        Construct a text prompt based on input messages using a chat template.

        Args:
            tokenizer: The tokenizer with optional chat_template support.
            messages: A list of role-based chat message dictionaries.

        Returns:
            A formatted prompt string suitable for model inference.
        """
        if hasattr(tokenizer, "chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.get_tools(),
                enable_thinking=self.enable_thinking
            )

        template_type = self.get_template_type()
        if template_type == "chatml":
            prompt_template = """
        <|im_start|>system
        {settings.system_message}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
                """
            return prompt_template.format(system_message="", prompt=messages[0]["content"])

        print("Warning: no chat template found. Using llama2.")

        template = ""
        for message in messages:
            if message["role"] == "system":
                continue
            if message["role"] == "user":
                template += f"[INST]{message['content']}[/INST]"
            elif message["role"] == "assistant":
                template += f"{message['content']}</s>"
        return template

    def get_model_config(self) -> dict[str, Any]:
        """
        Load the model's configuration from its config.json file.

        Returns:
            A dictionary containing model configuration parameters.
        """
        config_file = f"{self.model_path}/config.json"

        if not os.path.isdir(self.model_path) or not os.path.isfile(config_file):
            return {}

        with open(config_file, "r", encoding="utf-8") as file:
            model_config = json.load(file)
        return model_config

    def get_config_option(self, name: str, default: Any = None) -> Any:
        """
        Retrieve a model configuration value from the model_info dictionary.

        Args:
            name: The key of the configuration option.
            default: The fallback value if the key is not present.

        Returns:
            The value of the configuration option or the default.
        """
        if name in self.model_info[self.model_name]:
            return self.model_info[self.model_name][name]
        return default

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Build and return a quantization configuration if enabled or required.

        Returns:
            A BitsAndBytesConfig object for 4-bit quantization or None.
        """
        if self.quantize or self.model_info[self.model_name].get("quantize", None):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        return None

    def get_tools(self) -> Optional[list[Any]]:
        """
        Dynamically import and return callable tool functions if configured.

        Returns:
            A list containing the tool function or None if not specified.
        """
        if self.tool_name and self.tool_list:
            main_module = importlib.import_module(f"modules.{self.tool_name}")
            func = getattr(main_module, self.tool_list)
            return [func]
        return None

    def get_tokenizer(self) -> Any:
        """
        Load and return the tokenizer or processor for the specified model.

        Returns:
            A tokenizer or processor instance compatible with the model.
        """
        if self.get_config_option("qwen_vision", False):
            tokenizer = AutoProcessor.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            tokenizer.padding_side = "right"

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Required for the unsloth_gemma-2-2b-it-bnb-4bit model
        if "unsloth_gemma-2-2b-it-bnb-4bit" in self.model_path:
            tokenizer.add_special_tokens({"eos_token": "<end_of_turn>"})

        return tokenizer

    def load_model(self) -> None:
        """
        Load the model into memory based on its type and configuration.
        """
        model_config = self.get_model_config()
        args = {
            "device_map": {"": 0},
            "trust_remote_code": True
        }

        if "quantization_config" not in model_config:
            args["quantization_config"] = self.get_quantization_config()

        if settings.use_flash_attention:
            args["attn_implementation"] = "flash_attention_2"

        if self.get_config_option("qwen_vision", False):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **args
            )
        elif "awq" in self.model_name.lower():
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                **args,
                fuse_layers=self.get_config_option("fuse_layers", True),
                safetensors=True,
                batch_size=1,
                max_memory={0: "8000MiB", "cpu": "99GiB"}
            ).model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **args
            )

    def generate(self, messages: list[dict[str, Any]]) -> Generator[str, None, None]:
        """
        Generate a streaming text response from the model given input messages.

        Args:
            messages: A list of chat-style messages, each with a role and content.

        Yields:
            Segments of the model's generated response as strings.
        """
        if not self.tool_name:
            try:
                index = next(i for i, item in enumerate(messages) if item["role"] == "system")
                messages[index]["content"] = settings.system_message
            except StopIteration:
                messages.insert(0, {"role": "system", "content": settings.system_message})

        if "gemma" in self.model_name.lower():
            # Gemma models don't support the system role
            messages = [x for x in messages if x["role"] != "system"]

        prompt_template = self.get_prompt_template(self.tokenizer, messages)

        if self.get_config_option("add_bos_token"):
            prompt_template = f"<|begin_of_text|>{prompt_template}"

        args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.get_config_option("do_sample", True),
        }

        if args["do_sample"]:
            args["temperature"] = self.temperature
            args["top_p"] = self.top_p
            args["top_k"] = self.top_k

        if not self.tool_name and "llama" in self.model_name.lower():
            args["eos_token_id"] = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
            ]
        # If we're using tools, we don't want to skip special tokens
        #  in the response.
        skip_special_tokens = self.tool_name is None

        if self.get_config_option("qwen_vision", False):
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.tokenizer(
                text=[prompt_template],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            if self.model is None:
                raise RuntimeError("Model must be loaded before generating output.")

            # Qwen2 Vision doesn't yet support pipeline streaming
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            yield from output_text
            return

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=skip_special_tokens
        )
        generator = pipeline("text-generation", streamer=streamer, **args)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "return_full_text": False
        }
        thread = Thread(
            target=generator,
            args=(prompt_template,),
            kwargs=generation_kwargs
        )
        thread.start()

        yield from streamer
        thread.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model-path",
        help="The path to the model directory"
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Quantize the target model on-the-fly",
        action="store_true"
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
    parser.add_argument(
        "-i",
        "--image",
        help="The image to identify by a vision model",
    )

    config = parser.parse_args()
    arg_model_path = config.model_path
    arg_quantize = config.quantize
    tts = config.tts
    stt = config.stt
    image = config.image

    inference = Inference(
        model_path=arg_model_path,
        quantize=arg_quantize,
    )
    inference.load_model()

    from modules.chatbot import ChatBot
    chatbot = ChatBot(stt=stt, tts=tts)

    if image:
        image_messages = inference.prepare_image(image)
        inference.context.add(image_messages, True)
        response = inference.generate(inference.context.get())
        for chunk in response:
            print(chunk, end="", flush=True)
        print()
    else:
        chatbot.interactive(inference=inference)

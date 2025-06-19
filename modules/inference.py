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
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Generator, List, Optional

import torch
import transformers
from api import settings
from qwen_vl_utils import process_vision_info
from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig,
                          Qwen2_5_VLForConditionalGeneration,
                          TextIteratorStreamer, pipeline)

from modules.context import Context
from modules.util import get_model_info

# Suppress the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"


class Inference:
    """
    Encapsulates model loading, prompt construction, and text generation for
    various language and vision models.

    Attributes:
        max_new_tokens (int): The maximum number of tokens to generate.
        temperature_default (float): The default sampling temperature.
        top_p (float): The nucleus sampling probability.
        top_k (int): The number of top tokens to consider for sampling.
    """

    max_new_tokens: int = 4096
    temperature_default: float = 0.7
    top_p: float = 0.95
    top_k: int = 40

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
        Initializes the Inference class.

        Args:
            model_path: Path to the model directory.
            temperature: Sampling temperature for generation.
            quantize: If True, applies 4-bit quantization to the model.
            tool_name: The name of the tool module to use.
            tool_list: The function name to use from the tool module.
            enable_thinking: If True, enables tool reasoning mode.
            debug: If True, enables verbose debug output.
        """
        self.model_path = model_path
        self.model_name = Path(model_path).parts[-1]
        self.quantize = quantize
        self.debug = debug

        self.context = Context()
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default

        self.tool_name = tool_name
        self.tool_list = tool_list
        self.enable_thinking = enable_thinking
        self.tools = self.load_tools()

        self.tokenizer = self.load_tokenizer()
        self.model: Optional[Any] = None

    def load_model(self) -> None:
        """
        Load the appropriate language or vision-language model into memory.

        This method determines the type of model to load based on the configuration
        and model name, then initializes it using the Hugging Face `from_pretrained()`
        interface. It supports standard causal language models, Qwen2-VL vision models,
        and 4-bit AWQ quantized models.

        Behavior:
          - For Qwen2-VL models: uses `Qwen2_5_VLForConditionalGeneration`.
          - For models containing 'awq' in their name: dynamically imports and loads
            `AutoAWQForCausalLM` with quantization-specific arguments.
          - For all other models: uses `AutoModelForCausalLM`.
        """
        model_config_args = self.get_model_loading_args()

        if self._is_vision_model():
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, **model_config_args
            )
        elif "awq" in self.model_name.lower():
            # Dynamically import for AWQ models to avoid hard dependency
            try:
                from awq import AutoAWQForCausalLM
            except ImportError as e:
                raise ImportError(
                    "The 'awq' package is required for this model but is not installed."
                ) from e

            awq_args = {
                "fuse_layers": self.get_config_option("fuse_layers", True),
                "safetensors": True,
                "batch_size": 1,
                "max_memory": {0: "8000MiB", "cpu": "99GiB"},
            }
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path, **awq_args, **model_config_args
            ).model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **model_config_args
            )

    def generate(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        """
        Generates a streaming text response from the model.

        This function orchestrates the generation process by preparing messages,
        applying the correct chat template, and dispatching to the appropriate
        model-specific generation method.

        Args:
            messages: A list of chat messages, each with a 'role' and 'content'.

        Yields:
            A generator that produces chunks of the response text.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before calling generate().")

        prepared_messages = self.prepare_messages_for_generation(messages)
        prompt = self.apply_chat_template(prepared_messages)

        if self._is_vision_model():
            yield from self.generate_with_vision_model(prompt, prepared_messages)
        else:
            yield from self.generate_with_text_model(prompt)

    def load_tokenizer(self) -> Any:
        """
        Load and return the appropriate tokenizer or processor for the model.

        This method selects either an `AutoTokenizer` or an `AutoProcessor` depending
        on whether the model supports vision inputs (e.g., Qwen-VL). It applies any
        model-specific tokenizer adjustments, such as setting padding behavior or
        adding special tokens required by certain model variants.

        Returns:
            A tokenizer or processor instance compatible with the target model.
        """
        if self._is_vision_model():
            processor = AutoProcessor.from_pretrained(self.model_path)
            return processor

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Model-specific tokenizer adjustments
        if "unsloth_gemma-2-2b-it-bnb-4bit" in self.model_path:
            tokenizer.add_special_tokens({"eos_token": "<end_of_turn>"})

        return tokenizer

    def load_tools(self) -> Optional[List[Any]]:
        """
        Dynamically import a callable tool function from a specified module.

        This method attempts to import a function (or callable) named by `self.tool_list`
        from a module path `modules.<tool_name>`. If successful, the function is returned
        as a single-item list to be used as a tool by the model's chat template or agent.

        If either `tool_name` or `tool_list` is not set, or if the import fails, the method
        returns None and logs a warning.

        Returns:
            A list containing the imported function if successful, or None otherwise.
        """
        if self.tool_name and self.tool_list:
            try:
                module = importlib.import_module(f"modules.{self.tool_name}")
                func = getattr(module, self.tool_list)
                return [func]
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not load tool '{self.tool_list}' from '{self.tool_name}': {e}")
        return None

    def get_model_loading_args(self) -> Dict[str, Any]:
        """
        Build the keyword arguments used to load the model from disk.

        This method assembles configuration flags required by the `from_pretrained()`
        method for loading Hugging Face-compatible models. It includes:
          - CUDA device mapping
          - Trust flag for loading custom/model-specific code
          - Optional quantization configuration (if not already present in config.json)
          - Flash attention support if enabled via settings

        Returns:
            A dictionary of keyword arguments to pass to the model loader.
        """
        args = {"device_map": {"": 0}, "trust_remote_code": True}

        model_config = self._get_model_config_from_file()
        if "quantization_config" not in model_config:
            args["quantization_config"] = self.get_quantization_config()

        if settings.use_flash_attention:
            args["attn_implementation"] = "flash_attention_2"

        return args

    def get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Return a 4-bit quantization configuration if enabled for this model.

        This method checks whether quantization is either explicitly requested via
        the constructor (`self.quantize`) or specified in the model's configuration
        metadata (under the `quantize` key). If so, it returns a BitsAndBytesConfig
        object suitable for loading the model in 4-bit NF4 format.

        Returns:
            A configured BitsAndBytesConfig object for 4-bit quantization, or None
            if quantization is not requested.
        """
        if self.quantize or self.get_config_option("quantize", False):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        return None

    def prepare_messages_for_generation(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare and normalize chat messages prior to prompt generation.

        This method performs several preprocessing steps:
          - Creates a shallow copy of the input to avoid mutating the original list.
          - Ensures a system message is included if tools are not being used.
          - Replaces or inserts the system message using the value from settings.
          - Removes the system message entirely for models (e.g. Gemma) that do not support it.

        Args:
            messages: A list of dictionaries representing role-based chat messages.

        Returns:
            A new list of messages, adjusted to meet the input requirements of the current model.
        """
        # Make a copy to avoid modifying the original list
        processed_messages = list(messages)

        # Inject the system message if tools are not being used
        if not self.tools:
            try:
                # Find and update an existing system message
                sys_msg_index = next(
                    i for i, item in enumerate(processed_messages) if item["role"] == "system"
                )
                processed_messages[sys_msg_index]["content"] = settings.system_message
            except StopIteration:
                # Or insert a new one at the beginning
                processed_messages.insert(
                    0, {"role": "system", "content": settings.system_message}
                )

        # Handle model-specific requirements
        if "gemma" in self.model_name.lower():
            # Gemma models do not support the "system" role
            processed_messages = [m for m in processed_messages if m["role"] != "system"]

        return processed_messages

    def apply_chat_template(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert a sequence of chat messages into a single prompt string.

        This method first checks whether the tokenizer has a built-in `chat_template`
        attribute. If available, that is used to format the messages appropriately.
        Otherwise, a fallback manual template is constructed based on the model's
        template type, such as 'chatml' or 'llama2'.

        Args:
            messages: A list of role-based chat messages, each a dictionary with
                      'role' and 'content' keys.

        Returns:
            str: A single string prompt suitable for passing to the model.
        """
        # Prefer the tokenizer's built-in template
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.tools,
                enable_thinking=self.enable_thinking,
            )
        else:
            # Fallback to manual templating
            template_type = self.get_config_option("template", "llama2")
            print(f"Warning: Tokenizer has no chat_template. Falling back to '{template_type}'.")
            if template_type == "chatml":
                prompt_template = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
                system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
                prompt = prompt_template.format(system=system_content, user=user_content)
            else:  # Default to LLaMA2-style template
                template = ""
                for msg in messages:
                    if msg["role"] == "user":
                        template += f"[INST]{msg['content']}[/INST]"
                    elif msg["role"] == "assistant":
                        template += f"{msg['content']}</s>"
                prompt = template

        # Add beginning-of-sequence token if required by the model
        if self.get_config_option("add_bos_token"):
            prompt = f"<|begin_of_text|>{prompt}"

        return prompt

    def generate_with_text_model(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate text using a standard causal language model (e.g., LLaMA, Mistral).

        This method sets up a streaming text-generation pipeline using the Hugging Face
        `pipeline()` API. It applies appropriate sampling settings, handles special token
        behavior, and runs generation in a background thread to support non-blocking output.

        Args:
            prompt: The textual prompt to feed into the model.

        Yields:
            Segments of the model's decoded response as strings, streamed in real time.
        """
        # If we're using tools, we don't want to skip special tokens in the response.
        skip_special_tokens = self.tools is None

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=skip_special_tokens
        )

        pipeline_args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.get_config_option("do_sample", True),
            "streamer": streamer,
        }

        if pipeline_args["do_sample"]:
            pipeline_args.update({
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
            })

        if not self.tools and "llama" in self.model_name.lower():
            pipeline_args["eos_token_id"] = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Llama3 EOT token
            ]

        generator = pipeline("text-generation", **pipeline_args)

        # Run generation in a separate thread to enable streaming
        thread = Thread(
            target=generator,
            args=(prompt,),
            kwargs={"max_new_tokens": self.max_new_tokens, "return_full_text": False},
        )
        thread.start()

        yield from streamer
        thread.join()

    def generate_with_vision_model(
        self, prompt: str, messages: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """
        Generate text using a vision-language model (e.g., Qwen2-VL).

        This method processes the input `messages` for image and video content,
        tokenizes them along with the prompt, and performs generation using
        the vision-capable model. Unlike text-only models, this bypasses the
        streaming pipeline due to current model limitations.

        Args:
            prompt: A text prompt constructed from the message context.
            messages: A list of chat message dicts, potentially containing vision inputs.

        Yields:
            Segments of the decoded text output as strings.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before generation.")

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.tokenizer(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Qwen2 Vision does not yet support the pipeline streamer
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Trim the input token IDs from the generated output
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.tokenizer.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        yield from output_text

    def prepare_image_prompt(self, image_path: str, text: str) -> List[Dict[str, Any]]:
        """
        Creates a vision model prompt from an image file and text.

        Args:
            image_path: The path to the image file.
            text: The text to accompany the image.

        Returns:
            A list containing a single message dictionary formatted for vision models.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"The image file was not found at: {image_path}")

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/jpeg;base64,{encoded_string}"},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def get_config_option(self, name: str, default: Any = None) -> Any:
        """
        Retrieves a model configuration value from the model_info dictionary.

        Args:
            name: The key of the configuration option.
            default: The fallback value if the key is not present.

        Returns:
            The value of the configuration option or the specified default.
        """
        return self.model_info.get(self.model_name, {}).get(name, default)

    def _is_vision_model(self) -> bool:
        """
        Determine whether the currently-selected model supports vision inputs.

        The check simply forwards to :py:meth:`get_config_option`, expecting the
        metadata key ``"qwen_vision"`` to be ``True`` for Qwen-VL and other
        vision-language variants.

        Returns:
            bool: ``True`` if the model can accept image/video content,
                  otherwise ``False``.
        """
        return self.get_config_option("qwen_vision", False)

    def _get_model_config_from_file(self) -> Dict[str, Any]:
        """
        Load model configuration from the model's `config.json` file.

        This method attempts to read the configuration JSON file located in the model
        directory specified by `self.model_path`. If the file does not exist, an empty
        dictionary is returned.

        Returns:
            dict: A dictionary representing the model's configuration, or an empty
            dict if the file is missing.
        """
        config_path = Path(self.model_path) / "config.json"
        if not config_path.is_file():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


def main() -> None:
    """
    Parse command-line arguments and run the inference engine.

    This function initializes the inference engine with the provided model path and
    configuration flags, loads the model, and runs either image-based inference
    or an interactive chatbot loop depending on the arguments.

    Command-line arguments:
        -m, --model-path   : Path to the model directory (required).
        -q, --quantize     : Enable 4-bit quantization.
        -i, --image        : Path to an image for vision-based prompting.
        --tts              : Enable text-to-speech output.
        --stt              : Enable speech-to-text input.
    """
    parser = argparse.ArgumentParser(description="Run inference with a specified model.")
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="The path to the model directory."
    )
    parser.add_argument(
        "-q",
        "--quantize",
        action="store_true",
        help="Quantize the model on-the-fly."
    )
    parser.add_argument(
        "-i",
        "--image",
        help="Path to an image for vision model inference."
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable Text-to-Speech (TTS)."
    )
    parser.add_argument(
        "--stt",
        action="store_true",
        help="Enable Speech-to-Text (STT)."
    )

    args = parser.parse_args()

    try:
        # Initialize the core inference engine
        inference = Inference(
            model_path=args.model_path,
            quantize=args.quantize,
        )
        print("Loading model...")
        inference.load_model()
        print("Model loaded successfully.")

        if args.image:
            # Handle image-based inference
            prompt_text = "Describe this image in detail."
            image_messages = inference.prepare_image_prompt(args.image, prompt_text)
            inference.context.add(image_messages, is_vision_prompt=True)

            print(f"{COLOR_GREEN}You: {prompt_text}{COLOR_RESET}")
            print(f"\n{COLOR_BLUE}AI: ", end="")

            response_generator = inference.generate(inference.context.get())
            for chunk in response_generator:
                print(chunk, end="", flush=True)
            print()
        else:
            # Enter interactive chat mode
            from modules.chatbot import ChatBot

            chatbot = ChatBot(stt=args.stt, tts=args.tts)
            chatbot.interactive(inference=inference)

    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()

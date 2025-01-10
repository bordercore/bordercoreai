import argparse
import base64
import importlib
import json
import os
from pathlib import Path
from threading import Thread

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
                          BitsAndBytesConfig, Qwen2VLForConditionalGeneration,
                          TextIteratorStreamer, pipeline)

from modules.context import Context
from modules.util import get_model_info

# This stifles the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"


class Inference:

    max_new_tokens = 4096
    temperature_default = 0.7
    top_p = 0.95
    top_k = 40

    def __init__(self, model_path, temperature=None, quantize=False, tool_name=None, tool_list=None, debug=False):
        self.context = Context()
        self.model_path = model_path
        self.model_name = Path(model_path).parts[-1]
        self.quantize = quantize
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default
        self.tokenizer = self.get_tokenizer()
        self.tool_name = tool_name
        self.tool_list = tool_list
        self.debug = debug

    def prepare_image(self, image_path):

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

    def get_template_type(self):
        if self.model_name in self.model_info and "template" in self.model_info[self.model_name]:
            return self.model_info[self.model_name]["template"]

        print("No chat template found in models.yaml. Using llama2.")
        return "llama2"

    def get_prompt_template(self, tokenizer, messages):
        if hasattr(tokenizer, "chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=self.get_tools()
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

    def get_model_config(self):
        config_file = f"{self.model_path}/config.json"

        if not os.path.isdir(self.model_path) or not os.path.isfile(config_file):
            return {}

        with open(config_file, "r") as file:
            config = json.load(file)
        return config

    def get_config_option(self, name, default=None):
        if name in self.model_info[self.model_name]:
            return self.model_info[self.model_name][name]
        else:
            return default

    def get_quantization_config(self):
        if self.quantize or self.model_info[self.model_name].get("quantize", None):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            return None

    def get_tools(self):
        if self.tool_name:
            main_module = importlib.import_module(f"modules.{self.tool_name}")
            func = getattr(main_module, self.tool_list)
            return [func]
        else:
            return None

    def get_tokenizer(self):
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

    def load_model(self):
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
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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

    def generate(self, messages):
        # Set the system message based on the user's settings. If it already exists,
        #  override it. Otherwise add it. If tools are being used,
        #  make no changes.
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

            # Qwen2 Vision doesn't yet support pipeline streaming
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for x in output_text:
                yield x
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

        for x in streamer:
            yield x
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

    args = parser.parse_args()
    model_path = args.model_path
    quantize = args.quantize
    tts = args.tts
    stt = args.stt
    image = args.image

    inference = Inference(
        model_path=model_path,
        quantize=quantize,
    )
    inference.load_model()

    from modules.chatbot import ChatBot
    chatbot = ChatBot(stt=stt, tts=tts)

    if image:
        messages = inference.prepare_image(image)
        inference.context.add(messages, True)
        response = inference.generate(inference.context.get())
        for x in response:
            print(x, end="", flush=True)
        print()
    else:
        chatbot.interactive(inference=inference)

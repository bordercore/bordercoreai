import argparse
import json
import os
import time

import torch
import transformers
from awq import AutoAWQForCausalLM
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TextStreamer, pipeline)

from context import Context
from util import get_model_info, get_tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# This stifles the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

system_message = ""

COLOR_GREEN = "\033[32m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"


class Inference:

    max_new_tokens = 750
    temperature_default = 0.7
    top_p = 0.95
    top_k = 40
    models_config_path = "models.yaml"

    def __init__(self, model_dir, model_name, temperature=None, quantize=False, stream=False, debug=False):
        self.model_name = model_name
        self.model_path = f"{model_dir}/{model_name}"
        self.quantize = quantize
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default
        self.tokenizer = get_tokenizer(self.model_path)
        self.stream = stream
        self.debug = debug

    def get_template_type(self):
        if self.model_name in self.model_info and "template" in self.model_info[self.model_name]:
            return self.model_info[self.model_name]["template"]
        else:
            print("No chat template found in models.yaml. Using llama2.")
            return "llama2"

    def get_prompt_template(self, tokenizer, messages):

        # Remove any 'system' roles to avoid this error when using a llama2 template:
        #   jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...
        # Is this required for all template types?
        messages = [x for x in messages if x["role"] != "system"]

        if hasattr(tokenizer, "chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            template_type = self.get_template_type()
            if template_type == "chatml":
                prompt_template = """
            <|im_start|>system
            {system_message}<|im_end|>
            <|im_start|>user
            {prompt}<|im_end|>
            <|im_start|>assistant
                    """
                return prompt_template.format(system_message="", prompt=messages[0]["content"])
            else:
                print("Warning: no chat template found. Using llama2.")

                template = ""
                for message in messages:
                    if message["role"] == "system":
                        next
                    elif message["role"] == "user":
                        # template += f"<s>[INST]{message['content']}[/INST]"
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

    def get_quantization_config(self):
        if self.quantize:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            return None

    def load_model(self):

        model_config = self.get_model_config()
        args = {
            "device_map": {"": 0},
            "trust_remote_code": True
        }

        if "quantization_config" not in model_config:
            args["quantization_config"] = self.get_quantization_config()

        if "awq" in self.model_name.lower():
            self.model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                **args,
                fuse_layers=True,
                safetensors=True,
                max_new_tokens=4096,
                batch_size=1,
                max_memory={0: "8000MiB", "cpu": "99GiB"}
            ).model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **args
            )
        self.tokenizer = get_tokenizer(self.model_path)

    def generate(self, messages):
        prompt_template = self.get_prompt_template(self.tokenizer, messages)

        if self.model_info[self.model_name].get("add_bos_token", None):
            prompt_template = f"<|begin_of_text|>{prompt_template}"

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
        ]

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "eos_token_id": terminators
        }
        if self.stream:
            args["streamer"] = streamer
            print(f"\n{COLOR_BLUE}Assistant{COLOR_RESET}: ", end="")

        start = time.time()

        generator = pipeline("text-generation", **args)
        generation_output = generator(prompt_template, return_full_text=False)
        response = generation_output[0]["generated_text"]

        num_tokens = len(response)
        speed = int(num_tokens / (time.time() - start))

        if self.debug:
            print("\n" + response)

        return response, num_tokens, speed

    def run(self):

        print(f"Using model {self.model_name}.\n")

        self.load_model()

        context = Context()

        while True:
            user_input = input(f"\n{COLOR_GREEN}User{COLOR_RESET}: ")
            context.add("user", user_input)
            response, num_tokens, speed = self.generate(context.get())
            context.add("assistant", response)

            if not self.stream:
                print(f"\n{COLOR_BLUE}Assistant{COLOR_RESET}: " + response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--directory",
        default=".",
        help="The model directory"
    )
    parser.add_argument(
        "-m",
        "--model-name",
        help="The target model",
        default="Mistral-7B-Instruct-v0.2-finetuned"
    )
    parser.add_argument(
        "-s",
        "--stream",
        help="Stream responses from LLM",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Quantize the target model on-the-fly",
        action="store_true"
    )

    args = parser.parse_args()
    dir = args.directory
    model_name = args.model_name
    stream = args.stream
    quantize = args.quantize

    inference = Inference(
        model_dir=dir,
        model_name=model_name,
        quantize=quantize,
        stream=stream
    )
    inference.run()

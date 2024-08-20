import argparse
import json
import os
from pathlib import Path
from threading import Thread

import torch
import transformers

try:
    from awq import AutoAWQForCausalLM
except ModuleNotFoundError:
    # Useful during testing the webapp, which does not have the awq package
    pass
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TextIteratorStreamer, pipeline)

from modules.context import Context
from modules.util import get_model_info, get_tokenizer

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

    def __init__(self, model_path, temperature=None, quantize=False, stream=False, interactive=False, debug=False):
        self.model_path = model_path
        self.model_name = Path(model_path).parts[-1]
        self.quantize = quantize
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default
        self.tokenizer = get_tokenizer(self.model_path)
        self.stream = stream
        self.interactive = interactive
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
                fuse_layers=self.get_config_option("fuse_layers", True),
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

    def generate_tokens(self, streamer):
        response = ""
        for x in streamer:
            response += x
            if self.interactive:
                print(x, end="", flush=True)
            else:
                yield x

    def generate(self, messages):

        prompt_template = self.get_prompt_template(self.tokenizer, messages)

        if self.get_config_option("add_bos_token"):
            prompt_template = f"<|begin_of_text|>{prompt_template}"

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
        ]

        args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "eos_token_id": terminators,
        }
        if self.stream:
            if self.interactive:
                print(f"\n{COLOR_BLUE}Assistant{COLOR_RESET}: ", end="")

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            generator = pipeline("text-generation", streamer=streamer, **args)

            generation_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                return_full_text=False
            )
            thread = Thread(
                target=generator,
                args=(prompt_template,),
                kwargs=generation_kwargs
            )
            thread.start()

            response = ""
            for x in streamer:
                response += x
                if self.interactive:
                    print(x, end="", flush=True)
                else:
                    yield x
            thread.join()

            if self.interactive:
                print()

        else:
            generator = pipeline("text-generation", **args)
            generation_output = generator(prompt_template, return_full_text=False)
            response = generation_output[0]["generated_text"]
            print(f"\n{COLOR_BLUE}Assistant{COLOR_RESET}: " + response)

        if self.interactive:
            self.context.add(response, True, role="assistant")

    def run(self):

        print(f"Using model {self.model_name}.\n")

        self.load_model()
        self.context = Context()

        while True:
            user_input = input(f"\n{COLOR_GREEN}User{COLOR_RESET}: ")
            self.context.add(user_input, True)
            generate_tokens = self.generate(self.context.get())
            try:
                next(generate_tokens)
            except StopIteration:
                pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model-path",
        help="The path to the model directory"
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
    model_path = args.model_path
    stream = args.stream
    quantize = args.quantize

    inference = Inference(
        model_path=model_path,
        quantize=quantize,
        stream=stream,
        interactive=True
    )
    inference.run()
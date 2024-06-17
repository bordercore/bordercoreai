import argparse
import json
import os
import re
import time

import torch
import transformers
from awq import AutoAWQForCausalLM
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          GenerationConfig, TextStreamer, pipeline)

from context import Context
from util import get_model_info, get_tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# This stifles the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

# TODO: What system message to use?
system_message = ""


class Inference:

    max_new_tokens = 256
    temperature_default = 0.7
    models_config_path = "models.yaml"

    def __init__(self, model_dir, model_name, temperature=None, quantize=False, debug=False):
        self.model_name = model_name
        self.model_path = f"{model_dir}/{model_name}"
        self.quantize = quantize
        self.model_info = get_model_info()
        self.temperature = temperature or self.temperature_default
        self.tokenizer = get_tokenizer(self.model_path)
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

    def parse_response(self, response):

        if self.model_name in self.model_info:
            response_type = self.model_info[self.model_name].get("type", None)
            if response_type == "llama2":
                return self.parse_response_llama2(response)
            elif response_type == "chatml":
                return self.parse_response_chatml(response)
            elif response_type == "phi":
                return self.parse_response_phi(response)
            else:
                return response
        else:
            return response

    def parse_response_chatml(self, response):
        pattern = ".*assistant\n(.*)"
        matches = re.search(pattern, response, re.DOTALL)

        # Extracting and printing the matched content
        if matches:
            return matches.group(1).strip()
        else:
            return f"Error: not able to parse response: {response}"

    def parse_response_llama2(self, response):
        pattern = r".*\[\/INST\](.*)"
        matches = re.search(pattern, response, re.DOTALL)

        # Extracting and printing the matched content
        if matches:
            return matches.group(1).strip()
        else:
            return f"Error: not able to parse response: {response}"

    def parse_response_phi(self, response):
        pattern = r".*\n(.*)"
        matches = re.search(pattern, response, re.DOTALL)

        # Extracting and printing the matched content
        if matches:
            return matches.group(1).strip()
        else:
            return f"Error: not able to parse response: {response}"

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

        token_input = self.tokenizer(
            prompt_template,
            return_tensors="pt"
        ).input_ids.to(device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
        ]

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            top_k=40,
            eos_token_id=terminators,
            max_new_tokens=750
        )

        start = time.time()

        generation_output = self.model.generate(
            token_input,
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_config,
        )
        num_tokens = len(generation_output[0])
        speed = int(num_tokens / (time.time() - start))

        # Get the tokens from the output, decode them, print them
        token_output = generation_output[0]
        text_output = self.tokenizer.decode(token_output, skip_special_tokens=True)
        response = self.parse_response(text_output)
        if self.debug:
            print("\n" + response)
        return response, num_tokens, speed

        # streamer = TextStreamer(self.tokenizer)

        # terminators = [
        #     self.tokenizer.eos_token_id,
        #     self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
        # ]

        # pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=self.tokenizer,
        #     streamer=streamer,
        #     max_new_tokens=self.max_new_tokens,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=self.temperature,
        #     top_p=0.95,
        #     top_k=40,
        #     repetition_penalty=1.1
        # )

        # pipe(prompt_template)[0]["generated_text"]
        # # self.parse_response(pipe(prompt_template)[0]["generated_text"])

    def run(self):

        print(f"Using model {self.model_name}.\n")

        self.load_model()

        context = Context()

        while True:
            user_input = input("\nPrompt: ")
            context.add("user", user_input)
            response, num_tokens, speed = self.generate(context.get())
            context.add("assistant", response)
            print("\n" + response)


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
        "-q",
        "--quantize",
        help="Quantize the target model on-the-fly",
        action="store_true"
    )

    args = parser.parse_args()
    dir = args.directory
    model_name = args.model_name
    quantize = args.quantize

    inference = Inference(model_dir=dir, model_name=model_name, quantize=quantize)
    inference.run()

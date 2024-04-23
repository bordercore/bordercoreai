import argparse
import json
import os
import re

import torch
import transformers
import yaml
from awq import AutoAWQForCausalLM
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          TextStreamer, pipeline)

# from .util import get_tokenizer
from util import get_tokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# This stifles the "Special tokens have been added in the vocabulary..." warning
transformers.logging.set_verbosity_error()

# TODO: What system message to use?
system_message = ""


class Inference:

    max_new_tokens = 256
    temperature = 0.7
    models_config_path = "models.yaml"

    def __init__(self, dir, model_name, quantize=False):
        self.model_name = model_name
        self.model_path = f"{dir}/{model_name}"
        self.quantize = quantize

        # Load the model config file
        full_path = os.path.join(os.path.dirname(__file__), self.models_config_path)
        with open(full_path, "r") as file:
            self.model_info = yaml.safe_load(file)

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
            # elif template_type == "llama3":
            #     prompt_template = """
            #     <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            #     {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
            #     {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            #     """
            #     return prompt_template.format(system_message="", prompt=messages[0]["content"])
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
            template_type = self.get_template_type()
            if template_type == "llama2":
                return self.parse_response_llama2(response)
            elif template_type == "chatml" or template_type == "llama3":
                return self.parse_response_chatml(response)
            else:
                print("No chat template found in models.yaml. Using llama2.")
                return self.parse_response_llama2(response)
        else:
            return response

    def parse_response_chatml(self, response):
        # pattern = ".*\n assistant\n(.*)"
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
            "device_map": {"": 0}
        }

        if "quantization_config" not in model_config:
            args["quantization_config"] = self.get_quantization_config()

        if "awq" in self.model_name.lower():
            model = AutoAWQForCausalLM.from_quantized(
                self.model_path,
                **args,
                fuse_layers=True,
                safetensors=True,
                max_new_tokens=4096,
                batch_size=1,
                max_memory={0: "8000MiB", "cpu": "99GiB"}
            ).model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **args
            )

        return model

    def run(self):

        print(f"Using model {self.model_name}.\n")

        model = self.load_model()

        self.tokenizer = get_tokenizer(self.model_path)

        while True:
            user_input = input("\nPrompt: ")

            messages = [
                {"role": "user", "content": user_input}
            ]
            prompt_template = self.get_prompt_template(self.tokenizer, messages)

            # generation_config = GenerationConfig(
            #     do_sample=True,
            #     temperature=self.temperature,
            #     top_p=0.95,
            #     top_k=40,
            #     max_new_tokens=256
            # )
            # # Generate output
            # # TODO: explicitly setting pad_token_id avoids a warning.
            # #  what exactly does this do?
            # generation_output = model.generate(
            #     token_input,
            #     pad_token_id=self.tokenizer.eos_token_id,
            #     generation_config=generation_config,
            # )

            # # Get the tokens from the output, decode them, print them
            # token_output = generation_output[0]
            # text_output = self.tokenizer.decode(token_output, skip_special_tokens=True)
            # print("\n" + self.parse_response(text_output) + "\n")

            streamer = TextStreamer(self.tokenizer)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
            ]

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                streamer=streamer,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1
            )

            pipe(prompt_template)[0]["generated_text"]
            # self.parse_response(pipe(prompt_template)[0]["generated_text"])


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

    inference = Inference(dir=dir, model_name=model_name, quantize=quantize)
    inference.run()

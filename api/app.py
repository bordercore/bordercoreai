# To run
#
# $ PYTHONPATH=. FLASK_RUN_HOST=0.0.0.0 flask --app app run

import gc
import json
import os
import re

import torch
import yaml
from awq import AutoAWQForCausalLM
from flask import Flask, jsonify, request
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          GenerationConfig)

import shared

from ..util import get_tokenizer

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


with open("models.yaml", "r") as file:
    model_info = yaml.safe_load(file)


def get_model_list(path):
    """Return a list of all models within the specified path."""

    if not os.path.isdir(path):
        raise ValueError(f"The specified path: {path} is not a valid directory.")

    directories = []
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path) or full_path.endswith("gguf"):
            directories.append(item)
    return directories


def get_prompt_template(tokenizer, messages):

    # Remove any 'system' roles to avoid this error when using a llama2 template:
    #   jinja2.exceptions.TemplateError: Conversation roles must alternate user/assistant/user/assistant/...
    # Is this required for all template types?
    messages = [x for x in messages if x["role"] != "system"]

    if hasattr(tokenizer, "chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        template_type = get_template_type()
        if template_type == "chatml":

            # Use chatML
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
            # Assume llama2
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
            # prompt_template = "<s>[INST]{prompt}[/INST]{response}</s>"
            # return prompt_template.format(system_message="", prompt=messages[0]["content"])


def get_template_type():
    if shared.model_name in model_info and "template" in model_info[shared.model_name]:
        return model_info[shared.model_name]["template"]
    else:
        print("No chat template found in models.yaml. Using llama2.")
        return "llama2"


def parse_response(response):

    if shared.model_name in model_info:
        template_type = get_template_type()
        if template_type == "llama2":
            return parse_response_llama2(response)
        elif template_type == "chatml":
            return parse_response_chatml(response)
        else:
            print("No chat template found in models.yaml. Using llama2.")
            return parse_response_llama2(response)
    else:
        return response


def parse_response_chatml(response):
    # pattern = ".*\n assistant\n(.*)"
    pattern = ".*assistant\n(.*)"
    matches = re.search(pattern, response, re.DOTALL)

    # Extracting and printing the matched content
    if matches:
        return matches.group(1).strip()
    else:
        return f"Error: not able to parse response: {response}"


def parse_response_llama2(response):
    pattern = r".*\[\/INST\](.*)"
    matches = re.search(pattern, response, re.DOTALL)

    # Extracting and printing the matched content
    if matches:
        return matches.group(1).strip()
    else:
        return f"Error: not able to parse response: {response}"


def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


def get_model_config(model_path):
    config_file = f"{model_path}/config.json"

    if not os.path.isdir(model_path) or not os.path.isfile(config_file):
        return {}

    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def unload_model():

    # Remove model from VRAM
    del shared.model
    gc.collect()
    torch.cuda.empty_cache()
    shared.model = None


def load_model(model_name):
    if shared.model:
        return

    shared.model_name = model_name

    model_path = f"{shared.model_dir}/{shared.model_name}"
    model_config = get_model_config(model_path)

    args = {
        "device_map": {"": 0}
    }

    if "quantization_config" not in model_config:
        args["quantization_config"] = get_quantization_config()

    if "awq" in model_name.lower():
        shared.model = AutoAWQForCausalLM.from_quantized(
            model_path,
            **args,
            fuse_layers=True,
            safetensors=True,
            max_new_tokens=512,
            batch_size=1,
            max_memory={0: "8000MiB", "cpu": "99GiB"}
        ).model
    else:
        shared.model = AutoModelForCausalLM.from_pretrained(model_path, **args)

    shared.tokenizer = get_tokenizer(model_path)


load_model(shared.model_name)


@app.route("/v1/internal/model/info")
def info():

    return jsonify(model_name=shared.model_name)


@app.route("/v1/internal/model/list")
def list():

    return jsonify(model_names=get_model_list(shared.model_dir))


@app.route("/v1/internal/model/load", methods=["POST"])
def load():

    try:
        unload_model()
        load_model(request.json["model_name"])
        status = "OK"
        message = ""
    except Exception as e:
        status = "Error"
        message = str(e)

    return jsonify(status=status, message=message)


@app.route("/v1/chat/completions", methods=["POST"])
def main(id=None):

    payload = request.json
    # prompt_template = get_prompt_template(shared.tokenizer, [payload["messages"][-1]])
    prompt_template = get_prompt_template(shared.tokenizer, payload["messages"])

    token_input = shared.tokenizer(
        prompt_template,
        return_tensors="pt"
    ).input_ids.to(device)

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=payload["temperature"] if "temperature" in payload else shared.temperature,
        top_p=0.95,
        top_k=40,
        eos_token_id=shared.tokenizer.eos_token_id,
        max_new_tokens=750
    )

    generation_output = shared.model.generate(
        token_input,
        pad_token_id=shared.tokenizer.eos_token_id,
        generation_config=generation_config,
    )

    # Get the tokens from the output, decode them, print them
    token_output = generation_output[0]
    text_output = shared.tokenizer.decode(token_output, skip_special_tokens=True)

    print(f"{text_output=}")
    response = parse_response(text_output)
    print(f"{response=}")
    return jsonify(choices=[{"message": {"content": response}}], status="OK")

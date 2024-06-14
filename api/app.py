# To run
#
# $ PYTHONPATH=. FLASK_RUN_HOST=0.0.0.0 flask --app app run

import gc
import os
import time

import torch
from flask import Flask, jsonify, request
from transformers import GenerationConfig

import settings

from ..inference import Inference
from ..util import get_tokenizer

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def unload_model():

    # Remove model from VRAM
    del settings.model
    gc.collect()
    torch.cuda.empty_cache()
    settings.model = None


def load_model(model_name):
    if settings.model:
        return

    settings.model_name = model_name

    model_path = f"{settings.model_dir}/{settings.model_name}"

    inference = Inference(
        dir=settings.model_dir,
        model_name=settings.model_name,
        quantize=True
    )
    settings.model = inference.load_model()
    settings.tokenizer = get_tokenizer(model_path)


def get_temperature(payload):
    temp = payload["temperature"] if "temperature" in payload else settings.temperature
    # the temperature must be a strictly positive float
    if temp == 0:
        temp = 0.1
    return temp


load_model(settings.model_name)


@app.route("/v1/internal/model/info")
def info():

    return jsonify(model_name=settings.model_name)


@app.route("/v1/internal/model/list")
def list():

    return jsonify(model_names=get_model_list(settings.model_dir))


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

    inference = Inference(dir=settings.model_dir, model_name=settings.model_name)
    prompt_template = inference.get_prompt_template(
        settings.tokenizer,
        payload["messages"]
    )

    if inference.model_info[settings.model_name].get("add_bos_token", None):
        prompt_template = f"<|begin_of_text|>{prompt_template}"

    token_input = settings.tokenizer(
        prompt_template,
        return_tensors="pt"
    ).input_ids.to(device)

    terminators = [
        settings.tokenizer.eos_token_id,
        settings.tokenizer.convert_tokens_to_ids("<|eot_id|>")  # LLama3
    ]

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=get_temperature(payload),
        top_p=0.95,
        top_k=40,
        eos_token_id=terminators,
        max_new_tokens=750
    )

    start = time.time()

    generation_output = settings.model.generate(
        token_input,
        pad_token_id=settings.tokenizer.eos_token_id,
        generation_config=generation_config,
    )
    num_tokens = len(generation_output[0])
    speed = int(num_tokens / (time.time() - start))

    # Get the tokens from the output, decode them, print them
    token_output = generation_output[0]
    text_output = settings.tokenizer.decode(token_output, skip_special_tokens=True)

    print(f"{text_output=}")
    response = inference.parse_response(text_output)
    return jsonify(choices=[{"message": {"content": response, "speed": speed, "num_tokens": num_tokens}}], status="OK")

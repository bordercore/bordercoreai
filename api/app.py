import gc
import os

import torch
from flask import Flask, jsonify, request
from inference import Inference
from util import get_tokenizer

from api import settings

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
        model_dir=settings.model_dir,
        model_name=settings.model_name,
        quantize=True
    )
    inference.load_model()
    settings.model = inference.model
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

    inference = Inference(
        model_dir=settings.model_dir,
        model_name=settings.model_name,
        temperature=get_temperature(payload),
        debug=True
    )
    inference.model = settings.model
    response, num_tokens, speed = inference.generate(payload["messages"])

    return jsonify(
        choices=[
            {
                "message": {
                    "content": response,
                    "speed": speed,
                    "num_tokens": num_tokens
                }
            }
        ], status="OK"
    )

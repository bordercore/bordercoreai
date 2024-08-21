import gc
import os

import torch
from flask import Flask, jsonify, request
from modules.inference import Inference

from api import settings

app = Flask(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = None


def get_model_list(directory):
    """Return a list of all models within the specified directory."""

    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    directories = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path) or full_path.endswith("gguf"):
            directories.append(item)
    return directories


def unload_model():
    global model

    # Remove model from VRAM
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = None


def load_model(model_name):
    global model

    if model:
        return

    settings.model_name = model_name

    model_path = f"{settings.model_dir}/{settings.model_name}"

    inference = Inference(
        model_path=model_path,
        quantize=True
    )

    inference.load_model()
    model = inference.model


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

    model_path = f"{settings.model_dir}/{settings.model_name}"
    inference = Inference(
        model_path=model_path,
        temperature=get_temperature(payload),
        debug=True,
        stream=True
    )
    inference.model = model
    return inference.generate(payload["messages"])

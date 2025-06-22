"""
Flask API for Model Management and Text Generation

This module defines a lightweight Flask server that exposes several
endpoints for interacting with local language models. Key features include:

- Listing available models
- Loading and unloading models
- Reloading configuration at runtime
- Handling text generation via chat completions

Environment:
    Settings are imported from `api.settings`, which defines the model path,
    temperature, and default model name.
"""

import gc
import importlib
import os
from typing import Any, Dict, Generator, List, Mapping

import torch
from flask import Flask, Response, jsonify, request
from modules.inference import Inference

from api import settings

app = Flask(__name__)

model = None


def get_model_list(directory: str) -> List[str]:
    """
    Return a list of all models within the specified directory.

    Args:
        directory: Path to the directory containing model files or subfolders.

    Returns:
        A list of model directory names or file names ending in 'gguf'.

    Raises:
        ValueError: If the provided directory path does not exist.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"{directory} is not a valid directory.")

    directories = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isdir(full_path) or full_path.endswith("gguf"):
            directories.append(item)
    return directories


def unload_model() -> None:
    """
    Unload the current model from memory, clearing GPU VRAM and Python GC.
    """
    global model

    # Remove model from VRAM
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = None


def load_model(model_name: str) -> None:
    """
    Load the specified model into memory if not already loaded.

    Args:
        model_name: The name of the model to load.
    """
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


def get_temperature(payload: Mapping[str, Any]) -> float:
    """
    Extract the temperature value from the payload or fallback to settings.

    Args:
        payload: The incoming request JSON payload.

    Returns:
        A float > 0 representing the model sampling temperature.
    """
    temp = payload["temperature"] if "temperature" in payload else settings.temperature
    # the temperature must be a strictly positive float
    if temp == 0:
        temp = 0.1
    return temp


load_model(settings.model_name)


@app.route("/v1/internal/model/info")
def info() -> Response:
    """
    Return the name of the currently loaded model.

    Returns:
        JSON response with the model name.
    """
    return jsonify(model_name=settings.model_name)


@app.route("/v1/internal/model/list")
def list_models() -> Response:
    """
    List all available models in the configured model directory.

    Returns:
        JSON response with a list of model names.
    """

    return jsonify(model_names=get_model_list(settings.model_dir))


@app.route("/v1/internal/model/load", methods=["POST"])
def load() -> tuple[Response, int]:
    """
    Load a model as specified in the POST request body.

    Request JSON should include:
        - model_name: str

    Returns:
        JSON response with status and error message if any.
    """
    payload_raw = request.get_json(silent=True)
    if not isinstance(payload_raw, dict) or "model_name" not in payload_raw:
        return jsonify(status="Error", message="model_name required"), 400

    try:
        unload_model()
        load_model(str(payload_raw["model_name"]))
        status = "OK"
        message = ""
    except Exception as e:
        status = "Error"
        message = str(e)

    return jsonify(status=status, message=message), 200


@app.route("/reload-settings", methods=["POST"])
def force_settings_reload() -> tuple[Response, int]:
    """
    Reload application settings from `api/settings.py` at runtime.

    Returns:
        JSON response indicating success or failure.
    """
    try:
        api = importlib.import_module("api")
        importlib.reload(api.settings)
        return jsonify({"message": "Settings reloaded successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/chat/completions", methods=["POST"])
def main() -> Generator[str, None, None] | tuple[Response, int]:
    """
    Handle chat completion request by running inference on input messages.

    Request JSON should include:
        - messages: List of message dictionaries with 'role' and 'content'
        - Optional: temperature, tool_name, tool_list, enable_thinking

    Returns:
        Streaming or static response from the inference engine.
    """
    payload_raw = request.get_json(silent=True)
    if not isinstance(payload_raw, dict) or "messages" not in payload_raw:
        return jsonify(error="Invalid JSON payload"), 400
    payload: Dict[str, Any] = payload_raw  # local, typed alias

    model_path = f"{settings.model_dir}/{settings.model_name}"
    inference = Inference(
        model_path=model_path,
        temperature=get_temperature(payload),
        tool_name=payload.get("tool_name", None),
        tool_list=payload.get("tool_list", None),
        enable_thinking=payload.get("enable_thinking", False),
        debug=True,
    )
    inference.model = model
    return inference.generate(payload["messages"])

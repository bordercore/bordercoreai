import json
import warnings
from pathlib import Path

import ffmpeg
import numpy as np
import requests
import sounddevice  # Adding this eliminates an annoying warning
from flask import (Flask, Response, jsonify, render_template, request, session,
                   stream_with_context)
from flask_session import Session
from modules.chatbot import CONTROL_VALUE, ChatBot

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from api import settings
from modules.audio import Audio
from modules.rag import RAG

NUM_STARS = 10

app = Flask(__name__)
app.debug = True
app.secret_key = settings.flask_secret_key
app.config["SESSION_TYPE"] = "filesystem"

Session(app)  # Initialize session management


@app.before_request
def before_request_func():
    session["tts_host"] = settings.tts_host
    session["tts_voice"] = settings.tts_voice


@app.route("/")
def main():

    return render_template(
        "index.html",
        session=dict(session),
        settings=dict(music_uri=settings.music_uri),
        num_stars=NUM_STARS,
        control_value=CONTROL_VALUE,
        chat_endpoint="/chat"
    )


@app.route("/rag")
def rag():

    return render_template(
        "rag.html",
        session=dict(session),
        settings=dict(music_uri=settings.music_uri),
        num_stars=NUM_STARS,
        control_value=CONTROL_VALUE,
        chat_endpoint="/rag/chat"
    )


@app.route("/rag/upload", methods=["POST"])
def rag_upload():
    name = request.files["file"].filename
    text = request.files["file"].read()
    chromdb = Path(__file__).resolve().parent.parent / "chromdb"
    rag = RAG(None, chromdb=str(chromdb))
    rag.add_document(text=text, name=name)

    return jsonify(
        {
            "sha1sum": rag.get_sha1sum()
        }
    )


@app.route("/rag/chat", methods=["POST"])
def rag_chat():

    sha1sum = request.form["sha1sum"]
    message = request.form["message"]
    model_name = request.form["model"]
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))
    temperature = float(request.form.get("temperature", 0.7))

    store_params_in_session(speak, audio_speed, temperature)

    chromdb = Path(__file__).resolve().parent.parent / "chromdb"
    rag = RAG(model_name, chromdb=str(chromdb))
    try:
        rag.get_collection(sha1sum=sha1sum)
        return rag.query_document(message)
    except ValueError:
        return {
            "status": "error",
            "message": "Document not found"
        }


@app.route("/audio")
def audio():

    return render_template(
        "audio.html",
        session=dict(session),
        settings=dict(music_uri=settings.music_uri),
        num_stars=NUM_STARS,
        control_value=CONTROL_VALUE,
        chat_endpoint="/audio/chat"
    )


@app.route("/audio/upload", methods=["POST"])
def audio_upload():
    audio_data = request.files["file"].read()
    audio = Audio()
    text = audio.transcribe(audio_data=audio_data)

    return jsonify(
        {
            "text": text
        }
    )


@app.route("/audio/chat", methods=["POST"])
def audio_chat():

    message = json.loads(request.form["message"])
    transcript = request.form["transcript"]
    model_name = request.form["model"]
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))
    temperature = float(request.form.get("temperature", 0.7))

    store_params_in_session(speak, audio_speed, temperature)

    audio = Audio()
    return audio.query_transcription(model_name, message, transcript)


@app.route("/speech2text", methods=["POST"])
def speech2text():

    audio_data = request.files["audio"].read()
    audio = Audio()
    result = audio.transcribe(audio_data=load_audio(audio_data))

    return jsonify(
        {
            "input": result
        }
    )


# Register any optional Flask Blueprints
try:
    from .local.optional import optional_bp
    app.register_blueprint(optional_bp)
except ModuleNotFoundError:
    pass


def generate_stream(chatbot, message):
    try:
        yield from chatbot.handle_message(message)
    except requests.exceptions.ConnectionError:
        yield "Error connecting to API"
    except Exception as error:
        yield f"An error occurred: {error}"


@app.route("/chat", methods=["POST"])
def chat():
    message = json.loads(request.form["message"])
    model_name = request.form["model"]
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))  # Playback speed
    temperature = float(request.form.get("temperature", 0.7))
    wolfram_alpha = request.form.get("wolfram_alpha", "false").lower() == "true"
    url = request.form.get("url", None)

    store_params_in_session(speak, audio_speed, temperature)

    chatbot = ChatBot(
        model_name=model_name,
        assistant=False,
        debug=False,
        voice=False,
        speak=False,
        temperature=temperature,
        wolfram_alpha=wolfram_alpha,
        url=url
    )
    return Response(stream_with_context(generate_stream(chatbot, message)), mimetype="text/plain")


@app.route("/info")
def info():

    info = ChatBot.get_model_info()
    return jsonify(info)


@app.route("/list")
def list_models():

    model_list = ChatBot.get_model_list()
    return jsonify(model_list)


@app.route("/load", methods=["POST"])
def load():

    model = request.form["model"]
    type = ChatBot.get_model_attribute(model, "type")

    if type == "api":
        info = {"status": "OK"}
    else:
        info = ChatBot.load_model(model)

    return jsonify(info)


# Source: https://github.com/openai/whisper/discussions/380#discussioncomment-3928648
def load_audio(file: (str, bytes), sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    if isinstance(file, bytes):
        inp = file
        file = "pipe:"
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def store_params_in_session(speak, audio_speed, temperature):

    session.permanent = True
    session["speak"] = speak.lower() == "true"  # Convert "true" to True, for example
    session["audio_speed"] = audio_speed
    session["temperature"] = temperature

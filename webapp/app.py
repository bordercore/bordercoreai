import base64
import io
import json
import traceback
import warnings
import wave
from pathlib import Path

import ffmpeg
import numpy as np
import piper
import sounddevice  # Adding this eliminates an annoying warning
from chatbot import ChatBot
from flask import Flask, jsonify, render_template, request, session
from flask_session import Session

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from api import settings
from audio import Audio
from rag import RAG

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
        num_stars=NUM_STARS
    )


@app.route("/rag")
def rag():

    return render_template(
        "rag.html",
        session=dict(session),
        settings=dict(music_uri=settings.music_uri),
        num_stars=NUM_STARS
    )


@app.route("/rag/upload", methods=["POST"])
def rag_upload():
    name = request.files["file"].filename
    text = request.files["file"].read()
    chromdb = Path(__file__).resolve().parent.parent / "chromdb"
    rag = RAG(chromdb=str(chromdb), use_openai=True)
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
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))
    temperature = float(request.form.get("temperature", 0.7))
    tts = request.form.get("tts", None)

    store_params_in_session(speak, audio_speed, temperature)

    chromdb = Path(__file__).resolve().parent.parent / "chromdb"
    rag = RAG(chromdb=str(chromdb), use_openai=True)
    try:
        rag.get_collection(sha1sum=sha1sum)
        answer = rag.query_document(message)

        audio = None
        if tts != "alltalk":
            audio = generate_audio(answer, audio_speed)
        response = {
            "response": answer,
            "audio": audio
        }
    except ValueError:
        response = {
            "status": "error",
            "message": "Document not found"
        }

    return jsonify(response)


@app.route("/audio")
def audio():

    return render_template(
        "audio.html",
        session=dict(session),
        settings=dict(music_uri=settings.music_uri),
        num_stars=NUM_STARS
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

    message = request.form["message"]
    transcript = request.form["transcript"]
    model_name = request.form["model"]
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))
    temperature = float(request.form.get("temperature", 0.7))
    tts = request.form.get("tts", None)

    store_params_in_session(speak, audio_speed, temperature)

    audio = Audio()
    response = audio.query_transcription(model_name, message, transcript)

    audio_data = None
    if tts != "alltalk":
        audio_data = generate_audio(response["content"], audio_speed)

    return jsonify(
        {
            **response,
            "audio": audio_data
        }
    )


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


def generate_audio(message, audio_speed):

    audio_speed = map_speech_rate_value(audio_speed)

    voice = piper.PiperVoice.load(
        model_path="en_US-amy-medium.onnx",
        config_path="en_US-amy-medium.onnx.json"
    )

    binary_stream = io.BytesIO()

    # Use the binary stream as the destination for the WAV data
    with wave.open(binary_stream, "wb") as wav:
        voice.synthesize(message, wav, length_scale=audio_speed)

    # Get the binary data from the stream
    binary_data = binary_stream.getvalue()

    return base64.b64encode(binary_data).decode("utf-8")


def map_speech_rate_value(input_value):

    # Source range
    in_min = 0
    in_max = 2

    # Target range
    out_min = 0.5
    out_max = 1.5

    # Normalize input to a 0-1 range
    normalized_input = (input_value - in_min) / (in_max - in_min)

    # Inverse the normalized value (1 becomes 0, 0 becomes 1)
    inverted_input = 1 - normalized_input

    # Map the inverted value to the output range
    output_value = inverted_input * (out_max - out_min) + out_min

    return output_value


@app.route("/chat", methods=["POST"])
def chat():
    message = json.loads(request.form["message"])
    model_name = request.form["model"]
    speak = request.form.get("speak", "false")
    audio_speed = float(request.form.get("audio_speed", 1.0))  # Playback speed
    temperature = float(request.form.get("temperature", 0.7))
    tts = request.form.get("tts", None)

    store_params_in_session(speak, audio_speed, temperature)

    chatbot = ChatBot(
        model_name=model_name,
        assistant=False,
        debug=False,
        chat_mode="instruct",
        voice=False,
        speak=False,
        temperature=temperature,
        new_conversation=True,
    )
    try:
        return chatbot.handle_message(message)
    except Exception as error:
        traceback.print_exc()
        response = {"content": str(error), "speed": None}

    audio = None
    if tts != "alltalk":
        audio = generate_audio(response["content"], audio_speed)

    return jsonify(
        {
            **response,
            "audio": audio
        }
    )


@app.route("/info")
def info():

    info = ChatBot.get_model_info()
    return jsonify(info)


@app.route("/list")
def list():

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

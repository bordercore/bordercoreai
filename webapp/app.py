import base64
import io
import warnings
import wave

import ffmpeg
import numpy as np
import piper
import sounddevice  # Adding this eliminates an annoying warning
from chatbot import ChatBot, Context
from flask import Flask, jsonify, render_template, request, session

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import whisper

app = Flask(__name__)
app.debug = True
app.secret_key = ""


@app.route("/")
def main():

    return render_template(
        "index.html",
        session=dict(session)
    )


@app.route("/speech2text", methods=["POST"])
def speech2text():

    audio = request.files["audio"].read()
    model = whisper.load_model("base")
    result = model.transcribe(load_audio(audio))

    return jsonify(
        {
            "input": result["text"]
        }
    )


def generate_audio(message, length_scale):
    voice = piper.PiperVoice.load(
        model_path="en_US-amy-medium.onnx",
        config_path="en_US-amy-medium.onnx.json"
    )

    binary_stream = io.BytesIO()

    # Use the binary stream as the destination for the WAV data
    with wave.open(binary_stream, "wb") as wav:
        voice.synthesize(message, wav, length_scale=length_scale)

    # Get the binary data from the stream
    binary_data = binary_stream.getvalue()

    return base64.b64encode(binary_data).decode("utf-8")


def map_speech_rate_value(x):
    """
    Map values used by Piper TTS, range 0.5 (fast) to 1.5 (slow)
    to values used by the UI, range 1 (slow) to 10 (fast)
    """

    # Source range
    a, b = 0.5, 1.5

    # Target range
    c, d = 10, 1

    return c + (d - c) * (x - a) / (b - a)


@app.route("/chat", methods=["POST"])
def chat():

    message = request.form["message"]
    speak = request.form.get("speak", "false")
    length_scale = float(request.form.get("length_scale", 1.0))  # Playback speed
    temperature = float(request.form.get("temperature", 0.7))
    control_lights = request.form.get("control_lights", False)

    session.permanent = True
    session["speak"] = speak.lower() == "true"  # Convert "true" to True, for example
    session["length_scale"] = map_speech_rate_value(length_scale)

    context = Context()
    chatbot = ChatBot(
        context,
        assistant=False,
        debug=False,
        chat_mode="instruct",
        voice=False,
        speak=False,
        temperature=temperature,
        new_conversation=True,
        control_lights=control_lights
    )
    response = chatbot.send_message_to_model(message)

    audio = None
    if speak:
        audio = generate_audio(response, length_scale)

    return jsonify(
        {
            "response": response,
            "audio": audio
        }
    )


@app.route("/info")
def info():

    info = ChatBot.get_model_info()

    return jsonify(
        {
            "response": info,
        }
    )


@app.route("/list")
def list():

    model_list = ChatBot.get_model_list()

    return jsonify(
        {
            "response": model_list,
        }
    )


@app.route("/load", methods=["POST"])
def load():

    model = request.form["model"]
    info = ChatBot.load_model(model)

    return jsonify(
        {
            "response": info,
        }
    )


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

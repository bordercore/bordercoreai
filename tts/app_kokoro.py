"""
TTS Audio Generator Web Service

This module provides a Flask web service that generates text-to-speech audio
using the Kokoro TTS pipeline. It streams the generated audio as a WAV file response.

Command line arguments:
    --voice: Voice to use for TTS (default: bf_emma)
    --debug: Enable debug mode for saving audio files and Flask debugging
    --tts_speed: Speed of TTS voice (default: 1.0)

Usage:
    python -m tts --voice bf_emma --tts_speed 1.2
"""

import argparse
import hashlib
import io
import os
import time
from typing import Iterator

import numpy as np
import soundfile as sf
from flask import Flask, Response, request, stream_with_context
from flask_cors import CORS
from kokoro import KPipeline

# Set up command line argument parser
parser = argparse.ArgumentParser(description="TTS audio generator")
parser.add_argument("--voice", dest="tts_voice", default="bf_emma",
                    help="Voice to use for TTS")
parser.add_argument("--tts_speed", dest="tts_speed", type=float, default=1.0,
                    help="Speed of TTS voice")
parser.add_argument("--debug", dest="debug", action="store_true", default=False,
                    help="Voice to use for TTS")
args = parser.parse_args()

app = Flask(__name__)
CORS(app)

# Get values from command line args
TTS_VOICE = args.tts_voice
TTS_SPEED = args.tts_speed
DEBUG = args.debug

# Load the model
pipeline = KPipeline(lang_code="a")  # <= make sure lang_code matches voice


@app.route("/", methods=["GET"])
def generate_tts_audio() -> Response:
    """
    **/ (GET)** – Generate text-to-speech audio.

    Expected query parameters
    -------------------------
    text : str
        The text that should be converted to speech.

    Returns
    -------
    flask.Response
        A streaming response with ``Content-Type: audio/wav``.
    """
    payload = request.args
    if DEBUG:
        # Create a filename based on timestamp and a hash of the text
        timestamp = int(time.time())
        text_hash = hashlib.md5(payload["text"].encode()).hexdigest()[:8]
        filename = f"tts_{timestamp}_{text_hash}.wav"
        save_path = os.path.join("saved_audio", filename)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set up streaming response
    def generate_audio_chunks() -> Iterator[bytes]:
        """
        Yield fixed-size chunks (**1024 bytes**) from an in-memory WAV buffer.

        The Kokoro pipeline first creates individual chunks, which are then
        concatenated into a single NumPy array, written to an in-memory WAV
        buffer, and finally streamed back to the caller.
        """
        # Generate, display, and save audio files in a loop.
        generator = pipeline(
            payload["text"],
            voice=TTS_VOICE,
            speed=TTS_SPEED,
            split_pattern=r"\n+"
        )
        all_audio_data = []
        for _, _, chunk in generator:
            audio_data = chunk.squeeze().cpu().numpy()
            all_audio_data.append(audio_data)

        # Concatenate into one array
        combined_audio_data = np.concatenate(all_audio_data) if all_audio_data else np.array([], dtype=np.float32)

        # Save the WAV file to disk
        if DEBUG:
            sf.write(save_path, combined_audio_data, 24000, format="WAV")

        # Write data to an in-memory buffer as a single WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, combined_audio_data, 24000, format="WAV")
        wav_buffer.seek(0)

        # Stream the buffer to the client
        chunk_size = 1024
        while True:
            data = wav_buffer.read(chunk_size)
            if not data:
                break
            yield data

    return Response(stream_with_context(generate_audio_chunks()),
                    mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=DEBUG)

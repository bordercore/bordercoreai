import asyncio
import json
import signal
import sys
import time
from threading import Thread

from flask import Flask, Response
from flask_cors import CORS

from .ld import LD

app = Flask(__name__)
CORS(app)

radar = LD()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


def generate_data():
    while True:
        try:
            radar.run(loop)
        finally:
            print("Disconnecting and closing the loop")
            disconnect()


thread = Thread(target=generate_data)
thread.daemon = True
thread.start()


@app.route("/stream")
def stream():
    def event_stream():
        while True:
            for state in radar.read():
                time.sleep(0.1)
                if state is not None:
                    yield f"data: {get_state_json(state)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


def get_state_json(state):
    state_json = {
        "moving_target_distance": state.moving_target_distance,
        "moving_target_energy": state.moving_target_energy,
        "static_target_distance": state.static_target_distance,
        "static_target_energy": state.static_target_energy,
        "detection_distance": state.detection_distance,
    }
    return json.dumps(state_json)


def disconnect():
    future = asyncio.run_coroutine_threadsafe(radar.disconnect(), loop)
    future.result()


def handle_sigint(signal, frame):
    """
    Gracefully disconnect from the Bluetooth device if the kill signal is sent.
    """
    print("Shutting down...")
    disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)

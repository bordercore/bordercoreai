"""
Expose real‑time *LD2410* radar data over *Server-Sent Events (SSE)*.

This module spins up an `asyncio` loop in a background thread, drives the
:class:`LD` sensor wrapper, and publishes detection frames via a Flask route
at `/stream`.  Clients can subscribe to that endpoint and receive a JSON
blob whenever the sensor reports a new state.
"""

import asyncio
import json
import signal
import sys
import time
from threading import Thread
from typing import Any, Iterator

from flask import Flask, Response
from flask_cors import CORS

from .ld import LD

app = Flask(__name__)
CORS(app)

radar = LD()

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


def generate_data() -> None:
    """Run the radar task inside the global *asyncio* loop.

    This function is executed on a daemon :pyclass:`threading.Thread` so that
    Flask’s main thread remains responsive.  It blocks forever (unless an
    exception occurs) because :meth:`LD.run` contains an infinite polling loop
    by design.
    """
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
def stream() -> Response:
    """Server‑Sent Events endpoint that pushes sensor frames to the client."""

    def event_stream() -> Iterator[str]:
        """Inner generator yielding SSE-formatted JSON blobs.

        Yields:
            Strings already formatted for *event‑stream* consumption, i.e. each
            line begins with `"data: "` and ends with a double newline as
            required by the SSE spec.
        """
        while True:
            for state in radar.read():
                time.sleep(0.1)
                if state is not None:
                    yield f"data: {get_state_json(state)}\n\n"

    return Response(event_stream(), mimetype="text/event-stream")


def get_state_json(state: Any) -> str:
    """Serialise the latest *LD2410* state into a compact JSON string.

    Args:
        state: A state object as emitted by the :pymeth:`LD.read` generator.

    Returns:
        A JSON‑encoded string with the key fields expected by the front‑end.
    """
    state_json = {
        "moving_target_distance": state.moving_target_distance,
        "moving_target_energy": state.moving_target_energy,
        "static_target_distance": state.static_target_distance,
        "static_target_energy": state.static_target_energy,
        "detection_distance": state.detection_distance,
    }
    return json.dumps(state_json)


def disconnect() -> None:
    """Synchronously tear down the BLE connection (if any)."""
    future = asyncio.run_coroutine_threadsafe(radar.disconnect(), loop)
    future.result()


def handle_sigint(sig: int, _frame: Any) -> None:
    """Handle *SIGINT* (Ctrl‑C) by shutting the radar down gracefully.

    Args:
        sig: POSIX signal number (*SIGINT* on Unix is `2`).
        _frame: The current stack frame (unused).
    """
    print("Shutting down...")
    disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)

"""
Interface for LD2410 radar sensor via Bluetooth LE or serial connection.

This module defines the `LD` class, which manages a radar sensor either over
Bluetooth LE (using `bleak` and `ld2410_ble`) or via a direct serial
connection (using the `LD2410` reference implementation).  It exposes helper
methods to initialise the link, read detection data, and gracefully disconnect.
"""

import argparse
import asyncio
import logging
import time
from typing import Any, Iterator, Optional

from api import settings
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from ld2410_ble import LD2410BLE

logging.basicConfig(level=logging.INFO)
logging.getLogger("ld2410_ble").setLevel(logging.INFO)


class LD():
    """
    Wrapper around the LD2410 radar sensor.

    The class attempts to discover the sensor over Bluetooth LE unless a
    serial device path is provided. Once connected, it keeps polling the
    sensor and stores the last detection payload in `last_detection`.
    """

    def __init__(self, serial_device: Optional[str] = None, debug: bool = False) -> None:
        """
        Initialize the radar interface.

        Args:
            serial_device: Path to the serial port (e.g. ``"/dev/ttyUSB0"``). If
                *None*, the wrapper will fall back to BLE discovery.
            debug: Emit verbose output, including every raw state update.
        """
        self.serial_device = serial_device
        self.debug = debug
        self.last_detection = None
        # Populated only in BLE mode
        self.scanner: Optional[BleakScanner] = None
        self.ld2410b: Optional[LD2410BLE] = None
        # Populated only in serial mode
        self.radar: Any = None

    def run(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start streaming detections using the provided *asyncio* loop.

        Args:
            loop: The event loop that BLE coroutines should run on.

        This call blocks indefinitely, either inside an asynchronous BLE task
        or in a synchronous serial polling loop.
        """
        if self.serial_device:
            self.handle_serial()
        else:
            loop.run_until_complete(self.handle_bluetooth())

    def on_state_changed(self, state: Any) -> None:
        """
        Internal callback fired whenever fresh data arrives from the sensor.

        Args:
            state: A dictionary-like payload emitted by the underlying driver.
        """
        if self.debug:
            print(state)
        self.last_detection = state

    async def handle_bluetooth(self) -> None:
        """
        Perform BLE discovery, connect to the sensor, and stream notifications.

        This coroutine blocks forever so that the event loop stays alive and
        incoming notifications can be handled.
        """
        def on_detected(device: BLEDevice, adv: AdvertisementData) -> None:
            """Scanner callback used to find the target MAC address."""
            if future.done():
                return
            if self.debug:
                print(f"Detected: {device}")
            if device.address.lower() == settings.sensor_bt_address.lower():
                print(f"Found device: {device.address}")
                future.set_result(device)

        self.scanner = BleakScanner(detection_callback=on_detected)
        future: asyncio.Future[BLEDevice] = asyncio.Future()

        await self.scanner.start()

        device = await future

        ld2410b = LD2410BLE(device)
        self.ld2410b = ld2410b
        cancel_callback = ld2410b.register_callback(self.on_state_changed)
        await ld2410b.initialise()

        # We must keep this coroutine running indefinitely
        while True:
            await asyncio.sleep(1)

        cancel_callback()
        await self.scanner.stop()

    def handle_serial(self) -> None:
        """
        Connect to the sensor over a UART adapter and print detections forever.

        **Warning** – this method never returns; it uses a blocking loop.
        """
        from LD2410 import LD2410, PARAM_BAUD_256000

        self.radar = LD2410(self.serial_device, PARAM_BAUD_256000, verbosity=logging.INFO)

        # Set max detection gate for moving to 2, static to 3, and empty timeout to 1s
        self.radar.edit_detection_params(0, 0, 1)

        # Set the gate 3 moving energy sentivity to 50 and static sensitivity to 40
        # Note: Static sensitivity cannot be set for gate 1 and 2, it must be set to zero e.g (1, 50, 0)
        self.radar.edit_gate_sensitivity(3, 80, 80)
        self.radar.edit_gate_sensitivity(4, 50, 50)
        self.radar.edit_gate_sensitivity(5, 50, 50)
        self.radar.edit_gate_sensitivity(6, 50, 50)
        self.radar.edit_gate_sensitivity(7, 50, 50)
        self.radar.edit_gate_sensitivity(8, 50, 50)

        # Start the radar polling. The radar polls asynchronously at 10Hz.
        self.radar.start()

        while True:
            # The right 2 arrays will be blank since we are polling in standard mode
            print(self.radar.get_data())
            time.sleep(0.1)

    def read_detection_params(self) -> None:
        """
        Query the sensor for its current detection parameters and log them.
        """
        params = self.radar.read_detection_params()
        logging.info("Detection params: %s", params)
        logging.info("Thresholds")
        logging.info("  Max Moving Gate: %s", params[0][0])
        logging.info("  Max Static Gate: %s", params[0][1])
        logging.info("  Empty Timeout: %s", params[0][2])
        logging.info("Moving Gate Sensitivies")
        for gate in range(0, 9):
            logging.info("  Gate {gate}: %s", params[1][gate])
        logging.info("Static Gate Sensitivies")
        for gate in range(0, 9):
            logging.info("  Gate %s: %s", gate, params[2][gate])

    def read(self) -> Iterator[Optional[Any]]:
        """
        Continuously yield the most recent detection payload.

        Yields:
            The latest detection dictionary, or ``None`` when no data has been
            received yet. The generator never terminates on its own.
        """
        while True:
            yield self.last_detection
            time.sleep(0.1)

    async def disconnect(self) -> None:
        """Terminate the BLE connection if one is active."""
        if self.ld2410b and self.ld2410b._client.is_connected:
            await self.ld2410b.stop()

    def notification_handler(self, _sender: int, data: bytearray) -> None:
        """
        Debug helper that dumps raw BLE notifications in hexadecimal.

        Args:
            _sender: GATT handle emitting the notification (unused).
            data: Raw payload as returned by BlueZ.
        """
        print(f"Notification received: %{data.hex()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true", default=False)
    parser.add_argument("-s", "--serial-device", help="Serial device")
    config = parser.parse_args()

    arg_debug = config.debug
    arg_serial_device = config.serial_device

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    radar = LD(serial_device=arg_serial_device, debug=arg_debug)
    radar.run(event_loop)

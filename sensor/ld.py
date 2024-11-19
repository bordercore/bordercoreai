import argparse
import asyncio
import logging
import time

from api import settings
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from ld2410_ble import LD2410BLE

logging.basicConfig(level=logging.INFO)
logging.getLogger("ld2410_ble").setLevel(logging.INFO)


class LD():

    def __init__(self, serial_device=None, debug=False):
        self.serial_device = serial_device
        self.debug = debug
        self.last_detection = None

    def run(self, loop):
        if self.serial_device:
            self.handle_serial()
        else:
            loop.run_until_complete(self.handle_bluetooth())

    def on_state_changed(self, state):
        if self.debug:
            print(state)
        self.last_detection = state

    async def handle_bluetooth(self) -> None:

        def on_detected(device: BLEDevice, adv: AdvertisementData) -> None:
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
        cancel_callback = ld2410b.register_callback(lambda x: self.on_state_changed(x))
        await ld2410b.initialise()

        # We must keep this coroutine running indefinitely
        while True:
            await asyncio.sleep(1)

        cancel_callback()
        await self.scanner.stop()

    def handle_serial(self):
        from LD2410 import LD2410, PARAM_BAUD_256000

        self.radar = LD2410(serial_device, PARAM_BAUD_256000, verbosity=logging.INFO)

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

        while (True):
            # The right 2 arrays will be blank since we are polling in standard mode
            print(self.radar.get_data())
            time.sleep(0.1)

    def read_detection_params(self):
        params = self.radar.read_detection_params()
        logging.info(f"Detection params: {params}")
        logging.info("Thresholds")
        logging.info(f"  Max Moving Gate: {params[0][0]}")
        logging.info(f"  Max Static Gate: {params[0][1]}")
        logging.info(f"  Empty Timeout: {params[0][2]}")
        logging.info("Moving Gate Sensitivies")
        for gate in range(0, 9):
            logging.info(f"  Gate {gate}: {params[1][gate]}")
        logging.info("Static Gate Sensitivies")
        for gate in range(0, 9):
            logging.info(f"  Gate {gate}: {params[2][gate]}")

    def read(self):
        while (True):
            yield self.last_detection
            time.sleep(0.1)

    async def disconnect(self):
        if self.ld2410b._client.is_connected:
            await self.ld2410b.stop()

    def notification_handler(self, _sender: int, data: bytearray) -> None:
        """Handle notification responses."""
        print(f"Notification received: %{data.hex()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", help="Debug mode", action="store_true", default=False)
    parser.add_argument("-s", "--serial-device", help="Serial device", default="/dev/serial0")
    args = parser.parse_args()

    debug = args.debug
    serial_device = args.serial_device

    radar = LD(serial_device=serial_device, debug=debug)
    radar.run()

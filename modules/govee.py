# Inspired by https://medium.com/@richardhayes777/using-chatgpt-to-control-hue-lights-37729959d94f

import argparse

import requests
from api import settings
from http_constants.status import HttpStatus
from requests.exceptions import HTTPError

URL_BASE = "https://developer-api.govee.com/v1"
headers = {
    "Govee-API-Key": settings.govee_api_key,
    "Content-Type": "application/json"
}

CYAN = "\033[36m"
MAGENTA = "\033[35m"
RED = "\033[91m"
END = "\033[0m"


def get_devices():

    url = f"{URL_BASE}/devices"

    response = requests.get(url, headers=headers)

    return response.json()


def balance_braces(input_string):
    # Remove excessive close braces from an LLM response.
    #  Useful if the LLM returns bogus JSON.
    open_count = 0
    close_count = 0

    # Count opening and closing braces
    for char in input_string:
        if char == "{":
            open_count += 1
        elif char == "}":
            close_count += 1

    # Determine the number of extra closing braces
    extra_closing = max(0, close_count - open_count)

    # Remove extra closing braces from the end
    if extra_closing > 0:
        input_string = input_string.rstrip("}")
        input_string += "}" * (close_count - extra_closing)

    return input_string


def build_prompt(device_list):
    device_string = ""
    for device in reversed(device_list["data"]["devices"]):
        device_string += f"One device has id '{device['device']}' with model name '{device['model']}'. I will refer to this device as '{device['deviceName']}'.\n"

    return f"""
I want to control several lights, which I call devices, each of which can be turned to a specific color. I will give you an instruction, and you will give me a JSON document based on that instruction.

{device_string}

Each JSON should have three fields: "device", "model", and "cmd".

The "device" field has a value that matches the device id.

The "model" field has a value that matches the device model name.

The "cmd" field has a nested object with two fields, "name" and "value". The "name" field must have the value "color". The "value" field is a nested object with 3 fields, "r", "g", and "b", which represent red, green, and blue color values. Use these values if I want to change the color. You can change more than one of these values at a time to match the target color.

Each color ranges from 0, which is the lowest value, to 255, which is the highest value.

Only give me one JSON object.

Verify that the JSON object represents valid JSON. If it does not, correct the JSON so that it is valid.

Give me the JSON to configure the lights in response to the instructions below. Give only the JSON and no additional characters, text, or comments.

Do not format the JSON by including newlines.

""" + \
"""
For example, if I told you to "turn the floor lights purple", you'd respond with the following JSON: {"device": "3E:D8:A4:C1:38:A4:C9:6E", "model": "H6159", "cmd": {"name": "color", "value": {"r": 128, "g": 0, "b": 128}}}

Here is the instruction:
"""


def control_device(payload):
    url = f"{URL_BASE}/devices/control"

    response = requests.put(url, headers=headers, data=payload)

    if response.status_code != HttpStatus.OK:
        print(response.json()["message"])


def control_lights(model_name, command, device_list=None):

    if not device_list:
        device_list = get_devices()

    if "status" in device_list and device_list["status"] != HttpStatus.OK:
        raise HTTPError(f"Error getting device list from Govee: {device_list['message']}")

    args = {"temperature": 0.1}

    from modules.chatbot import ChatBot
    chatbot = ChatBot(model_name)
    response = chatbot.send_message_to_model(
        f"{build_prompt(device_list)}\n{command}",
        args
    )
    content = ChatBot.get_streaming_message(response)
    content = balance_braces(content)
    control_device(content)

    return "Done"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model",
        choices=["openai", "llama"],
        default="llama",
        help="The model to use: chatgpt or llama"
    )
    args = parser.parse_args()
    model = args.model

    device_list = get_devices()

    print("Devices available: " + ", ".join([x["deviceName"] for x in device_list["data"]["devices"]]))
    while True:
        command = input(f"{MAGENTA}Command:{END} ")
        control_lights(model, command, device_list=device_list)

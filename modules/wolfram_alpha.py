"""
This module defines a function to query Wolfram Alpha for simple
calculations and a FunctionCall integration for use in model-based toolchains.
"""

import urllib.parse

import requests
from api import settings

from modules.function_calling import FunctionCall


def calculate(query: str) -> str:
    """
    Perform a mathematical calculation using the Wolfram Alpha API.

    Args:
        query: A natural language or symbolic math expression.
    Returns:
        The textual result of the calculation returned by Wolfram Alpha.
    """
    uri_api = f"http://api.wolframalpha.com/v1/result?appid={settings.wolfram_alpha_app_id}&i={urllib.parse.quote(query)}"
    result = requests.get(uri_api, timeout=20).text

    if settings.debug:
        print(result)

    return result


class WolframAlphaFunctionCall(FunctionCall):
    """
    A wrapper class for calling Wolfram Alpha as a tool function in an agent chain.
    """

    tool_name = "wolfram_alpha"
    tool_list = "calculate"


if __name__ == "__main__":
    while True:
        user_input = input("Query: ")
        func = WolframAlphaFunctionCall("")
        print(func.run(user_input))

import argparse

import requests
from api import settings

from modules.function_calling import FunctionCall


def calculate(query: str) -> float:
    """
    Perform some mathematical calculation.

    Args:
        query: The query containing the calculation to perform.
    Returns:
        The result of the mathematical calculation.
    """
    URI_API = f"http://api.wolframalpha.com/v1/result?appid={settings.wolfram_alpha_app_id}&i={query}"
    result = requests.get(URI_API).text

    if settings.debug:
        print(result)

    return result


class WolframAlphaFunctionCall(FunctionCall):

    tool_name = "wolfram_alpha"
    tool_list = "calculate"


if __name__ == "__main__":

    while True:
        query = input("Query: ")
        func = WolframAlphaFunctionCall("")
        print(func.run(query))

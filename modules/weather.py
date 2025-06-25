"""
This module provides a function to retrieve current weather conditions and forecasts
from a remote API, format the information into natural language, and generate a
response to a weather-related question using a language model.
"""

from typing import TYPE_CHECKING

import requests
from api import settings

if TYPE_CHECKING:
    from mypackage.chatbot import ChatBot



def get_weather_info(chatbot: "ChatBot", command: str) -> str:
    """
    Retrieve current weather and forecast data from a weather API, construct a weather summary,
    and generate a response to a weather-related question using a language model.

    Args:
        chatbot: ChatBot instance providing LLM access
        command: The weather-related question or command from the user.

    Returns:
        The generated response from the language model, based on the weather data.
    """
    uri_api = f"http://api.weatherapi.com/v1/forecast.json?key={settings.weather_api_key}&q=02138&days=1&aqi=yes&alerts=yes"
    weather_info = requests.get(uri_api, timeout=20).json()

    weather_description = f"""
    The current temperature is {int(weather_info['current']['temp_f'])}.
    The current condition is {weather_info['current']['condition']['text']}.
    The current humidity is {weather_info['current']['humidity']}.
    The current wind chill is {int(weather_info['current']['windchill_f'])}.
    The weather forecast is {weather_info['forecast']['forecastday'][0]['day']['condition']['text']} with a high of {int(weather_info['forecast']['forecastday'][0]['day']['maxtemp_f'])} and a low of {int(weather_info['forecast']['forecastday'][0]['day']['mintemp_f'])}.
    The sunrise is {weather_info['forecast']['forecastday'][0]['astro']['sunrise']}.
    The sunset is {weather_info['forecast']['forecastday'][0]['astro']['sunset']}.
    The moon phase is {weather_info['forecast']['forecastday'][0]['astro']['moon_phase']}.
    """

    if len(weather_info["alerts"]["alert"]) > 0:
        weather_description += f"""
        There is a weather alert: {weather_info['alerts']['alert'][0]['event']}. The description for this alert is {weather_info['alerts']['alert'][0]['desc']}. It expires on {weather_info['alerts']['alert'][0]['expires']}
        """

    prompt = f"""
    I will give you a series of statements that describes either the current weather, or the weather forecast. I want you to answer a weather question based on those statements and nothing else. In particular, if the question is a general query about the weather, give me a concise summary of the weather. If there is a weather alert, tell me what kind of weather alert it is, its description, and when it expires, in a date and time with the format like January 1, 2023 at 3pm. Put the weather alert information in a separate paragraph. If there is no weather alert information present then don't mentio weather alerts. The weather question is the following: {command}. Answer that question based on the following series of statements about the weather: {weather_description}.
    """
    args = {"temperature": 0.1}

    return chatbot.send_message_to_model(prompt, args)

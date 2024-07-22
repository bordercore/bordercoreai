import requests
from api import settings

URI_API = f"http://api.weatherapi.com/v1/forecast.json?key={settings.weather_api_key}&q=02138&days=1&aqi=yes&alerts=yes"


def get_weather_info(model_name, command):

    weather_info = requests.get(URI_API).json()

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

    prompt = f"""
    I will give you a series of statements that describes either the current weather, or the weather forecast. I want you to answer a weather question based on those statements and nothing else. In particular, if the question is a general query about the weather, say 'Currently it's 79 degrees with mostly sunny skies'. But instead of that particular weather statement substitute the actual weather conditions based on the statements. The weather question is the following: {command}. Answer that question based on the following series of statements about the weather: {weather_description}.
    """
    args = {"temperature": 0.1}

    from modules.chatbot import ChatBot
    chatbot = ChatBot(model_name)
    response = chatbot.send_message_to_model(prompt, args)

    return {
        "content": response["content"],
        "speed": response["speed"]
    }

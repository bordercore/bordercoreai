import json
import os
import urllib.parse

import requests
from api import settings

URI_MUSIC = f"{settings.music_api_host}/api/search/music"


def play_music(model_name, command):
    prompt = """
    I will give you an instruction to play music. From the instruction you must select an artist name, a song name, or an ablum name, or all three. I want your response in JSON format. For example, if the instruction is "Play me With or Without You by U2", you would respond with the following JSON: {"artist": "U2", "song": "With or Without You"}. For the instruction "Play Just Drive by Wolf Club", you would respond with the following JSON: {"artist": "Wolf Club", "song": "Just Drive"}. If no artist is provided, do not include an artist field in the JSON; only include the song. For example, for the instruction "Play the song Promise", you would respond with the following JSON: {"song": "Promise"}. If no song is provided in the instruction, do not include a song field in the JSON. In this case I want to play an artist and not a song. For example, for the instruction "Play Foo Fighters", you would respond with the following JSON: {"artist": "Foo Fighters"}.

Verify that the JSON object represents valid JSON. If it does not, correct the JSON so that it is valid.

Give me only the JSON and no additional characters, text, or comments.

Do not format the JSON by including newlines.

Here is the instruction:
    """

    prompt = prompt + command
    args = {"temperature": 0.1}

    from chatbot import ChatBot
    chatbot = ChatBot(model_name)
    response = chatbot.send_message_to_model(prompt, args)

    # Get the song info from the music API
    headers = {
        "Authorization": f"Token {os.environ.get('DRF_TOKEN_JERRELL')}",
    }
    query_string = urllib.parse.urlencode(json.loads(response["content"]))
    music_info = requests.get(f"{URI_MUSIC}?{query_string}", headers=headers).json()
    print(music_info)
    if not music_info:
        content = "Sorry, no music found that matches."
    elif "album" in response["content"]:
        content = f"Playing album by {music_info[0]['artist']}."
    elif len(music_info) > 1:
        content = f"More than one song found. Playing the first one, {music_info[0]['title']} by {music_info[0]['artist']}."
    else:
        content = f"Playing {music_info[0]['title']} by {music_info[0]['artist']}."

    return {
        "music_info": music_info,
        "content": content,
        "speed": response["speed"]
    }

"""
Music playback command parser and executor using an LLM and a music search API.

This module accepts natural language music commands, uses an LLM to extract structured
JSON containing artist/song/album information, then queries a music API and returns
a formatted response with playback instructions.
"""

import json
import os
import urllib.parse

import requests
from api import settings


def play_music(model_name: str, command: str) -> str:
    """
    Parse a natural language command to play music using an LLM, then look up matching music.

    Args:
        model_name: The name of the LLM model to use for parsing the command.
        command: A string instruction like "Play Just Drive by Wolf Club".

    Returns:
        A JSON-formatted string prefixed with CONTROL_VALUE, containing:
            - content: A user-facing string summarizing playback.
            - music_info: A list of results from the music API.
    """
    prompt = """
    I will give you an instruction to play music. From the instruction you must select an artist name, a song name, or an ablum name, or all three. I want your response in JSON format. For example, if the instruction is "Play me With or Without You by U2", you would respond with the following JSON: {"artist": "U2", "song": "With or Without You"}. For the instruction "Play Just Drive by Wolf Club", you would respond with the following JSON: {"artist": "Wolf Club", "song": "Just Drive"}. If no artist is provided, do not include an artist field in the JSON; only include the song. For example, for the instruction "Play the song Promise", you would respond with the following JSON: {"song": "Promise"}. If I do not explicitly mention playing a song in the instruction, assume I am asking for an artist and do not include a song field in the JSON. In this case I want to play an artist and not a song. For example, for the instruction "Play Foo Fighters", you would respond with the following JSON: {"artist": "Foo Fighters"}.

Verify that the JSON object represents valid JSON. If it does not, correct the JSON so that it is valid.

Give me only the JSON and no additional characters, text, or comments.

Do not format the JSON by including newlines.

Here is the instruction:
    """

    prompt += command
    args = {"temperature": 1.0}

    from modules.chatbot import CONTROL_VALUE, ChatBot
    chatbot = ChatBot(model_name)
    content = json.loads(chatbot.send_message_to_model(prompt, args))

    # Get the song info from the music API
    uri_music = f"{settings.music_api_host}/api/search/music"
    headers = {
        "Authorization": f"Token {os.environ.get('DRF_TOKEN_JERRELL')}",
    }
    query_string = urllib.parse.urlencode(content)
    music_info = requests.get(f"{uri_music}?{query_string}", headers=headers, timeout=20).json()

    if settings.debug:
        print(music_info)

    if not music_info:
        content = "Sorry, no music found that matches."
    elif "album" in content:
        content = f"Playing album by {music_info[0]['artist']}."
    elif len(music_info) > 1:
        content = f"More than one song found. Playing the first one, {music_info[0]['title']} by {music_info[0]['artist']}."
    else:
        content = f"Playing {music_info[0]['title']} by {music_info[0]['artist']}."

    return CONTROL_VALUE + json.dumps({
        "music_info": music_info,
        "content": content,
    })

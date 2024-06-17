import os

flask_secret_key = "b'LT\xec\xb3\xc5\xeb\xa3,\x96\x96\xfebJ1\xf1k\xc2k\x14\x11\xe1\r;\x82'"

api_host = "http://10.3.2.5:5000"
model_name = "hermes-pro-llama3-awq"
model_dir = "../models"
temperature = 0.7
model = None
tokenizer = None

discord_channel_id = "1119339657830858924"

tts_host = "10.3.2.5:7851"
tts_voice = "valerie.wav"

openai_api_key = os.environ.get("OPENAI_API_KEY")
govee_api_key = ""

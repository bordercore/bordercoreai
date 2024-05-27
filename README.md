![Bordercore AI Logo](/logo.jpg)

---

Bordercore AI is a web-based chat UI supporting multiple open-weight LLMs, Text to Speech (TTS) and Speech to Text (STT). Discord bots are also supported.

![Screenshot](/screenshot.png)

# Features

## Text to Speech (TTS)

Two TTS engines are supported: [AllTalk TTS](https://github.com/erew123/alltalk_tts) and [Piper](https://github.com/rhasspy/piper).

## Speech to Text (STT)

[Whisper MIC](https://github.com/mallorbc/whisper_mic) is used for STT, which is based on OpenAI's [Whisper](https://github.com/openai/whisper).

## Discord Bot Support

Discord bots can be backed by either OpenAI's ChatGPT or an open source LLM.

Set your server's channel ID in `shared.discord_channel_id`.

Set the environment variables `DISCORD_TOKEN_CHAD` (for a ChatGPT bot) and `DISCORD_TOKEN_FLOYD` (for an open source LLM).

To run the Floyd bot:

```bash
python3 chatbot.py -m floyd
```

To run the ChatGPT bot:

```bash
python3 chatbot.py -m chad
```


# Installation

The code is divided into two components: a webapp and an API. These can be and often are run on two different systems.

## Webapp

To install the webapp, first create and activate a virtual environment. Then:

```bash
pip install requirements/webapp.txt
```

Build the front-end package:

```bash
npm run build
```

To run:

```bash
cd webapp
PYTHONPATH=.. flask --app app run
```

To access: http://localhost:5000/

## API

To install the API, first create and activate a virtual environment. Then:

```bash
pip install requirements/api.txt
```

Copy `api/shared_template.py` to `api/shared.py` and set the following:

**model_name**: default model to load
**model_dir**: the relative directory containing your models

Edit `models.yaml` to add configuration options for your models. Here is an example entry:

```yaml
NousResearch_Nous-Hermes-2-Mistral-7B-DPO:
  name: Nous Research Hermes 2 Mistral 7B DPO
  template: chatml
  type: mistral
```

The **name** is an alias used in the UI.
The **template** is which chat template type used by the model (eg ChatML, Alpaca, Llama2, etc).
The **type** is used to parse the response from the model.

To run:

```bash
cd api
PYTHONPATH=. FLASK_RUN_HOST=0.0.0.0 flask --app app run
```

## Command line

You can interact with the API via a command-line option:

```bash
python3 ./chatbot.py -m interactive
```

Options:

- -s: enable AllTalk TTS

# Usage

## UI

Type your text into the input box to send a message to the chatbot.

To the immediate right of the input box are two buttons. The first is **Regenerate Response**, which will re-send the last message to the chatbot, presumably in hopes that a different response will result. The second is **New Chat**, which will clear the chat history.

Click the **MIC** switch to enable microphone voice input. Click it off to submit the result to send the result to the chatbot via STT.

Click the **VAD** switch to enable **Voice Activation Detection**. This will turn on the microphone and use VAD to detect when you're done talking to the chatbot, initiating a back-and-forth conversation.

The **Selected Model** dropdown lets you choose which LLM the API uses to respond to your prompt.

The hamburger menu to the upper-right lets you adjust the following:

**Temperature**: The "temperature" of the LLM, which controls the randomness of the model's output.

**Audio Speed**: How fast the TTS audio plays.

**Speak**: This enables TTS.

**TTS Host**: The hostname and port for the TTS server.

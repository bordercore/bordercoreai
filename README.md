![Bordercore AI Logo](/logo.jpg)

---

Bordercore AI is a web-based AI chatbot and voice assistant supporting multiple open-weight and commercial LLMs, Text to Speech (TTS), Speech to Text (STT), audio transcription and RAG (Retrieval Augmented Generation). Discord bots are also supported.

![Screenshot](/screenshot.png)

# Features

## Text to Speech (TTS)

Two TTS engines are supported: [AllTalk TTS](https://github.com/erew123/alltalk_tts) and [Piper](https://github.com/rhasspy/piper).

## Speech to Text (STT)

[Whisper MIC](https://github.com/mallorbc/whisper_mic) is used for STT, which is based on OpenAI's [Whisper](https://github.com/openai/whisper).

## RAG (Retrieval Augmented Generation)

Chat with your uploaded documents.

## Audio Transcription

Upload audio files to convert them to text, then ask questions based on the generated transcription.

## Discord Bot Support

Discord bots can be backed by either OpenAI's ChatGPT or an open source LLM.

Set your server's channel ID in `settings.discord_channel_id`.

Set the environment variable `DISCORD_TOKEN`.

To run the local LLM bot:

```bash
python3 -m modules.chatbot -m localllm
```

To run the ChatGPT bot:

```bash
python3 -m modules.chatbot -m chatgpt
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
python3 -m webapp
```

To access: http://localhost:5000/

## API

To install the API, first create and activate a virtual environment. Then:

```bash
pip install requirements/api.txt
```

Copy `api/settings_template.py` to `api/settings.py` and set the following:

**model_name**: default model to load
**model_dir**: the relative directory containing your models

Edit `models.yaml` to add configuration options for your models. Here is an example entry:

```yaml
NousResearch_Nous-Hermes-2-Mistral-7B-DPO:
  name: Nous Research Hermes 2 Mistral 7B DPO
  template: chatml
gpt-4o:
  name: ChatGPT-4o
  type: api
  vendor: openai
```

The **name** is a human-friendly alias used in the UI.
The **template** is the chat template type used by the model (eg ChatML, Alpaca, Llama2, etc).
The **type** specifies an API-based (as opposed to local) model.
The **vendor** specifies the vendor for commercial models. Can be set to *openai* or *anthropic*.
Set **quantize: true** to automatically quantize models to 4bits using the bitsandbytes library.

To run:

```bash
python -m api
```

## Command line

You can interact with the API via a command-line option:

```bash
python3 -m modules.chatbot -m interactive
```

Options:

- -s: enable AllTalk TTS

To use RAG with a local file:

```bash
python3 ./rag.py -f <filename>
```

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

# Tests

To run the unit tests:

```bash
pytest
```

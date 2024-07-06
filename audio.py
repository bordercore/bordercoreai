# How to improve performance:
#
# Use torch_dtype = 16 instead of leaving it set to the default of 32.
# Use the distil-whisper models instead of the openai/whisper-* ones.
# Set the batch_size when creating the pipeline.
#
# I wasn't able to get Speculative Decoding to work because of
#  a transformers error.
#
# This runs much faster if the attn_implementation parameter for
#  AutoModelForSpeechSeq2Seq is set to "eager" instead of the default
#  of "sdpa" for PyTorch 2.1.1 and later. Why?

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Audio():

    DEFAULT_MODEL = "distil-whisper/distil-large-v3"

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name

    def transcribe(self, filename=None, audio_data=None, timestamps=False):

        if not filename and audio_data is None:
            raise ValueError("Either filename or audio_data must be provided")

        start = time.time()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
            attn_implementation="eager"
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_name)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        args = {
            "batch_size": 8,
            "return_timestamps": True
        }
        if filename:
            result = pipe(filename, **args)
        else:
            result = pipe(audio_data, **args)

        if timestamps:
            output_filename = f"{Path(filename).stem}_chunks.txt"
            with Path(output_filename).open("w") as file:
                file.write(str(result["chunks"]))

        print(f"Time: {time.time() - start}")

        return result["text"]

    def query_transcription(self, model_name, message, transcript):

        prompt = f"""
    Answer the following question based ONLY on the following text. Do not use any other source of information. The question is '{message}'. The text is '{transcript}'
    """

        from chatbot import ChatBot
        chatbot = ChatBot(model_name)
        response = chatbot.send_message_to_model(prompt)

        return {
            "content": response["content"],
            "speed": response["speed"]
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-m",
        "--model-name",
        help="The whisper model",
        default=Audio.DEFAULT_MODEL
    )
    parser.add_argument(
        "filename",
        help="The audio file to transcribe"
    )
    parser.add_argument(
        "-t",
        "--timestamps",
        help="Get segment-level timestamps",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    model_name = args.model_name
    filename = args.filename
    timestamps = args.timestamps

    audio = Audio(model_name=model_name)
    result = audio.transcribe(filename=filename, timestamps=timestamps)

    output_filename = f"{Path(filename).stem}.txt"
    with Path(output_filename).open("w") as file:
        file.write(result)

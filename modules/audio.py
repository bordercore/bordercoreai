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
import os
import time
from pathlib import Path

import torch
import yt_dlp
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
            "return_timestamps": True
        }
        if filename:
            result = pipe(filename, **args)
        else:
            result = pipe(audio_data, **args)

        if timestamps:
            fixed_timestamps = self.fix_timestamps(result["chunks"])
            output_filename = f"{Path(filename).stem}_chunks.txt"
            with Path(output_filename).open("w") as file:
                file.write(str(fixed_timestamps))

        print(f"Time: {time.time() - start}")

        return result["text"]

    def download_audio(self, url):
        """
        Download the audio from a Youtube video as an mp3 file.
        """
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "outtmpl": "/tmp/%(title)s.%(ext)s",
            }

            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                audio_file = ydl.prepare_filename(info_dict)

                # Replace the original extension with .mp3
                mp3_file = os.path.splitext(audio_file)[0] + ".mp3"

            return mp3_file
        except Exception as e:
            print(f"Error downloading Youtube audio: {e}")
            return None

    def fix_timestamps(self, timestamps):
        """
        Adjust a list of timestamps to create a continuous sequence.
        This method processes a list of timestamp dictionaries, adjusting the start
        and end times to ensure a continuous sequence.
        """
        fixed_timestamps = []
        offset = 0

        for i, x in enumerate(timestamps):
            if x["timestamp"][0] == 0 and i != 0:
                offset += timestamps[i - 1]["timestamp"][1]
            fixed_timestamps.append(
                {
                    "text": x["text"],
                    "timestamp": [
                        round(x["timestamp"][0] + offset, 1),
                        round(x["timestamp"][1] + offset, 1)
                    ]
                }
            )

        return fixed_timestamps

    def query_transcription(self, model_name, messages, transcript):

        content = messages[-1]["content"]
        prompt = f"""
    Answer the following question based ONLY on the following text. Do not use any other source of information. The question is '{content}'. The text is '{transcript}'
    """

        from modules.chatbot import ChatBot
        chatbot = ChatBot(model_name)
        response = chatbot.send_message_to_model(prompt, {})
        return ChatBot.get_streaming_message(response)


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

    config = parser.parse_args()
    model_name = config.model_name
    filename = config.filename
    timestamps = config.timestamps

    audio = Audio(model_name=model_name)
    result = audio.transcribe(filename=filename, timestamps=timestamps)

    input_path = Path(filename)
    output_path = input_path.with_suffix(".txt")

    with output_path.open("w", encoding="utf-8") as file:
        file.write(str(result))

"""audio.py
=================
High‑level wrapper utilities for transcribing or interrogating audio using
Whisper‑style models.

The module exposes an :class:`Audio` helper that can

* **Transcribe** an audio file or raw waveform.
* **Download** audio from YouTube and save it as an MP3 file.
* **Normalise** Whisper timestamp chunks so they form a continuous timeline.
* **Query** a transcript with an LLM to obtain answers that are grounded in
  the text.

The script section under ``if __name__ == "__main__"`` allows the file to be
executed directly from the command line::

    python audio.py --model-name distil-whisper/distil-medium.en sample.mp3

How to improve performance:

Use torch_dtype = 16 instead of leaving it set to the default of 32.
Use the distil-whisper models instead of the openai/whisper-* ones.
Set the batch_size when creating the pipeline.

I wasn't able to get Speculative Decoding to work because of a transformers error.

This runs much faster if the attn_implementation parameter for
 AutoModelForSpeechSeq2Seq is set to "eager" instead of the default
 of "sdpa" for PyTorch 2.1.1 and later. Why?
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TypedDict

import torch
import yt_dlp
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

Timestamp = List[float]  # [start, end] seconds


class Chunk(TypedDict):
    """Single segment of a Whisper/ASR transcript.

    Attributes
    ----------
    timestamp : tuple[float, float]
        A pair ``(start, end)`` giving the segment’s start and end times in seconds.
    text : str
        The raw transcript text that spans the timestamp interval.
    """
    timestamp: Tuple[float, float]
    text: str


class Audio:
    """Utility class for Whisper transcription and related tasks."""

    DEFAULT_MODEL: str = "distil-whisper/distil-large-v3"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Create a new :class:`Audio` helper.

        Parameters
        ----------
        model_name:
            Hugging Face model identifier to load.  Defaults to
            ``distil-whisper/distil-large-v3``.
        """

        self.model_name = model_name

    def transcribe(
        self,
        *,
        filename: Optional[str] = None,
        audio_data: Optional[Any] = None,
        timestamps: bool = False,
    ) -> str:
        """Transcribe *either* an audio file *or* in‑memory waveform.

        Exactly one of ``filename`` or ``audio_data`` must be supplied.

        Parameters
        ----------
        filename:
            Path to a local audio file supported by *ffmpeg*.
        audio_data:
            Raw waveform or numpy array produced by ``librosa.load`` or
            similar.  Must be compatible with 🤗 Transformers pipelines.
        timestamps:
            If *True*, write per‑chunk timestamps to ``<filename>_chunks.txt``.

        Returns
        -------
        str
            The transcribed text.

        Raises
        ------
        ValueError
            If neither ``filename`` nor ``audio_data`` is provided.
        """
        if not filename and audio_data is None:
            raise ValueError("Either filename or audio_data must be provided")

        start_time = time.time()

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(self.model_name)

        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            torch_dtype=torch_dtype,
            device=device,
        )

        result = (
            asr_pipeline(filename, return_timestamps=True)
            if filename
            else asr_pipeline(audio_data, return_timestamps=True)
        )

        if timestamps and filename is not None:
            fixed = self.fix_timestamps(result["chunks"])
            out_file = f"{Path(filename).stem}_chunks.txt"
            Path(out_file).write_text(str(fixed), encoding="utf-8")

        print(f"Time: {time.time() - start_time}")
        return str(result["text"])

    def download_audio(self, url: str) -> Optional[str]:
        """Download a YouTube video's audio track as an MP3 file.

        Parameters
        ----------
        url:
            Full YouTube URL to download.

        Returns
        -------
        str | None
            Absolute path to the downloaded ``.mp3`` file or *None* if the
            download fails.
        """
        ydl_opts: Dict[str, Any] = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "/tmp/%(title)s.%(ext)s",
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                audio_file = ydl.prepare_filename(info_dict)

            return os.path.splitext(audio_file)[0] + ".mp3"
        except Exception as e:
            print(f"Error downloading YouTube audio: {e}")
            return None

    def fix_timestamps(self, timestamps: List[Chunk]) -> List[Chunk]:
        """Convert Whisper timestamp *chunks* into a continuous timeline.

        Whisper occasionally resets the timestamp counter within a long file.
        This helper **adds an offset** when such a reset is detected so that the
        output is monotonically increasing.

        Parameters
        ----------
        timestamps:
            List of chunk dictionaries from the Whisper pipeline.

        Returns
        -------
        list[Chunk]
            New list with adjusted ``timestamp`` ranges.
        """
        fixed: List[Chunk] = []
        offset: float = 0.0

        for idx, chunk in enumerate(timestamps):
            start, end = chunk["timestamp"]
            if start == 0 and idx != 0:  # reset detected ➜ add offset
                offset += timestamps[idx - 1]["timestamp"][1]

            fixed.append(
                {
                    "text": chunk["text"],
                    "timestamp": (
                        round(start + offset, 1),
                        round(end + offset, 1),
                    ),
                }
            )

        return fixed

    def query_transcription(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        transcript: str,
    ) -> Iterator[str]:
        """Answer a user question *solely* from a provided transcript.

        The LLM receives a system prompt that forbids it from using any data
        other than *transcript*.

        Parameters
        ----------
        model_name:
            Identifier for the chat model to load.
        messages:
            Chat history, where the most recent message is the user question.
        transcript:
            The text that serves as the knowledge base for answering.

        Returns
        -------
        Iterator[str]
            Streaming response from the chat model.
        """
        from modules.chatbot import \
            ChatBot  # local import to avoid heavy start‑up cost

        content: str = messages[-1]["content"]
        prompt: str = (
            "Answer the following question based ONLY on the following text. "
            "Do not use any other source of information. "
            f"The question is '{content}'. The text is '{transcript}'"
        )

        chatbot = ChatBot(model_name)
        response = chatbot.send_message_to_model(prompt, {})
        return ChatBot.get_streaming_message(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI wrapper for audio.py")
    parser.add_argument(
        "filename",
        help="Audio file to transcribe (any format supported by ffmpeg)",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        help="Hugging Face model name (default: distil-whisper/distil-large-v3)",
        default=Audio.DEFAULT_MODEL,
    )
    parser.add_argument(
        "-t",
        "--timestamps",
        help="Write per‑chunk timestamps to <filename>_chunks.txt",
        action="store_true",
        default=False,
    )

    config = parser.parse_args()

    audio = Audio(model_name=config.model_name)
    TEXT = audio.transcribe(filename=config.filename, timestamps=config.timestamps)

    output_path = Path(config.filename).with_suffix(".txt")
    output_path.write_text(TEXT, encoding="utf-8")

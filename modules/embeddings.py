"""
A module for generating text embeddings using the OpenAI API.

This utility provides functions to create embedding vectors from text. The main
function, `len_safe_get_embedding`, handles texts longer than the model's
context limit by chunking the input, embedding each chunk, and averaging
the results into a single, normalized vector.

Requires the `OPENAI_API_KEY` environment variable to be set.
"""

import os
from itertools import islice
from typing import Iterable, Iterator, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import openai
import tiktoken

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = "cl100k_base"

openai.api_key = os.environ.get("OPENAI_API_KEY")

T = TypeVar("T")


def get_embedding(text_or_tokens: Union[str, Sequence[int]], model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate an embedding vector for the supplied text or token sequence.

    Args:
        text_or_tokens: A string of text to embed, or an iterable of
            integer token IDs that have already been encoded.
        model: The name of the embedding model to query.

    Returns:
        A list of floating-point numbers that represents the embedding.
    """
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]


def batched(iterable: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    """
    Yield successive tuples of `n` items from `iterable`.

    Example:
        batched("ABCDEFG", 3) -> ABC DEF G

    Args:
        iterable: The source of elements to batch.
        n: The maximum size of each batch; must be at least 1.

    Yields:
        An iterator that produces tuples containing up to `n` consecutive
        elements from the iterable. The final tuple may contain fewer than
        `n` items if the total number of elements is not a multiple of `n`.

    Raises:
        ValueError: If `n` is less than 1.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


def chunked_tokens(text: str, encoding_name: str, chunk_length: int) -> Iterator[Tuple[int, ...]]:
    """
    Yield successive tuples of token IDs of length `chunk_length` from *text*.

    Example:
        chunked_tokens("ABCDEFG", "gpt2", 3) -> (65, 66, 67) (68, 69, 70) (71,)

    Args:
        text: The text to tokenize and split into chunks.
        encoding_name: The name of the tiktoken encoding to use.
        chunk_length: The maximum size of each chunk; must be at least 1.

    Yields:
        An iterator that yields tuples containing up to `chunk_length`
        token IDs. The final tuple may contain fewer than `chunk_length`
        tokens if the total count is not a multiple of `chunk_length`.

    Raises:
        ValueError: If `chunk_length` is less than 1.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def len_safe_get_embedding(
    text: str,
    model: str = EMBEDDING_MODEL,
    max_tokens: int = EMBEDDING_CTX_LENGTH,
    encoding_name: str = EMBEDDING_ENCODING,
) -> List[float]:
    """
    Compute a single embedding for lengthy text by chunking and averaging.

    The function tokenizes the input text, splits the token sequence into
    chunks of at most max_tokens, generates an embedding for each chunk,
    averages the embeddings with weights proportional to chunk lengths,
    and normalizes the result to unit length.

    Args:
        text: The text to embed.
        model: The embedding model identifier.
        max_tokens: Maximum tokens allowed in each chunk.
        encoding_name: The token encoding used for splitting.

    Returns:
        A unit-length embedding vector as a list of floats.
    """
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    averaged = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
    normalised = averaged / np.linalg.norm(averaged)
    return normalised.tolist()

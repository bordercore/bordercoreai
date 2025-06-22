"""
Retrieval-Augmented Generation (RAG) system for local document ingestion and querying.

This script enables document ingestion via PDF or plain text, chunking and embedding
the content into a ChromaDB vector store, and querying the indexed content using an LLM.

Usage:
    Run this script from the command line with arguments for model name, file/text,
    whether to force re-indexing, and whether to use OpenAI embeddings.
"""

import argparse
import hashlib
import os
import uuid
from io import BytesIO
from typing import BinaryIO, List

import chromadb
import openai
import pypdf
from chromadb.api.models.Collection import Collection
from nltk.tokenize import sent_tokenize

from modules.chatbot import ChatBot
from modules.embeddings import len_safe_get_embedding

CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"
RED = "\033[91m"
END = "\033[0m"

openai.api_key = os.environ.get("OPENAI_API_KEY")


class RAG:
    """
    A class for performing retrieval-augmented generation using ChromaDB and OpenAI or local embeddings.
    """

    chunk_size: int = 1024
    overlap: int = 20

    def __init__(self, model_name: str | None, chromdb: str = "chromdb", use_openai_embeddings: bool = True):
        """
        Initialize the RAG system.

        Args:
            model_name: The name of the LLM to use.
            chromdb: Path to ChromaDB persistent directory.
            use_openai_embeddings: Whether to use OpenAI for embedding generation.
        """
        self.model_name = model_name
        self.use_openai_embeddings = use_openai_embeddings
        self.client = chromadb.PersistentClient(path=chromdb)
        self.document: Collection | None = None
        self.collection = None

    def extract_text_from_pdf(self, pdf: BinaryIO) -> str:
        """
        Extract text content from a PDF.

        Args:
            pdf: A binary file-like object of the PDF.

        Returns:
            The extracted text as a single string.
        """
        reader = pypdf.PdfReader(pdf)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        return content

    def add_document(self, text: bytes | None = None, filename: str | None = None, name: str | None = None) -> None:
        """
        Load and index a document into ChromaDB.

        Args:
            text: Raw byte content of the document.
            filename: Path to the document file.
            name: Optional name for the document (used for metadata).
        """
        if filename:
            with open(filename, "rb") as file:
                if filename.endswith("pdf"):
                    self.document = self.extract_text_from_pdf(file)
                else:
                    self.document = file.read().decode("utf-8")
        else:
            if text is None:
                raise ValueError("Text must not be None when no filename is provided.")
            if name and name.lower().endswith("pdf"):
                self.document = self.extract_text_from_pdf(BytesIO(text))
            else:
                self.document = text.decode("utf-8")

        sha1sum = self.get_sha1sum()
        try:
            self.collection = self.client.get_collection(sha1sum)
        except chromadb.errors.InvalidCollectionException:
            self.collection = self.client.create_collection(
                name=sha1sum,
                metadata={
                    "hnsw:space": "cosine",
                    "filename": filename or name
                }
            )
            self.index()

    def get_collection(self, sha1sum: str) -> None:
        """
        Load a ChromaDB collection by its SHA-1 hash.

        Args:
            sha1sum: The SHA-1 hash of the document text.
        """
        self.collection = self.client.get_collection(sha1sum)

    def list_collections(self) -> None:
        """
        Print metadata filenames for all indexed collections.
        """
        for collection in self.client.list_collections():
            print(collection.metadata["filename"])

    def get_sha1sum(self) -> str:
        """
        Compute SHA-1 hash of the currently loaded document.

        Returns:
            The SHA-1 hash as a hex string.
        """
        if self.document is None:
            raise ValueError("No document loaded. Cannot compute SHA-1.")

        hash_sha1 = hashlib.sha1()
        hash_sha1.update(self.document.encode("utf-8"))
        return hash_sha1.hexdigest()

    def chunk_text_by_sentence(self, text: str) -> list[str]:
        """
        Chunk text into overlapping segments based on sentence boundaries.

        Args:
            text: The full input document text.

        Returns:
            A list of text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def index(self) -> None:
        """
        Index the loaded document into ChromaDB.
        """
        print("Indexing...")

        if not self.collection:
            raise RuntimeError("No collection is loaded to index into.")

        chunks = self.chunk_text_by_sentence(self.document)

        for chunk in chunks:
            args = {
                "documents": [chunk],
                "ids": [str(uuid.uuid4())],
            }
            if self.use_openai_embeddings:
                embeddings = len_safe_get_embedding(chunk)
                args["embeddings"] = [embeddings]
            self.collection.add(**args)
        print(f"Added {self.collection.count()} chunks.")

    def get_response(self, question: str, chunks: list[str]) -> str:
        """
        Generate an LLM response based on retrieved chunks.

        Args:
            question: User query string.
            chunks: Relevant document chunks.

        Returns:
            A streamed string response from the LLM.
        """
        prompt = f"Please answer the following question based only on the provided text and nothing else. Here is the question: {question}."
        for chunk in chunks:
            prompt += f" Here is one chunk of text: {chunk}."

        chatbot = ChatBot(self.model_name)
        response = chatbot.send_message_to_model(prompt, {})
        return ChatBot.get_streaming_message(response)

    def query_document(self, query: str) -> str:
        """
        Query the indexed document and retrieve relevant chunks.

        Args:
            query: The user's input query string.

        Returns:
            A streamed string response based on the query.
        """
        if not self.collection:
            raise RuntimeError("No collection is loaded to index into.")

        args = {"n_results": 3}
        if self.use_openai_embeddings:
            embeddings = len_safe_get_embedding(query)
            args["query_embeddings"] = [embeddings]
        else:
            args["query_texts"] = [query]

        results = self.collection.query(**args)
        return ChatBot.get_streaming_message(self.get_response(query, results["documents"]))

    def run(self, force_index: bool = False) -> None:
        """
        Start the interactive query loop.

        Args:
            force_index: If True, re-index the document before querying.
        """
        if force_index:
            self.index()

        while True:
            user_input = input(f"\n\n{MAGENTA}You>{END} ")
            answer = self.query_document(user_input)
            print(f"\n{answer}")


def main() -> None:
    """
    Command-line entry point for running RAG-based document ingestion and querying.

    This function allows users to index or query documents using a retrieval-augmented
    generation (RAG) engine backed by a local or OpenAI LLM. Users can provide either
    a file or a text string to query, and optionally choose the embedding source.

    Command-line arguments:
        -i, --index        : Force re-indexing of the document.
        -m, --model-name   : The model name to use for answering queries (default: gpt-4o-mini).
        -o, --openai       : Use OpenAI embeddings instead of local embedding models.
        -f, --filename     : Path to a document file to ingest and query.
        -t, --text         : Raw text string to ingest and query.
        -l, --list         : List all indexed document collections.

    Behavior:
        - If both `--filename` and `--text` are specified, an error is raised.
        - If neither is provided, an error is raised.
        - Otherwise, the specified input is indexed and a query is run.
        - If `--list` is set, it lists available document collections instead.
    """
    parser = argparse.ArgumentParser(description="Run RAG document ingestion and query engine.")
    parser.add_argument("-i", "--index", help="Index document", action="store_true")
    parser.add_argument("-m", "--model-name", help="The LLM model name", default="gpt-4o-mini")
    parser.add_argument("-o", "--openai", help="Use OpenAI embeddings", action="store_true", default=False)
    parser.add_argument("-f", "--filename", help="The file to query")
    parser.add_argument("-t", "--text", help="The text to query")
    parser.add_argument("-l", "--list", help="List collections", action="store_true")

    args = parser.parse_args()

    if args.filename and args.text:
        raise ValueError("Error: you cannot specify both a filename and text.")
    if not args.filename and not args.text:
        raise ValueError("Error: you must specify either a filename or some text to query.")

    rag = RAG(args.model_name, use_openai_embeddings=args.use_openai_embeddings)
    rag.add_document(text=args.text.encode("utf-8") if args.text else None, filename=args.filename, name=args.filename or "stdin")

    if args.list_collections:
        rag.list_collections()
    else:
        rag.run(args.force_index)


if __name__ == "__main__":
    main()

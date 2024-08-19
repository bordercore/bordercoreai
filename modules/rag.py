import argparse
import hashlib
import os
import uuid
from io import BytesIO

import chromadb
import openai
import PyPDF2
from nltk.tokenize import sent_tokenize

from modules.chatbot import ChatBot
from modules.embeddings import len_safe_get_embedding

CYAN = "\033[36m"
WHITE = "\033[37m"
MAGENTA = "\033[35m"
RED = "\033[91m"
END = "\033[0m"

openai.api_key = os.environ.get("OPENAI_API_KEY")


class RAG():

    # The size of each chunk in characters.
    chunk_size = 1024

    # The number of characters each chunk overlaps with the next.
    overlap = 20

    def __init__(self, model_name, chromdb="chromdb", use_openai_embeddings=True):
        self.model_name = model_name
        self.use_openai_embeddings = use_openai_embeddings
        self.client = chromadb.PersistentClient(path=chromdb)

    def extract_text_from_pdf(self, pdf):
        reader = PyPDF2.PdfReader(pdf)
        content = ""
        for page in reader.pages:
            content += page.extract_text()
        return content

    def add_document(self, text=None, filename=None, name=None):
        if filename:
            with open(filename, "rb") as file:
                if filename.endswith("pdf"):
                    self.document = self.extract_text_from_pdf(file)
                else:
                    self.document = file.read()
        else:
            if name.lower().endswith("pdf"):
                self.document = self.extract_text_from_pdf(BytesIO(text))
            else:
                self.document = text.decode("utf-8")

        sha1sum = self.get_sha1sum()
        try:
            self.collection = self.client.get_collection(sha1sum)
        except ValueError:
            self.collection = self.client.create_collection(
                name=sha1sum,
                metadata={
                    "hnsw:space": "cosine",
                    "filename": filename or name
                }
            )
            self.index()

    def get_collection(self, sha1sum):
        self.collection = self.client.get_collection(sha1sum)

    def list_collections(self):
        for collection in self.client.list_collections():
            print(collection.metadata["filename"])

    def get_sha1sum(self):
        hash_sha1 = hashlib.sha1()
        hash_sha1.update(self.document.encode("utf-8"))
        return hash_sha1.hexdigest()

    def chunk_text_by_sentence(self, text):
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)

        # Prepare to chunk the text
        chunks = []
        current_chunk = []

        # Current length of the chunk
        current_length = 0

        # Iterate over each sentence
        for sentence in sentences:
            # Check if adding this sentence would exceed the max chunk size
            if current_length + len(sentence) > self.chunk_size:
                # If the current chunk is too large, start a new chunk
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                # Otherwise, add the sentence to the current chunk
                current_chunk.append(sentence)
                current_length += len(sentence)

        # Add the last chunk if it contains any sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def index(self):

        print("Indexing...")

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

    def get_response(self, question, chunks):
        prompt = f"Please answer the following question based only on the provided text and nothing else. Here is the question: {question}."
        for chunk in chunks:
            prompt += f"Here is one chunk of text: {chunk}. "

        from modules.chatbot import ChatBot
        chatbot = ChatBot(self.model_name)
        response = chatbot.send_message_to_model(prompt, {})
        return ChatBot.get_streaming_message(response)

    def query_document(self, query):
        args = {
            "n_results": 3
        }
        if self.use_openai_embeddings:
            embeddings = len_safe_get_embedding(query)
            args["query_embeddings"] = [embeddings]
        else:
            args["query_texts"] = [query]

        results = self.collection.query(**args)

        return ChatBot.get_streaming_message(self.get_response(query, results["documents"]))

    def run(self, force_index=False):

        if force_index:
            self.index()

        while True:
            user_input = input(f"\n\n{MAGENTA}You>{END} ")
            answer = self.query_document(user_input)
            print(f"\n{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--index", help="Index document", action="store_true")
    parser.add_argument("-m", "--model-name", help="The LLM model name", default="gpt-4o-mini")
    parser.add_argument("-o", "--openai", help="Use OpenAI embeddings", action="store_true", default=False)
    parser.add_argument("-f", "--filename", help="The file to query")
    parser.add_argument("-t", "--text", help="The text to query")
    parser.add_argument("-l", "--list", help="List collections", action="store_true")
    args = parser.parse_args()

    force_index = args.index
    model_name = args.model_name
    use_openai_embeddings = args.openai
    filename = args.filename
    text = args.text
    list_collections = args.list

    if filename and text:
        raise ValueError("Error: you cannot specify both a filename and text.")
    if not filename and not text:
        raise ValueError("Error: you must specify either a filename or some text to query.")

    rag = RAG(model_name, use_openai_embeddings=use_openai_embeddings)
    rag.add_document(text=text, filename=filename)

    if list_collections:
        rag.list_collections()
    else:
        rag.run(force_index)

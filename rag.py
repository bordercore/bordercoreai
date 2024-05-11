import argparse
import hashlib
import os
import sys
import uuid

import chromadb
# import nltk
import openai
from aws.index_blobs.lib.embeddings import len_safe_get_embedding
from nltk.tokenize import sent_tokenize

# nltk.download("punkt")


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

    def __init__(self, file, use_openai, force_index):
        self.file = file
        self.use_openai = use_openai
        self.client = chromadb.PersistentClient(path="chromdb")

    def get_collection(self):
        sha1sum = self.sha1sum(self.file)
        try:
            self.collection = self.client.get_collection(sha1sum)
            if force_index:
                self.index(self.file)
        except ValueError:
            self.collection = self.client.create_collection(
                name=sha1sum,
                metadata={
                    "hnsw:space": "cosine",
                    "filename": self.file
                }
            )
            self.index(file)

    def list_collections(self):
        for collection in self.client.list_collections():
            print(collection.metadata["filename"])

    def sha1sum(self, filename):
        hash_sha1 = hashlib.sha1()

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha1.update(chunk)

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

    def index(self, file):

        print("Indexing...")

        with open(file, "r") as file:
            text = file.read()

        chunks = self.chunk_text_by_sentence(text)
        # chunks = self.divide_text_into_chunks(text)

        for chunk in chunks:
            args = {
                "documents": [chunk],
                "ids": [str(uuid.uuid4())],
            }
            if self.use_openai:
                embeddings = len_safe_get_embedding(chunk)
                args["embeddings"] = [embeddings]
            self.collection.add(**args)
        print(f"Added {self.collection.count()} chunks.")

    def get_response(self, question, chunks):
        prompt = f"Please answer the following question based only on the provided chunks of text and nothing else. Here is the question: {question}."
        for chunk in chunks:
            prompt += f"Here is one chunk: {chunk}. "

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        return response["choices"][0]["message"]["content"]

    def query(self):

        if not self.file:
            print("Error: no file specified")
            sys.exit(1)

        self.get_collection()

        while True:
            user_input = input(f"\n\n{MAGENTA}You>{END} ")
            args = {
                "n_results": 3
            }
            if self.use_openai:
                embeddings = len_safe_get_embedding(user_input)
                args["query_embeddings"] = [embeddings]
            else:
                args["query_texts"] = [user_input]

            results = self.collection.query(**args)

            answer = self.get_response(user_input, results["documents"])
            print(f"\n{answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--index", help="Index document", action="store_true")
    parser.add_argument("-o", "--openai", help="Use OpenAI embeddings", action="store_true", default=False)
    parser.add_argument("-f", "--file", help="The file to query")
    parser.add_argument("-l", "--list", help="List collections", action="store_true")
    args = parser.parse_args()

    force_index = args.index
    use_openai = args.openai
    file = args.file
    list_collections = args.list

    rag = RAG(file=file, use_openai=use_openai, force_index=force_index)

    if list_collections:
        rag.list_collections()
    else:
        rag.query()

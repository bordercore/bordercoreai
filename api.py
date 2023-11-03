import json

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()


@app.get("/")
async def root(text):
    return {"embedding": json.dumps(model.encode(text).tolist())}

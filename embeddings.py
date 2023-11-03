import csv
import json
import os

import openai
import requests
import tiktoken
from sentence_transformers import SentenceTransformer

DRF_TOKEN = os.environ.get("DRF_TOKEN")
ELASTICSEARCH_ENDPOINT = os.environ.get("ELASTICSEARCH_ENDPOINT", "localhost")
ELASTICSEARCH_INDEX = os.environ.get("ELASTICSEARCH_INDEX", "bordercore")

ES_INDEX = "embeddings"

# model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = ""


def store_in_elasticsearch(doc_id, embeddings):

    url = f"http://{ELASTICSEARCH_ENDPOINT}:9200/{ELASTICSEARCH_INDEX}/_update/{doc_id}"
    headers = {"Content-Type": "application/json"}

    data = {
        "script": {
            "source": "ctx._source.embeddings = params.value",
            "lang": "painless",
            "params": {
                "value": embeddings
            }
        }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        print(f"Failed to store data. Response from Elasticsearch: {response.content}")
    else:
        print(f"{doc_id} Data stored successfully.")

# def store_in_elasticsearch(index, doc_type, doc_id, data):
#     url = f"http://localhost:9201/{index}/{doc_type}/{doc_id}"

#     headers = {"Content-Type": "application/json"}
#     response = requests.put(url, headers=headers, data=json.dumps(data))

#     if response.status_code != 200:
#         print(f"Failed to store data. Response from Elasticsearch: {response.content}")
#     else:
#         print(f"{doc_id} Data stored successfully.")


def read_uuids(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    # with open(file_path, 'r') as file:
    #     uuids = [line.strip() for line in file]
    # return uuids
    return data

# model.max_seq_length = 200
# print("Max Sequence Length:", model.max_seq_length)

# uuid = "aa6035e5-0c69-4ff2-91ce-f97f2e459576"


def chunk_string(string, chunk_size):
    chunks = [string[i:i + chunk_size] for i in range(0, len(string), chunk_size)]
    return chunks


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_embeddings_openai(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]


def populate(uuids):

    headers = {"Authorization": f"Token {DRF_TOKEN}"}
    session = requests.Session()
    session.trust_env = False

    # uuids = read_uuids("/tmp/uuids.txt")

    for uuid in uuids:
        print(uuid)
        r = session.get(f"https://www.bordercore.com/api/blobs/{uuid}/", headers=headers)

        if r.status_code != 200:
            raise Exception(f"Error when accessing Bordercore REST API: status code={r.status_code}")

        info = r.json()

        embeddings = get_embeddings_openai(info["content"])
        num_tokens = num_tokens_from_string(info["content"], "cl100k_base")
        print(f"token count: {num_tokens}")

        # model = SentenceTransformer("all-MiniLM-L6-v2")
        # chunks = chunk_string(info["content"], model.max_seq_length)

        # embeddings = None
        # for chunk in chunks:
        #     if embeddings is not None:
        #         embeddings = embeddings + model.encode(chunk)
        #     else:
        #         embeddings = model.encode(chunk)

        # print(embeddings.shape)

        if embeddings is not None:
            store_in_elasticsearch(info["uuid"], embeddings)

        # for sentence, embedding in zip(sentences, embeddings):
        #     print(embedding)  # numpy.ndarray
        #     print(embeddings.shape)


def get_query_vector(uuid):

    headers = {"Authorization": f"Token {DRF_TOKEN}"}
    session = requests.Session()
    session.trust_env = False

    r = session.get(f"https://www.bordercore.com/api/blobs/{uuid}/", headers=headers)

    if r.status_code != 200:
        raise Exception(f"Error when accessing Bordercore REST API: status code={r.status_code}")

    info = r.json()

    embeddings = model.encode(info["content"])
    return embeddings.tolist()


def search(uuid):

    query_vector = get_query_vector(uuid)

    body = {
        "_source": ["uuid", "title"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "doc['embeddings_vector'].size() == 0 ? 0 : cosineSimilarity(params.query_vector, 'embeddings_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        },
        "size": 5,
    }

    url = f"http://localhost:9201/{ES_INDEX}/_search"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        print("Failed to execute query. Response from Elasticsearch:")
        print(response.content)
    else:
        return response.json()


populate(["99b1de68-32cc-44ad-bbf8-ca8a3cf4beaa"])
# result = search("ef23dee7-9fea-4a95-902c-105400f0b80a")

# for hit in result["hits"]["hits"]:
#     print(f"{hit['_score']} {hit['_source']['uuid']} {hit['_source']['title']}")

# embeddings = get_embeddings_openai("This is some sample text")
# print(len(embeddings))

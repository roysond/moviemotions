"""Run one query against both models' vectors and compare the top 5."""

import json
import math
import os
import sys

import boto3
from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = 1024
TOP_N = 5

client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def invoke(model_id: str, body: dict) -> dict:
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())


def embed_titan(text: str) -> list[float]:
    payload = invoke(
        os.environ["BEDROCK_MODEL_EMBED_TITAN"],
        {"inputText": text, "dimensions": DIMENSIONS, "normalize": True},
    )
    return payload["embedding"]


def embed_nova(text: str) -> list[float]:
    payload = invoke(
        os.environ["BEDROCK_MODEL_EMBED_NOVA"],
        {
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": DIMENSIONS,
                "text": {"truncationMode": "END", "value": text},
            },
        },
    )
    return payload["embeddings"][0]["embedding"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


PROVIDERS = {"titan": embed_titan, "nova": embed_nova}

query = " ".join(sys.argv[1:]) or "a cosy film about forgiveness"

with open("data/embeddings.json") as handle:
    corpus = json.load(handle)["films"]

print(f'\nQUERY: "{query}"')

for name, embed in PROVIDERS.items():
    query_vector = embed(query)
    ranked = sorted(
        ((cosine(query_vector, film["vectors"][name]), film["title"]) for film in corpus),
        reverse=True,
    )
    print(f"\n--- {name.upper()} ---")
    for position, (score, title) in enumerate(ranked[:TOP_N], start=1):
        print(f"{position}. {score:.4f}  {title}")
"""Embed every film overview with both models and save to disk."""

import json
import os
import time

import boto3
import psycopg
from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = 1024
OUT_PATH = "data/embeddings.json"

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


PROVIDERS = {"titan": embed_titan, "nova": embed_nova}

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    rows = conn.execute(
        "SELECT movie_id, title, overview FROM movies "
        "WHERE overview IS NOT NULL AND overview <> '' ORDER BY id"
    ).fetchall()

print(f"{len(rows)} films to embed")

result = {"dimensions": DIMENSIONS, "films": []}

for index, (film_id, title, overview) in enumerate(rows, start=1):
    entry = {"id": film_id, "title": title, "vectors": {}}
    for name, embed in PROVIDERS.items():
        entry["vectors"][name] = embed(overview)
        time.sleep(0.05)
    result["films"].append(entry)
    print(f"{index}/{len(rows)}  {title}")

os.makedirs("data", exist_ok=True)
with open(OUT_PATH, "w") as handle:
    json.dump(result, handle)

print(f"\nSaved {len(result['films'])} films × 2 models → {OUT_PATH}")
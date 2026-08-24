"""Check both embedding models respond, and report the vector width."""

import json
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = 1024
SAMPLE_TEXT = "A cozy, slow-burning drama about forgiveness in a small town."

client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def call_model(model_id: str, body: dict) -> dict:
    """Send one request to Bedrock and return the parsed response."""
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())


def find_vector(payload):
    """Locate the first long list of numbers, wherever it is nested."""
    if isinstance(payload, list):
        if len(payload) > 10 and all(isinstance(x, (int, float)) for x in payload):
            return payload
        for item in payload:
            found = find_vector(item)
            if found:
                return found
    if isinstance(payload, dict):
        for value in payload.values():
            found = find_vector(value)
            if found:
                return found
    return None


def report(label: str, model_id: str, body: dict) -> None:
    print(f"\n--- {label} ---")
    try:
        payload = call_model(model_id, body)
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        return
    vector = find_vector(payload)
    print("top-level keys:", list(payload.keys()))
    print("vector length :", len(vector) if vector else "NOT FOUND")


report(
    "TITAN",
    os.environ["BEDROCK_MODEL_EMBED_TITAN"],
    {"inputText": SAMPLE_TEXT, "dimensions": DIMENSIONS, "normalize": True},
)

report(
    "NOVA",
    os.environ["BEDROCK_MODEL_EMBED_NOVA"],
    {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": DIMENSIONS,
            "text": {"truncationMode": "END", "value": SAMPLE_TEXT},
        },
    },
)
"""Ask a model to describe a film's mood and themes from its overview."""

import json
import os

import boto3
import psycopg
from dotenv import load_dotenv

load_dotenv()

TITLE = "The Karate Kid"

SYSTEM_PROMPT = """You read a film's plot summary and describe how the film feels.

Return only JSON, no other text.

Rules:
- Base everything on the summary. Do not use outside knowledge of the film.
- "moods" are how watching it feels: tense, cosy, uplifting, bleak, funny.
- "themes" are what it is about underneath the plot: forgiveness, revenge,
  coming of age, loss.
- 3 to 6 entries each. Lowercase. No duplicates between the two lists.

Shape:
{"moods": [], "themes": []}"""

client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    overview = conn.execute(
        "SELECT overview FROM movies WHERE title = %s", (TITLE,)
    ).fetchone()[0]

response = client.converse(
    modelId=os.environ["BEDROCK_MODEL_TEXT"],
    system=[{"text": SYSTEM_PROMPT}],
    messages=[{"role": "user", "content": [{"text": overview}]}],
    inferenceConfig={"maxTokens": 300, "temperature": 0},
)

print(response["output"]["message"]["content"][0]["text"])
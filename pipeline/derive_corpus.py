"""Derive moods and themes for every film, save to disk for review."""

import json
import os
import time

import boto3
import psycopg
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You read a film's plot summary and describe how the film feels.

Return only JSON, no other text.

Rules:
- Base everything on the summary. Do not use outside knowledge of the film.
- Use words specific to THIS film. Avoid generic labels that would fit
  most films.
- Never use these words: tense, coming of age, uplifting, bleak.
- "feel" is one sentence describing the emotional experience of watching it.
- "moods" are 2 to 4 words for how it feels.
- "themes" are 2 to 4 things it is about underneath the plot.
- If the summary does not support something, leave it out. Do not guess.

Shape:
{"feel": "", "moods": [], "themes": []}"""

client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])


def derive(overview: str) -> dict:
    response = client.converse(
        modelId=os.environ["BEDROCK_MODEL_TEXT"],
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": overview}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0},
    )
    return json.loads(response["output"]["message"]["content"][0]["text"])


with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    films = conn.execute(
        "SELECT movie_id, title, overview FROM movies ORDER BY movie_id"
    ).fetchall()

derived = []

for movie_id, title, overview in films:
    tags = derive(overview)
    derived.append({"id": movie_id, "title": title, **tags})
    print(f"\n{title}")
    print(f"   feel  : {tags.get('feel', '')}")
    print(f"   moods : {', '.join(tags.get('moods', []))}")
    print(f"   themes: {', '.join(tags.get('themes', []))}")
    time.sleep(0.1)

with open("data/derived.json", "w") as handle:
    json.dump(derived, handle, indent=2)

print(f"\n{len(derived)} films -> data/derived.json")

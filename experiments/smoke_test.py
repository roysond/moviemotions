"""
Pass 0 smoke test — prove we can reach Bedrock from Python.

Deliberately crude: one file, no structure, no error handling.
Its only job is to answer "do the credentials work and does the
model reply?" We refactor into proper components once it does.
"""

import os
import time

import boto3
from dotenv import load_dotenv

# The ONE place a file is read. Everything else uses os.environ.
# When this moves to a secret manager later, only this line changes.
load_dotenv()

# boto3 finds AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY on its own —
# they are AWS's standard environment variable names.
client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])

SYSTEM_PROMPT = """You convert movie requests into structured search filters.
Return only JSON, no other text.

Rules:
- Return people's FULL names as they appear in film credits.
  "Denzel Washington", never "Denzel".
- If you cannot confidently resolve a name, put it in "unresolved"
  instead of "actors". Do not guess.
- Convert time phrases into minutes.

Shape:
{"actors": [], "unresolved": [], "max_runtime_minutes": null,
 "mood": "", "genres": []}

Example:
Request: "a Tom Hanks film under 90 minutes"
{"actors": ["Tom Hanks"], "unresolved": [], "max_runtime_minutes": 90,
 "mood": "", "genres": []}"""

USER_QUERY = "something cozy with Denzel, nothing over two hours"


def main() -> None:
    wall_start = time.perf_counter()

    response = client.converse(
        modelId=os.environ["BEDROCK_MODEL_TEXT"],
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": USER_QUERY}]}],
        inferenceConfig={"maxTokens": 300, "temperature": 0},
    )

    wall_ms = (time.perf_counter() - wall_start) * 1000

    answer = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]
    model_ms = response["metrics"]["latencyMs"]

    print("\n--- MODEL REPLY ---")
    print(answer)

    print("\n--- MEASUREMENTS ---")
    print(f"input tokens   : {usage['inputTokens']}")
    print(f"output tokens  : {usage['outputTokens']}")
    print(f"model latency  : {model_ms} ms      (Bedrock's own timing)")
    print(f"wall clock     : {wall_ms:.0f} ms   (what a user would feel)")
    print(f"network overhead: {wall_ms - model_ms:.0f} ms")


if __name__ == "__main__":
    main()

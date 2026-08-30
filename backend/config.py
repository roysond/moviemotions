"""Settings. Read once, at import, so a misconfigured machine fails immediately
rather than halfway through a run.

os.environ[...] rather than .get(): a missing DATABASE_URL is not a thing to carry a
default for. os.environ.get(name, default) is used only where a default is genuinely
correct — the rerank model, which is swappable.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = 1024

EVIDENCE_CHARS = 640

MODEL_ID = os.environ["BEDROCK_MODEL_EMBED_NOVA"]

DATABASE_URL = os.environ["DATABASE_URL"]

REGION = os.environ["AWS_REGION"]

RERANK_URL = "https://openrouter.ai/api/v1/rerank"

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

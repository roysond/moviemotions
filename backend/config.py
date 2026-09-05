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

# ─── The agent's model ────────────────────────────────────────────────────────
# Which VENDOR answers, and which MODEL of theirs. Two settings, not one, because
# the model id means nothing without knowing whose catalogue it comes from.
#
# bedrock is the default and the only provider proven in production. vertex is
# switchable so the two can be compared on the same eval — never so that a live
# deployment quietly changes model.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock")

# Per provider, because "the agent's model" is a different string on each. The
# Bedrock one falls back to BEDROCK_MODEL_TEXT: the agent and the build-time
# enrichment share a model until someone deliberately splits them.
AGENT_MODEL_BEDROCK = os.environ.get("BEDROCK_MODEL_AGENT",
                                     os.environ["BEDROCK_MODEL_TEXT"])
AGENT_MODEL_VERTEX = os.environ.get("VERTEX_MODEL_AGENT", "gemini-3.8-flash")

# What is ACTUALLY running. The UI header and every trace report this one, so a
# glance at the page answers "which model wrote this?" without reading config.
AGENT_MODEL = AGENT_MODEL_VERTEX if LLM_PROVIDER == "vertex" else AGENT_MODEL_BEDROCK

# Vertex only. No key: credentials come from Application Default Credentials,
# written outside this project by `gcloud auth application-default login`.
# `global`, not a region — current Gemini models 404 on us-central1.
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

RERANK_URL = "https://openrouter.ai/api/v1/rerank"

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

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

# ─── Three model roles, three switches ────────────────────────────────────────
# There are THREE generation jobs in this application and they want different models.
# One switch cannot serve them: flipping it to reach a better writer would also move
# the agent, which was measured on 2 Sep at 57s and 48K tokens against Nova's 2s and
# 10K for the same question. Three roles, three settings, and none of them can
# silently become another.
#
#   AGENT    live. A person is waiting. This is the REASONER: it chooses tools, reads
#            what came back, and decides which films answer the question. It writes no
#            prose. Speed matters, because the loop may go round more than once.
#
#   WRITER   live. Turns the reasoner's decision into sentences. Chooses nothing, calls
#            no tool, runs exactly once. Only the VOICE matters here.
#
#   DERIVE   build time. Nobody is waiting. Writes each film's mood, theme and
#            premise, once, into the database. Only quality matters here.
#
# Each role picks a PROVIDER, and the provider decides which model id is read. The id
# alone means nothing — "gemini-2.5-pro" is not a thing Bedrock can be asked for.

AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "bedrock")
DERIVE_PROVIDER = os.environ.get("DERIVE_PROVIDER", "vertex")

# The agent, per provider. The Bedrock one falls back to BEDROCK_MODEL_TEXT so a
# machine with only that variable still starts.
AGENT_MODEL_BEDROCK = os.environ.get("BEDROCK_MODEL_AGENT",
                                     os.environ["BEDROCK_MODEL_TEXT"])
AGENT_MODEL_VERTEX = os.environ.get("VERTEX_MODEL_AGENT", "gemini-3.8-flash")

# The deriver, per provider. Vertex defaults to the quality tier rather than a Flash
# model: this job runs offline over the whole corpus, so a slow answer costs nothing
# and a badly written one is stored forever.
DERIVE_MODEL_BEDROCK = os.environ.get("BEDROCK_MODEL_DERIVE",
                                      os.environ["BEDROCK_MODEL_TEXT"])
DERIVE_MODEL_VERTEX = os.environ.get("VERTEX_MODEL_DERIVE", "gemini-2.5-pro")

# What is ACTUALLY running for each role. The UI header and every trace report
# AGENT_MODEL, so a glance at the page answers "which model wrote this?".
AGENT_MODEL = AGENT_MODEL_VERTEX if AGENT_PROVIDER == "vertex" else AGENT_MODEL_BEDROCK
DERIVE_MODEL = DERIVE_MODEL_VERTEX if DERIVE_PROVIDER == "vertex" else DERIVE_MODEL_BEDROCK

# ─── The writer ───────────────────────────────────────────────────────────────
# It DEFAULTS TO THE AGENT'S OWN MODEL, so today one model does both jobs and nothing
# about the deployment changed. That is the point: the split being built here is a
# SEAM, not a model choice. The seam is worth having on its own — the reasoner stops
# being asked to think and perform in one breath — and the day a fine-tuned writer
# exists, WRITER_PROVIDER is the only line that moves.
WRITER_PROVIDER = os.environ.get("WRITER_PROVIDER", AGENT_PROVIDER)
WRITER_MODEL_BEDROCK = os.environ.get("BEDROCK_MODEL_WRITER", AGENT_MODEL_BEDROCK)
WRITER_MODEL_VERTEX = os.environ.get("VERTEX_MODEL_WRITER", AGENT_MODEL_VERTEX)
WRITER_MODEL = (WRITER_MODEL_VERTEX if WRITER_PROVIDER == "vertex"
                else WRITER_MODEL_BEDROCK)

# Vertex only. No key: credentials come from Application Default Credentials, written
# outside this project by `gcloud auth application-default login`.
# `global`, not a region — current Gemini models 404 on us-central1.
GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")

RERANK_URL = "https://openrouter.ai/api/v1/rerank"

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

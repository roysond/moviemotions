"""Every call that leaves this machine to reach a model.

THIS IS THE SEAM. Swapping Bedrock for another embedding provider, or Cohere for
another reranker, means editing this file and nothing else — `retrieval.py` asks for
`embed(text)` and `rerank(query, docs, n)` and does not know or care who answers.

The roadmap called this EmbeddingProvider. It is not an abstract class, because there
is exactly one implementation and inventing an interface for one implementation is its
own smell. It is a module boundary, which is the same idea with less ceremony: the
dependency points inward, and nothing above here names a vendor.
"""

import json
import random
import time

import boto3
import httpx
from botocore.config import Config

from backend.config import (AGENT_MODEL, DIMENSIONS, GCP_LOCATION, GCP_PROJECT,
                            LLM_PROVIDER, MODEL_ID, OPENROUTER_API_KEY, REGION,
                            RERANK_MODEL, RERANK_URL)
from backend.tracing import traceable

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
)

def _hide_vectors(inputs):
    """A 1024-float vector in a trace is noise, not evidence. Log its shape instead."""
    shown = dict(inputs)
    if shown.get("query_vector"):
        shown["query_vector"] = f"<{DIMENSIONS}-dim vector, pre-computed>"
    return shown

def _embedding_shape(output):
    return {"dimensions": len(output) if output else 0}

_pace = 0.0

PACE_MAX = 8.0

THROTTLE_ATTEMPTS = 8

def _invoke_with_backoff(**kwargs):
    """Call Bedrock, backing off exponentially on ThrottlingException."""
    global _pace
    for attempt in range(THROTTLE_ATTEMPTS):
        if _pace:
            time.sleep(_pace)
        try:
            response = _bedrock.invoke_model(**kwargs)
            _pace = max(0.0, _pace * 0.9)          # calm down again once it is flowing
            return response
        except _bedrock.exceptions.ThrottlingException:
            _pace = min(PACE_MAX, max(0.5, _pace * 2))
            wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
            print(f"  [throttled — waiting {wait:.0f}s, pacing at {_pace:.1f}s/call]")
            time.sleep(wait)
    raise RuntimeError(
        f"Bedrock still throttling after {THROTTLE_ATTEMPTS} attempts. "
        "Re-run — finished work is committed and embeddings are cached."
    )

@traceable(run_type="embedding", name="bedrock.embed",
           process_outputs=_embedding_shape)
def embed(text):
    """Turn a piece of text into a vector."""
    response = _invoke_with_backoff(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": DIMENSIONS,
                "text": {"truncationMode": "END", "value": text},
            },
        }),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())["embeddings"][0]["embedding"]

@traceable(run_type="chain", name="cohere.rerank")
def rerank(query, documents, top_n):
    """Reorder documents by reading each against the query. Returns [{index, score}].

    Hosted on OpenRouter (Cohere rerank underneath): the cross-encoder reads the
    query and each document together, so it can tell "creatures hunting people"
    apart from "a tiger in the bathroom" — the thing a single vector cannot.
    """
    response = httpx.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {"index": r["index"], "score": r.get("relevance_score", r.get("relevanceScore"))}
        for r in data["results"]
    ]


def chat_model():
    """The agent's model — the third and last vendor call, and the newest seam.

    WHY THIS FUNCTION EXISTS
        `backend/agent.py` used to build ChatBedrockConverse itself, which made the
        README's claim — that this file is the only one naming a vendor — false for
        weeks. The embedder and the reranker were behind the seam; the model doing
        the actual reasoning was not.

    WHY THE IMPORTS ARE INSIDE THE FUNCTION
        A Bedrock-only deployment must not need the Google packages installed. They
        are ~60MB and the production image has no use for them while LLM_PROVIDER is
        bedrock. Importing at module level would put them in requirements-runtime.txt
        and into every image, for a provider that is not running.

    WHY VERTEX TAKES `vertexai=True` AND NO KEY
        The same class reaches Gemini two ways. With an api_key it uses Google AI
        Studio — a long-lived secret in a file. With vertexai=True and a project it
        uses Vertex AI and Application Default Credentials, which live outside this
        project and rotate. Same model; only one keeps a key out of the repository.
    """
    if LLM_PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(model_id=AGENT_MODEL, region_name=REGION,
                                   temperature=0)

    if LLM_PROVIDER == "vertex":
        if not GCP_PROJECT:
            raise RuntimeError(
                "LLM_PROVIDER=vertex needs GCP_PROJECT in .env, and credentials "
                "from: gcloud auth application-default login")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as missing:                      # optional dependency
            raise RuntimeError(
                "LLM_PROVIDER=vertex needs a package the container does not ship "
                "by default:\n\n    pip install langchain-google-genai\n\n"
                "It is deliberately absent from requirements-runtime.txt — a "
                "Bedrock-only image should not carry the Google libraries."
            ) from missing
        return ChatGoogleGenerativeAI(model=AGENT_MODEL, vertexai=True,
                                      project=GCP_PROJECT, location=GCP_LOCATION,
                                      temperature=0)

    # Fail at startup with the list, rather than at the first question with a
    # NameError. A typo in .env is a configuration error, not a runtime surprise.
    raise ValueError(
        f"LLM_PROVIDER={LLM_PROVIDER!r} is not a provider. Use 'bedrock' or 'vertex'.")

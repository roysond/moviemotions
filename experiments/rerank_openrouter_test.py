"""OpenRouter rerank smoke test — does it answer, and does it demote the tiger?

Sends one query and three plot blurbs to OpenRouter's rerank endpoint. If the
reranker is working, Predator (creatures hunting people) should rank #1 and
The Hangover (a tiger sits in a bathroom, but nothing is hunted) should sink.

Uses httpx, which verifies TLS against certifi's CA bundle — so no macOS
"certificate verify failed" surprise.
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = "https://openrouter.ai/api/v1/rerank"
MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")
QUERY = "a movie where creatures chase and hunt people, very intense"
DOCS = [
    "In the jungle, an elite military team is hunted one by one by a "
    "technologically advanced alien creature that stalks them as prey.",
    "Three friends wake from a bachelor party with no memory, a baby in the "
    "closet and a tiger in the bathroom, and must find their missing friend.",
    "A cowboy doll grows jealous when a spaceman toy arrives, but the two must "
    "cooperate when they are separated from their owner.",
]
LABELS = ["Predator", "The Hangover", "Toy Story"]

try:
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": MODEL, "query": QUERY, "documents": DOCS, "top_n": len(DOCS)},
        timeout=30,
    )
    response.raise_for_status()
except httpx.HTTPStatusError as error:
    print(f"FAILED {error.response.status_code}: {error.response.text[:400]}")
    raise SystemExit(1)
except KeyError:
    print("FAILED: OPENROUTER_API_KEY is not set in .env")
    raise SystemExit(1)

data = response.json()
print("model:", MODEL)
print("top-level keys:", list(data.keys()))
print("---")
for rank, r in enumerate(data.get("results", []), start=1):
    idx = r["index"]
    score = r.get("relevance_score", r.get("relevanceScore"))
    print(f"{rank}. {score:.4f}  {LABELS[idx]}")

"""Measure how spread out each model's vectors are, ignoring any query."""

import json
import math
from itertools import combinations


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


films = json.load(open("data/embeddings.json"))["films"]

for model in ("titan", "nova"):
    scores = [
        cosine(a["vectors"][model], b["vectors"][model])
        for a, b in combinations(films, 2)
    ]
    scores.sort()
    print(
        f"{model:>6}  pairs={len(scores)}  "
        f"min={scores[0]:.3f}  mean={sum(scores)/len(scores):.3f}  max={scores[-1]:.3f}"
    )
"""
gate_effect.py — what the knowledge-graph gate is actually worth, and what it costs.

WHY THIS EXISTS
    On 30 Aug the graph became a pre-filter: genre, actor and director narrow the pool
    before anything is ranked. The note written to celebrate it claimed a film went from
    0.089 to 0.626 "same query, same corpus". It was not the same query — two real
    numbers from two different runs were spliced into one causal claim, and it reached
    two documents before anyone checked. This script is the retraction, done properly.

    ONE variable changes: the gate is off, then on. Same query text, same embedding
    (computed once and passed in), same corpus, same reranker.

    It is built to show HARM as well as help. Case 4 asks a question whose right answers
    span three genres and gates on one of them — if a gate can delete correct answers,
    this is where it will, and a script that could only ever flatter the feature would
    not be worth running.

    python experiments/gate_effect.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.graph import graph_find                    # noqa: E402
from backend.models import embed                        # noqa: E402
from backend.retrieval import search                    # noqa: E402
from backend.tools import genre_key                     # noqa: E402

LIMIT = 5

# query, the gate, and the films a human says genuinely answer the question.
# `expect` is ground truth about the QUESTION, not about the gate — so a gate that
# removes one of these has cost something real, whatever it did to the scores.
CASES = [
    {"name": "gate agrees with the question",
     "query": "a frightening film about being trapped somewhere with no way out",
     "gate": {"genre": "Horror"},
     "expect": ["Get Out", "Alien"]},

    {"name": "gate on a person",
     "query": "a tense, clever film that keeps you guessing",
     "gate": {"director": "Christopher Nolan"},
     "expect": ["Inception", "The Dark Knight"]},

    {"name": "gate narrower than the question",
     "query": "a tense film where people are hunted by something dangerous",
     "gate": {"genre": "Horror"},
     "expect": ["Predator", "Alien", "Jurassic Park"]},

    {"name": "gate on a fact the question never mentioned",
     "query": "a warm, gentle film for a quiet evening at home",
     "gate": {"genre": "Horror"},
     "expect": ["Finding Nemo", "Toy Story"]},
]


def to_search_args(gate):
    """The tool's names -> search()'s names. Genre becomes its graph key."""
    args = dict(gate)
    raw = args.pop("genre", None)
    if raw:
        args["genre_key"] = genre_key(raw)
    return args


def eligible(gate):
    """Which films does this gate leave? Asked of the graph directly, not inferred."""
    found = graph_find(**{k: v for k, v in gate.items()
                          if k in ("genre", "actor", "director")})
    return [f["title"] for f in found["films"]]


def placing(films, title):
    for rank, film in enumerate(films, start=1):
        if film["title"] == title:
            return rank, film["score"]
    return None, None


def show(label, films):
    print(f"  {label}")
    for rank, film in enumerate(films, start=1):
        print(f"    {rank}. {film['title']:<38} {film['score']:.3f}")
    if not films:
        print("    (nothing)")


if __name__ == "__main__":
    print("=" * 78)
    print("GATE EFFECT — one variable: the graph filter off, then on")
    print("=" * 78)

    for case in CASES:
        gate_text = ", ".join(f"{k}={v}" for k, v in case["gate"].items())
        print(f"\n{'─' * 78}\n{case['name'].upper()}")
        print(f'query : "{case["query"]}"')
        print(f"gate  : {gate_text}")
        pool = eligible(case["gate"])
        print(f"pool  : {len(pool)} film(s) survive the gate — {', '.join(sorted(pool))}")

        vector = str(embed(case["query"]))               # embedded ONCE, used by both
        without = search(case["query"], limit=LIMIT, query_vector=vector)
        with_gate = search(case["query"], limit=LIMIT, query_vector=vector,
                           **to_search_args(case["gate"]))

        print()
        show("gate OFF", without)
        show("gate ON ", with_gate)

        print("\n  expected film            gate OFF        gate ON")
        lost = []
        for title in case["expect"]:
            r_off, s_off = placing(without, title)
            r_on, s_on = placing(with_gate, title)
            off = f"#{r_off} {s_off:.3f}" if r_off else "absent"
            on = f"#{r_on} {s_on:.3f}" if r_on else "ABSENT"
            if r_on is None:
                lost.append(title)
            print(f"  {title:<24} {off:<15} {on}")

        if lost:
            print(f"\n  COST: the gate removed {len(lost)} correct answer(s) — "
                  f"{', '.join(lost)}. Deleted before ranking, so no score could save them.")
        else:
            print("\n  No correct answer was lost to the gate.")

    print(f"\n{'=' * 78}")
    print("WHAT THIS DOES AND DOES NOT LICENSE")
    print("=" * 78)
    print("""
  It licenses: a statement about what the gate does to THESE queries on a 20-film
  catalogue, with everything else held still.

  It does NOT license: a general claim that gating improves retrieval. Four cases is
  four cases, and a pool this small is the friendliest possible setting for a filter —
  narrowing 20 films to 2 is not the same problem as narrowing 200,000 to 4,000.

  The rule the last two cases are here to test: gate on facts the USER named, never on
  facts the system inferred. If a gate the user did not ask for deletes a correct
  answer, that is the whole argument, in one line of output.
""")

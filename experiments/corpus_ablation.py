"""What is each corpus actually worth? Leave-one-out ablation over the golden set.

THE QUESTION THIS ANSWERS
    Adding a corpus and watching the score rise tells you it helped ONCE. It does not tell
    you the corpus is still needed — a later addition may have made it redundant, and a
    corpus can help on average while quietly hurting a class of query.

    The honest test is the opposite move: REMOVE it and read the damage.

        "adding X helped"     -> X was useful at the time
        "removing X hurts"    -> X is earning its place TODAY   <-- the one that matters

    That difference is why this file exists. It is also the answer to "why do you still
    have that in your pipeline?" — a question with no good answer except a number.

THE SPECIFIC MYSTERY IT WAS BUILT FOR
    Adding 20 genre chunks lifted recall@3 from 86.2% to 96.6% on a golden set where NOT
    ONE query names a genre. Three answers flipped from miss to hit. Nobody knows which
    three, or why.

    Two competing explanations, and they lead to opposite decisions:

      DIRECT   the genre chunk itself is being retrieved and reranked as evidence.
               -> genre earns its place; keep it.
      EVICTION the 20 genre chunks are pushing weak `derived`/`overview` chunks out of the
               10-slot non-plot quota, and the win is really the removal of noise.
               -> the finding is about `derived` being bad, not genre being good, and the
                  right fix is deleting `derived`, not celebrating genre.

    The discriminator is printed below: for every case that changes, this shows WHICH
    corpus the winning film actually matched on. If the fixed cases match on `genre`, it
    is DIRECT. If they match on `plot` while genre merely exists, it is EVICTION.

ARMS
    all four            the system as it stands today
    - genre             \\
    - derived            }  leave-one-out: the cost of removing each corpus
    - overview          /
    - plot              /
    plot only           the floor — scene text alone, no supporting corpora

    Every arm runs the CHAMPION header configuration (arm D), so the only variable is
    which corpora are allowed into the candidate pool.

COST
    6 arms x 25 queries = 150 rerank calls. Query vectors are embedded ONCE and reused
    across all arms, so 25 embeddings, not 150.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import embed, search  # noqa: E402

GOLDEN = "data/golden_set.json"
TOP_N = 3
CHAMPION = {"variant": "context_header", "header_at_rerank": False}

ALL = ["plot", "overview", "derived", "genre"]
ARMS = [
    ("all four",   ALL),
    ("- genre",    [s for s in ALL if s != "genre"]),
    ("- derived",  [s for s in ALL if s != "derived"]),
    ("- overview", [s for s in ALL if s != "overview"]),
    ("- plot",     [s for s in ALL if s != "plot"]),
    ("plot only",  ["plot"]),
]
BASE = "all four"


def run_arm(cases, sources, vectors):
    hits = expected = ceiling = 0
    quiet, per_case = [], {}
    for case in cases:
        films = search(case["query"], limit=TOP_N, query_vector=vectors[case["id"]],
                       sources=sources, **CHAMPION)
        titles = [f["title"] for f in films]
        top = films[0]["score"] if films else 0.0
        # which corpus the representative chunk of each returned film came from
        srcs = {f["title"]: f["source"] for f in films}
        if case["expect"]:
            found = sum(1 for want in case["expect"] if want in titles)
            hits += found
            expected += len(case["expect"])
            ceiling += min(len(case["expect"]), TOP_N)   # only TOP_N films fit in a top-N
            per_case[case["id"]] = (found, len(case["expect"]), titles, top, srcs)
        else:
            quiet.append(top)
            per_case[case["id"]] = (None, 0, titles, top, srcs)
    return {
        "recall": hits / expected if expected else 0.0,
        "achievable": hits / ceiling if ceiling else 0.0,
        "hits": hits, "expected": expected, "ceiling": ceiling,
        "quiet": sum(quiet) / len(quiet) if quiet else 0.0,
        "per_case": per_case,
    }


def main():
    cases = json.load(open(GOLDEN))["cases"]
    print(f"golden set: {len(cases)} queries, "
          f"{sum(len(c['expect']) for c in cases)} expected answers, "
          f"{sum(1 for c in cases if not c['expect'])} no-answer cases")
    print(f"header config held constant at the champion (arm D): {CHAMPION}\n")

    print("embedding each query once (reused across all arms)...")
    vectors = {}
    for n, case in enumerate(cases, start=1):
        vectors[case["id"]] = str(embed(case["query"]))
        if n % 5 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)}")

    results = {}
    for label, sources in ARMS:
        print(f"\nrunning {label:12} sources={sources}")
        results[label] = run_arm(cases, sources, vectors)

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    print(f"  {'arm':14} {'achievable@3':>13} {'recall@3':>9}  {'hits':>9}   {'quiet@3':>8}")
    print(f"  {'-'*14} {'-'*13} {'-'*9}  {'-'*9}   {'-'*8}")
    for label, r in results.items():
        print(f"  {label:14} {r['achievable']*100:12.1f}% {r['recall']*100:8.1f}%"
              f"  {r['hits']:4}/{r['ceiling']:<4}   {r['quiet']:8.4f}")
    print("\n  achievable@3 = hits / REACHABLE answers — the headline. See eval_variants.py.")

    print("\n" + "=" * 78)
    print("WHAT EACH CORPUS IS WORTH  (answers lost when it is removed)")
    print("=" * 78)
    base = results[BASE]
    print(f"  baseline 'all four': {base['hits']}/{base['expected']} answers\n")
    for label, r in results.items():
        if label == BASE or not label.startswith("- "):
            continue
        cost = base["hits"] - r["hits"]
        verdict = ("EARNS ITS PLACE" if cost > 0 else
                   "REDUNDANT — costs nothing to delete" if cost == 0 else
                   "ACTIVELY HARMFUL — removing it IMPROVES recall")
        sign = "+" if cost < 0 else ""
        print(f"  removing {label[2:]:9} -> {r['hits']:2}/{r['expected']:<3} "
              f"({sign}{-cost:+d} answers)   {verdict}")

    print("\n" + "=" * 78)
    print("THE GENRE QUESTION  (cases where 'all four' and '- genre' differ)")
    print("=" * 78)
    nogen = results["- genre"]
    changed = 0
    for case in cases:
        a = base["per_case"][case["id"]]
        b = nogen["per_case"][case["id"]]
        if a[0] == b[0] and a[2] == b[2]:
            continue
        changed += 1
        print(f"\n  [{case['id']:>2}] {case['type']:9} \"{case['query'][:58]}\"")
        print(f"       expect: {case['expect'] or '(nothing — should stay quiet)'}")
        for name, r in (("all four", a), ("- genre ", b)):
            mark = f"{r[0]}/{r[1]}" if r[0] is not None else f"top={r[3]:.3f}"
            shown = ", ".join(f"{t} [{r[4].get(t, '?')}]" for t in r[2][:3])
            print(f"       {name}  {mark:>7}   {shown}")
    if not changed:
        print("\n  none — the genre corpus changes nothing on this golden set.")
    else:
        print("\n  READ THE [tags]: they name the corpus the winning chunk came from.")
        print("  [genre] on a fixed case  -> DIRECT   (genre is real evidence)")
        print("  [plot] on a fixed case   -> EVICTION (genre won by displacing noise)")

    out = {l: {k: v for k, v in r.items() if k != "per_case"} for l, r in results.items()}
    with open("data/ablation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved data/ablation_results.json")


if __name__ == "__main__":
    main()

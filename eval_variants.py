"""Settle the context-header question by measurement, not argument.

THE QUESTION
    A film-level header ("Finding Nemo (2003). ...his worrisome father Marlin...") fixed
    the retrieval failures. But WHERE does it have to be applied? Baking it into every
    stored chunk cost 31% of the corpus in duplication and diluted every vector. If the
    header only matters at the moment a model READS the text, it can be composed at query
    time from movies.context_header — free, no duplication, no dilution.

THE FOUR ARMS
    Two independent switches: which stored vector we search, and whether the reranker is
    shown the header. That is a 2x2, and all four cells are worth knowing.

                                    reranker sees header?
                                    NO                  YES
        vector = clean              A  no header        C  header only where read
        vector = context_header     D  header in index  B  header everywhere

    A  the honest baseline — the header does nothing
    B  what ran before the migration
    C  the cheap hypothesis — same benefit, none of the storage cost
    D  what ran accidentally after the migration, and scored well

WHAT IS MEASURED
    recall@3   of the expected films, how many appear in the top 3. The honest metric —
               a bare pass/fail hides a 3-of-3 -> 1-of-3 decay.
    quiet@3    on no-answer cases, the top score. Semantic search ALWAYS returns something,
               so the only defence is a score low enough to distinguish "here you go" from
               "nothing here". Lower is better.

COST CONTROL
    A query's vector does not depend on the arm, so each query is embedded ONCE and the
    vector is reused across all four arms — 25 embeddings instead of 100.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import embed, search  # noqa: E402

GOLDEN = "data/golden_set.json"
TOP_N = 3

ARMS = [
    ("A  no header anywhere",      {"variant": "clean",          "header_at_rerank": False}),
    ("B  header everywhere",       {"variant": "context_header", "header_at_rerank": True}),
    ("C  header at rerank only",   {"variant": "clean",          "header_at_rerank": True}),
    ("D  header in index only",    {"variant": "context_header", "header_at_rerank": False}),
]


def run_arm(cases, config, vectors):
    hits = expected = 0
    quiet_scores = []
    per_case = {}
    for case in cases:
        films = search(case["query"], limit=TOP_N,
                       query_vector=vectors[case["id"]], **config)
        titles = [f["title"] for f in films]
        top_score = films[0]["score"] if films else 0.0

        if case["expect"]:
            found = sum(1 for want in case["expect"] if want in titles)
            hits += found
            expected += len(case["expect"])
            per_case[case["id"]] = (found, len(case["expect"]), titles, top_score)
        else:
            quiet_scores.append(top_score)
            per_case[case["id"]] = (None, 0, titles, top_score)
    return {
        "recall": hits / expected if expected else 0.0,
        "hits": hits, "expected": expected,
        "quiet": sum(quiet_scores) / len(quiet_scores) if quiet_scores else 0.0,
        "per_case": per_case,
    }


def main():
    data = json.load(open(GOLDEN))
    cases = data["cases"]
    print(f"golden set: {len(cases)} queries, "
          f"{sum(len(c['expect']) for c in cases)} expected answers, "
          f"{sum(1 for c in cases if not c['expect'])} no-answer cases\n")

    print("embedding each query once (reused across all four arms)...")
    vectors = {}
    for n, case in enumerate(cases, start=1):
        vectors[case["id"]] = str(embed(case["query"]))
        if n % 5 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)}")

    results = {}
    for label, config in ARMS:
        print(f"\nrunning arm {label} ...")
        results[label] = run_arm(cases, config, vectors)

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    print(f"  {'arm':28} {'recall@3':>10}  {'hits':>9}   {'quiet@3':>8}")
    print(f"  {'-'*28} {'-'*10}  {'-'*9}   {'-'*8}")
    best = max(results.values(), key=lambda r: r["recall"])["recall"]
    for label, r in results.items():
        star = "  <-- best" if r["recall"] == best else ""
        print(f"  {label:28} {r['recall']*100:9.1f}%  {r['hits']:4}/{r['expected']:<4}"
              f"   {r['quiet']:8.4f}{star}")
    print("\n  recall@3: higher is better.  quiet@3: LOWER is better "
          "(top score on queries with no right answer).")

    print("\n" + "=" * 78)
    print("WHERE THE ARMS DISAGREE  (cases scored differently by at least one arm)")
    print("=" * 78)
    labels = list(results)
    for case in cases:
        scores = [results[l]["per_case"][case["id"]][0] for l in labels]
        if len(set(str(s) for s in scores)) == 1:
            continue
        print(f"\n  [{case['id']:>2}] {case['type']:9} \"{case['query'][:60]}\"")
        print(f"       expect: {case['expect'] or '(nothing — should stay quiet)'}")
        for l in labels:
            found, total, titles, top = results[l]["per_case"][case["id"]]
            mark = f"{found}/{total}" if found is not None else f"top={top:.3f}"
            print(f"       {l:28} {mark:>8}   {', '.join(titles[:3])}")

    with open("data/variant_results.json", "w") as f:
        json.dump({l: {k: v for k, v in r.items() if k != "per_case"}
                   for l, r in results.items()}, f, indent=2)
    print("\nsaved data/variant_results.json")


if __name__ == "__main__":
    main()

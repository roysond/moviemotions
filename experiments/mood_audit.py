"""Is mood retrieval matching MOOD, or just matching EMOTIONAL DENSITY?

THE SUSPICION
    Arm D returns The Shawshank Redemption at rank 1 for "warm and comforting", "uplifting",
    "sad" AND "funny and light". Three of those score as correct — but only because the human
    ground truth happens to list Shawshank under three moods. On the fourth it is plainly wrong
    and still ranked first.

    If one film tops every emotional query regardless of WHICH emotion, then the system is not
    distinguishing moods at all. It is ranking by how much feeling-laden language a film's text
    contains, and Shawshank's plot — hope, injustice, friendship, freedom, redemption — is the
    densest in the corpus. Some of the score would then be luck, not skill.

    That is a completely different defect from "the corpus has no mood information", and it needs
    a completely different fix. So it gets measured before anything is rewritten.

WHAT THIS PRINTS
    1. Arm D scored on EVERY case, not just the ones where arms disagreed — eval_variants.py
       only shows disagreements, so four cases are currently invisible.
    2. Across the mood cases only, how often each film lands in the top 3. A film appearing in
       most moods is an attractor, not an answer.
    3. The same count for situation/plot cases, as a control: if one film dominates there too,
       the problem is corpus-wide rather than mood-specific.

COST
    One arm, 30 queries — 30 embeddings and 30 rerank calls.

USAGE
    python experiments/mood_audit.py
"""

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import embed, search  # noqa: E402

GOLDEN = "data/golden_set.json"
TOP_N = 3
CHAMPION = {"variant": "context_header", "header_at_rerank": False}   # arm D
ATTRACTOR_AT = 0.5     # in more than half a category's cases = an attractor, not an answer


def main():
    cases = json.load(open(GOLDEN))["cases"]
    print(f"arm D (champion) over all {len(cases)} cases · top {TOP_N}\n")

    rows = []
    for n, case in enumerate(cases, start=1):
        films = search(case["query"], limit=TOP_N,
                       query_vector=str(embed(case["query"])), **CHAMPION)
        titles = [f["title"] for f in films]
        expect = case["expect"]
        found = sum(1 for want in expect if want in titles)
        rows.append({"id": case["id"], "type": case["type"], "query": case["query"],
                     "expect": expect, "titles": titles, "found": found,
                     "reach": min(len(expect), TOP_N),
                     "top_score": films[0]["score"] if films else 0.0})
        if n % 10 == 0:
            print(f"  {n}/{len(cases)}")

    print("\n" + "=" * 92)
    print("EVERY CASE, INCLUDING THE ONES eval_variants.py HIDES")
    print("=" * 92)
    for r in sorted(rows, key=lambda r: (r["type"], r["id"])):
        if r["expect"]:
            mark = f"{r['found']}/{r['reach']}"
            flag = "     " if r["found"] == r["reach"] else ("MISS " if r["found"] == 0 else "part ")
        else:
            mark = f"top={r['top_score']:.3f}"
            flag = "quiet"
        print(f"  {flag} [{r['id']:>2}] {r['type']:9} {mark:>10}  {r['query'][:44]:44} "
              f"-> {', '.join(t[:22] for t in r['titles'])}")

    print("\n" + "=" * 92)
    print("WHO SHOWS UP, AND FOR HOW MANY DIFFERENT QUERIES")
    print("=" * 92)
    for label, kinds in (("MOOD cases", {"mood"}), ("situation / plot (control)", {"situation", "plot"})):
        group = [r for r in rows if r["type"] in kinds and r["expect"]]
        counts = collections.Counter(t for r in group for t in r["titles"])
        print(f"\n  {label} — {len(group)} scorable cases")
        for title, n in counts.most_common(8):
            correct = sum(1 for r in group if title in r["titles"] and title in r["expect"])
            share = n / len(group)
            note = "  <-- ATTRACTOR" if share > ATTRACTOR_AT else ""
            bar = "#" * n
            print(f"    {title:34.34} in {n}/{len(group)} ({share*100:3.0f}%) "
                  f"· right {correct}/{n}  {bar}{note}")

    print("\n" + "=" * 92)
    print("READ IT LIKE THIS")
    print("=" * 92)
    print("""  A film in MOST mood queries is not answering the mood — it is answering "this query has
  feelings in it". Check its `right N/M`: a high count with a low hit rate means the ground
  truth happened to agree with it, not that retrieval distinguished anything.

  If the control group shows the same domination, the problem is corpus-wide emotional density,
  not a mood-specific gap — and the fix is different again.""")

    with open("data/mood_audit.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nsaved data/mood_audit.json")


if __name__ == "__main__":
    main()

"""Did the plot scenes earn their place, or is the premise still doing all the work?

THE QUESTION
    148 scene rows were added and the situation searches are still being won by
    premises. "Scenes do not help" is not a finding — WHICH row of WHICH film scored
    what is. This takes one query and one film and prints every row that film owns,
    scored against it, so the answer is a number rather than an impression.

    One embed call per query; everything else is read from stored vectors.

RUN
    python -m experiments.scene_vs_premise "a prisoner digging a tunnel" --film "The Shawshank Redemption"
    python -m experiments.scene_vs_premise "someone hunted through a jungle"
"""

import argparse

import psycopg

from backend.config import DATABASE_URL
from backend.models import embed
from backend.vectors import EMBED_VARIANT, as_literal

ROWS = """
SELECT m.title, d.data_kind, d.seq, d.content,
       1 - (v.embedding <=> %(query)s::vector) AS score
FROM movie_data d
JOIN movie_vectors v ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
JOIN movies m ON m.movie_id = d.movie_id
WHERE (%(film)s::text IS NULL OR lower(m.title) = lower(%(film)s))
ORDER BY score DESC
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--film", help="one film's every row, in score order")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    vector = as_literal(embed(args.query))

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(ROWS, {"query": vector, "variant": EMBED_VARIANT,
                                   "film": args.film}).fetchall()

    print(f'\nquery: "{args.query}"'
          + (f"   ·   film: {args.film}" if args.film else "   ·   whole catalogue"))
    print(f"{len(rows)} rows scored\n")

    for title, kind, seq, content, score in rows[:args.top]:
        where = f"{kind}" + (f" {seq}" if kind == "plot_scene" else "")
        print(f"  {score:.3f}  {title[:26]:26} {where:14} {content[:78]}…")

    # The comparison that answers the question: for each film, its best scene against
    # its own premise. A scene corpus that never beats the summary it was added to
    # improve on is 148 rows of cost and no benefit.
    best = {}
    for title, kind, seq, content, score in rows:
        slot = best.setdefault(title, {})
        if kind == "premise":
            slot["premise"] = score
        elif kind == "plot_scene":
            slot["scene"] = max(slot.get("scene", 0), score)

    wins = sum(1 for v in best.values()
               if v.get("scene", 0) > v.get("premise", 0))
    print(f"\n  films where the best SCENE beat that film's own PREMISE: "
          f"{wins} of {len(best)}")
    for title, v in sorted(best.items(), key=lambda kv: -kv[1].get("scene", 0))[:6]:
        mark = "scene" if v.get("scene", 0) > v.get("premise", 0) else "premise"
        print(f"    {title[:30]:30} scene {v.get('scene', 0):.3f} · "
              f"premise {v.get('premise', 0):.3f}   -> {mark}")


if __name__ == "__main__":
    main()

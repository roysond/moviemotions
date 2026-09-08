"""Which constraint is actually deciding a "similar to X" result?

THE QUESTION
    search() scores a film as the MINIMUM of its constraints, so one weak constraint
    decides the answer on its own. When a result looks wrong, "the search is wrong" is
    not a finding — WHICH constraint moved it is. This prints the same films ranked
    three ways: by mood alone, by theme alone, and the way search() actually ranks them.

    Reads stored vectors only. No model calls, so it costs nothing and can be run on
    every seed film without thinking about it.

DISPOSABLE
    experiments/ is the folder meant to be deleted. When this has answered its question,
    the answer belongs in docs/decisions.md and the file does not belong anywhere.

RUN
    python -m experiments.similar_breakdown "Toy Story" "Alien" "Get Out"
"""

import sys

import psycopg

from backend.config import DATABASE_URL
from backend.vectors import EMBED_VARIANT

SEED = """
SELECT d.data_kind, v.embedding
FROM movie_data d
JOIN movie_vectors v ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
WHERE d.movie_id = %(movie_id)s AND d.data_kind IN ('mood_feel', 'theme')
"""

FILM = """
SELECT movie_id, title,
       tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies WHERE lower(title) = lower(%(title)s)
"""

SCORES = """
SELECT m.title, 1 - (v.embedding <=> %(query)s::vector) AS score
FROM movie_data d
JOIN movie_vectors v ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
JOIN movies m ON m.movie_id = d.movie_id
WHERE d.data_kind = %(kind)s AND d.movie_id <> %(exclude)s
"""


def ranked(scores, n=5):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def main(titles):
    with psycopg.connect(DATABASE_URL) as conn:
        for title in titles:
            film = conn.execute(FILM, {"title": title}).fetchone()
            if not film:
                print(f"\n{title!r} is not in the catalogue.")
                continue
            movie_id, real_title, _ = film

            seeded = dict(conn.execute(
                SEED, {"movie_id": movie_id, "variant": EMBED_VARIANT}).fetchall())

            by = {}
            for constraint, kind in (("mood", "mood_feel"), ("theme", "theme")):
                rows = conn.execute(SCORES, {"query": seeded[kind], "kind": kind,
                                             "variant": EMBED_VARIANT,
                                             "exclude": movie_id}).fetchall()
                by[constraint] = dict(rows)

            both = {t: min(by["mood"][t], by["theme"][t]) for t in by["mood"]}

            print(f"\n\n=== similar to {real_title}")
            print(f"  {'MOOD only':<34} {'THEME only':<34} {'MIN of both (live)':<34}")
            for i in range(5):
                cells = []
                for scores in (by["mood"], by["theme"], both):
                    name, value = ranked(scores)[i]
                    cells.append(f"{value:.3f} {name[:26]}")
                print("  " + "  ".join(f"{c:<32}" for c in cells))

            # The spread is the only thing that says whether a ranking means anything.
            for label, scores in (("mood", by["mood"]), ("theme", by["theme"])):
                values = sorted(scores.values(), reverse=True)
                print(f"  {label:6} top {values[0]:.3f} · 5th {values[4]:.3f} · "
                      f"lowest {values[-1]:.3f} · spread {values[0] - values[-1]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Toy Story", "Alien", "Get Out", "Home Alone"])

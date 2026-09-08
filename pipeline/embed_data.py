"""Give every row in movie_data a vector, and skip the ones that already have one.

WHAT IT DOES
    Finds rows in movie_data with no vector for the CURRENT embed_variant, embeds each,
    and writes it to movie_vectors. It does not care what kind of text it is looking at,
    so plot scenes will be picked up by this same script, unchanged, the day they land.

WHY IT ASKS THE DATABASE WHAT IS MISSING
    Rather than being told what to embed. The table already knows: a row with no vector
    is the definition of work outstanding. That makes the script resumable by nature —
    interrupt it, run it again, it continues. No list to keep, no state file to go stale.

embed_variant — WHY A VECTOR IS NOT IDENTIFIED BY ITS TEXT ALONE
    An embedding is identified by TEXT x MODEL x WHAT WAS EMBEDDED. Two models produce
    different numbers for the same sentence, and neither is wrong. The variant string
    records which run produced this vector, so a second model can be measured against
    the first with both sets of numbers in the table at once, and a bad new model can be
    deleted by variant rather than by rebuilding everything.

COMMITS PER ROW
    Not once at the end. A rate limit or a lost connection then costs the rows still to
    do, never the ones already paid for.

RUN
    python -m pipeline.embed_data            embed whatever is missing
    python -m pipeline.embed_data --status   count what has a vector and what does not
"""

import argparse
import sys

import psycopg

from backend.config import DATABASE_URL, DIMENSIONS
from backend.models import embed
from backend.vectors import EMBED_VARIANT, as_literal

MISSING = """
SELECT d.data_id, d.data_kind, d.content, m.title
FROM movie_data d
JOIN movies m ON m.movie_id = d.movie_id
WHERE NOT EXISTS (
    SELECT 1 FROM movie_vectors v
    WHERE v.data_id = d.data_id AND v.embed_variant = %(variant)s
)
ORDER BY d.movie_id, d.data_kind, d.seq
"""

INSERT = """
INSERT INTO movie_vectors (data_id, embed_variant, embedding)
VALUES (%(data_id)s, %(variant)s, %(embedding)s::vector)
ON CONFLICT (data_id, embed_variant) DO NOTHING
"""

STATUS = """
SELECT d.data_kind,
       count(*)                                   AS rows,
       count(v.vector_id)                         AS with_vector,
       count(*) - count(v.vector_id)              AS missing
FROM movie_data d
LEFT JOIN movie_vectors v
       ON v.data_id = d.data_id AND v.embed_variant = %(variant)s
GROUP BY d.data_kind
ORDER BY d.data_kind
"""


def status(conn):
    rows = conn.execute(STATUS, {"variant": EMBED_VARIANT}).fetchall()
    print(f"variant  {EMBED_VARIANT}\n")
    print(f"  {'kind':12} {'rows':>6} {'vectors':>8} {'missing':>8}")
    for kind, total, have, missing in rows:
        print(f"  {kind:12} {total:>6} {have:>8} {missing:>8}")
    if not rows:
        print("  movie_data is empty.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true",
                        help="report coverage and write nothing")
    args = parser.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        if args.status:
            status(conn)
            return

        todo = conn.execute(MISSING, {"variant": EMBED_VARIANT}).fetchall()

        if not todo:
            print(f"Nothing to do — every row already has a {EMBED_VARIANT} vector.")
            return

        print(f"{len(todo)} rows to embed · {EMBED_VARIANT}\n")

        done = 0
        for data_id, data_kind, content, title in todo:
            vector = embed(content)

            # A wrong-length vector is refused by the column anyway, but the message
            # would name the column. Say which model returned what, here, instead.
            if len(vector) != DIMENSIONS:
                sys.exit(f"  !! {title} {data_kind}: model returned {len(vector)} "
                         f"numbers, the column holds {DIMENSIONS}. Nothing further "
                         f"written; {done} rows are committed.")

            conn.execute(INSERT, {"data_id": data_id, "variant": EMBED_VARIANT,
                                  "embedding": as_literal(vector)})
            conn.commit()                    # per row: a failure costs only what is left
            done += 1
            print(f"  {done:>3}/{len(todo)}  {title:34} {data_kind}")

        print(f"\nembedded {done}")
        status(conn)


if __name__ == "__main__":
    main()

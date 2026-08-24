"""Settle the genre-in-corpus question by measurement, not argument.

THE QUESTION
    Genre is a label from a closed list (13 values across this catalogue). A label has an
    exact yes/no test, so it belongs in a WHERE clause. But does ALSO putting it in the
    searchable corpus add anything, or does it only take away?

WHY THE HARM TEST RUNS FIRST
    Not one of the 25 golden-set queries names a genre. So this run CANNOT show a benefit —
    there is no query where genre could help. It can only show harm.

    That is deliberate. Run the cheapest test that could KILL the idea before spending
    effort on the test that could support it. If adding a genre corpus damages the 25
    queries we already care about, the idea is dead and no benefit test is needed.

WHAT IS PREDICTED (written down BEFORE running — this is the whole point)
    Recall falls and/or quiet@3 rises. Reason: "Action" is byte-identical text on 8 films,
    so it produces 8 identical vectors with 8 identical scores. The non-plot quota is 10
    slots (other_k). One vaguely action-flavoured query fills 8 of them with ties, and the
    `overview` and `derived` chunks that were actually carrying mood signal get squeezed
    out before the reranker ever sees them.

    This is not a guess about a new system. It is failure #2 from experiments/why_chunk.py
    wearing a different hat: cosine rewards CONCENTRATION, and a bare label is the most
    concentrated chunk it is possible to build.

    If the numbers come back flat or better, the prediction was wrong and the reasoning
    above needs revisiting. Being wrong here is a good outcome — it is information.

HOW GENRE ENTERS THE CORPUS
    As ROWS, not columns. The corpus grows by adding chunks with a new `source_field`.
    `chunk_embeddings` needs no change at all: its grain is (chunk x model x variant), so
    it neither knows nor cares what a chunk is. That is the payoff of the v2 grain work.

    chunk_index is set to 900 — far above any real chunk — so these rows are trivially
    findable and removable, and cannot collide with the (movie_id, chunk_index) unique key.

USAGE
    python experiments/genre_corpus.py --status     # what is in there now
    python experiments/genre_corpus.py --add        # build + embed 20 genre chunks
    python experiments/genre_corpus.py --remove     # undo completely
"""

import argparse
import glob
import json
import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import DATABASE_URL, DIMENSIONS, MODEL_ID, embed  # noqa: E402

SOURCE_FIELD = "genre"
CHUNK_INDEX = 900          # deliberately out of the way of real chunks (plot uses 2..~12)
RAW_GLOB = "data/raw/tmdb_*.json"

INSERT_CHUNK = """
INSERT INTO chunks (movie_id, chunk_index, source_field, content)
VALUES (%(movie_id)s, %(chunk_index)s, %(source_field)s, %(content)s)
ON CONFLICT (movie_id, chunk_index) DO UPDATE SET content = EXCLUDED.content,
                                                  source_field = EXCLUDED.source_field
RETURNING chunk_id
"""

INSERT_EMBEDDING = """
INSERT INTO chunk_embeddings (chunk_id, embedding, model_id, dimensions, embed_variant)
VALUES (%(chunk_id)s, %(embedding)s::vector, %(model_id)s, %(dimensions)s, 'clean')
ON CONFLICT (chunk_id, model_id, embed_variant) DO UPDATE SET embedding = EXCLUDED.embedding
"""


def raw_films():
    """(tmdb_id, title, 'Genre A, Genre B') for every downloaded TMDB record."""
    out = []
    for path in sorted(glob.glob(RAW_GLOB)):
        o = json.load(open(path))
        names = [g["name"] for g in o.get("genres", [])]
        if names:
            out.append((o.get("id"), o.get("title"), ", ".join(names)))
    return out


def movies_has_tmdb_id(conn):
    """Ask the database what columns it actually has instead of assuming."""
    return bool(conn.execute("""
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='movies' AND column_name='tmdb_id'
    """).fetchone())


def resolve_movie_id(conn, by_tmdb, tmdb_id, title):
    """Join by identifier when one exists; fall back to title and say so."""
    if by_tmdb:
        row = conn.execute("SELECT movie_id FROM movies WHERE tmdb_id = %s",
                           (tmdb_id,)).fetchone()
        if row:
            return row[0], "tmdb_id"
    row = conn.execute("SELECT movie_id FROM movies WHERE title = %s", (title,)).fetchone()
    return (row[0] if row else None), "title"


def status(conn):
    rows = conn.execute("""
        SELECT c.source_field,
               count(DISTINCT c.chunk_id)   AS chunks,   -- DISTINCT: a plot chunk has TWO
               count(ce.embedding_id)       AS vectors,  -- variants, so the join doubles it
               round(avg(length(c.content))) AS avg_chars
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.chunk_id
        GROUP BY c.source_field ORDER BY c.source_field
    """).fetchall()
    print(f"  {'source_field':16} {'chunks':>7} {'vectors':>8} {'avg chars':>10}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*10}")
    for field, chunks, vectors, avg in rows:
        mark = "  <-- experiment" if field == SOURCE_FIELD else ""
        print(f"  {field:16} {chunks:7} {vectors:8} {int(avg or 0):10}{mark}")


def add(conn):
    films = raw_films()
    if not films:
        sys.exit(f"no genre data found in {RAW_GLOB}")
    by_tmdb = movies_has_tmdb_id(conn)
    print(f"joining on {'tmdb_id' if by_tmdb else 'title (no tmdb_id column)'}\n")

    added, missed = 0, []
    for tmdb_id, title, genres in films:
        movie_id, how = resolve_movie_id(conn, by_tmdb, tmdb_id, title)
        if movie_id is None:
            missed.append(title)
            print(f"  MISS {title:42.42} not in movies")
            continue
        chunk_id = conn.execute(INSERT_CHUNK, {
            "movie_id": movie_id, "chunk_index": CHUNK_INDEX,
            "source_field": SOURCE_FIELD, "content": genres,
        }).fetchone()[0]
        conn.execute(INSERT_EMBEDDING, {
            "chunk_id": chunk_id, "embedding": str(embed(genres)),
            "model_id": MODEL_ID, "dimensions": DIMENSIONS,
        })
        conn.commit()                       # per-film commit: a throttle costs one film
        added += 1
        print(f"  ok   {title:42.42} [{how}] {genres}")

    print(f"\nadded {added} genre chunks" + (f", {len(missed)} missed" if missed else ""))


def remove(conn):
    n = conn.execute("""
        DELETE FROM chunk_embeddings
        WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE source_field = %s)
    """, (SOURCE_FIELD,)).rowcount
    m = conn.execute("DELETE FROM chunks WHERE source_field = %s",
                     (SOURCE_FIELD,)).rowcount
    conn.commit()
    print(f"removed {m} genre chunks and {n} vectors — database is back to baseline")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--add", action="store_true")
    g.add_argument("--remove", action="store_true")
    g.add_argument("--status", action="store_true")
    args = p.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        if args.add:
            add(conn)
        elif args.remove:
            remove(conn)
        print()
        status(conn)

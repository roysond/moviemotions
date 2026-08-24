"""Schema reorganisation v2 — run once.

FOUR CHANGES
  1. Convention B key naming. A key that joins two tables now has the SAME column name
     in both, so the name carries the grain:
         movies.id            -> movies.movie_id
         chunks.id            -> chunks.chunk_id
         chunk_embeddings.id  -> chunk_embeddings.embedding_id
     (chunks.movie_id and chunk_embeddings.chunk_id already matched — untouched.)

  2. The context header moves OUT of every chunk and INTO the film it belongs to.
     It was stored 145 times (31% of the plot corpus). Now: movies.context_header,
     stored once per film, composed onto a chunk at query time when wanted.

  3. chunk_embeddings gains `embed_variant`, and the uniqueness rule becomes
     (chunk_id, model_id, embed_variant). An embedding is now identified by
     *which chunk x which model x what text was actually embedded* — the third axis
     is what lets the header question be settled by measurement instead of argument.

  4. Existing vectors are RELABELLED, not recomputed. The 145 plot vectors were built
     from header+text, so they are exactly the 'context_header' variant already. The
     40 overview/derived vectors had no header, so they are 'clean'. Only the 145
     'clean' plot vectors are genuinely new work.

SAFETY
  Everything structural runs in ONE transaction — it all lands or none of it does.
  Take a backup first anyway:
      pg_dump "$DATABASE_URL" > backup_before_v2.sql
"""

import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DATABASE_URL, DIMENSIONS, MODEL_ID, embed  # noqa: E402

SEPARATOR = "\n\n"


def already_migrated(conn):
    found = conn.execute("""
        SELECT count(*) FROM information_schema.columns
        WHERE table_schema='public' AND table_name='movies' AND column_name='movie_id'
    """).fetchone()[0]
    return found > 0


def phase_1_structure(conn):
    print("PHASE 1 — structure (single transaction)\n")

    # ---- 1. extract the header from the plot chunks BEFORE stripping it -------------
    # Taken from the data itself rather than recomputed, so the text is guaranteed
    # identical to what was embedded. Verify one header per film first.
    bad = conn.execute("""
        SELECT m.title, count(DISTINCT split_part(c.content, %s, 1))
        FROM chunks c JOIN movies m USING (movie_id)
        WHERE c.source_field = 'plot' AND strpos(c.content, %s) > 0
        GROUP BY m.title HAVING count(DISTINCT split_part(c.content, %s, 1)) <> 1
    """, (SEPARATOR, SEPARATOR, SEPARATOR)).fetchall()
    if bad:
        raise SystemExit(f"ABORT — these films have inconsistent headers: {bad}")
    print("  ✓ every film's plot chunks share one identical header")

    conn.execute("ALTER TABLE movies ADD COLUMN IF NOT EXISTS context_header text")
    moved = conn.execute("""
        UPDATE movies m SET context_header = sub.header
        FROM (
            SELECT c.movie_id, min(split_part(c.content, %s, 1)) AS header
            FROM chunks c
            WHERE c.source_field = 'plot' AND strpos(c.content, %s) > 0
            GROUP BY c.movie_id
        ) sub
        WHERE m.movie_id = sub.movie_id
    """, (SEPARATOR, SEPARATOR)).rowcount
    print(f"  ✓ context_header lifted into movies for {moved} films (stored once each)")

    # ---- 2. strip the header out of the chunks --------------------------------------
    before = conn.execute(
        "SELECT sum(length(content)) FROM chunks WHERE source_field='plot'").fetchone()[0]
    stripped = conn.execute("""
        UPDATE chunks SET content = substr(content, strpos(content, %s) + 2)
        WHERE source_field = 'plot' AND strpos(content, %s) > 0
    """, (SEPARATOR, SEPARATOR)).rowcount
    after = conn.execute(
        "SELECT sum(length(content)) FROM chunks WHERE source_field='plot'").fetchone()[0]
    print(f"  ✓ header removed from {stripped} chunks — "
          f"{before:,} -> {after:,} chars ({100*(before-after)//before}% smaller)")

    # ---- 3. embed_variant, labelled from what was ACTUALLY embedded -----------------
    conn.execute("""
        ALTER TABLE chunk_embeddings
        ADD COLUMN IF NOT EXISTS embed_variant text NOT NULL DEFAULT 'clean'
    """)
    relabelled = conn.execute("""
        UPDATE chunk_embeddings ce SET embed_variant = 'context_header'
        FROM chunks c
        WHERE c.chunk_id = ce.chunk_id AND c.source_field = 'plot'
    """).rowcount
    print(f"  ✓ embed_variant added — {relabelled} plot vectors relabelled "
          f"'context_header', the rest stay 'clean'")

    conn.execute("""
        ALTER TABLE chunk_embeddings
        DROP CONSTRAINT IF EXISTS chunk_embeddings_chunk_id_model_id_key
    """)
    conn.execute("""
        ALTER TABLE chunk_embeddings
        ADD CONSTRAINT chunk_embeddings_chunk_model_variant_key
        UNIQUE (chunk_id, model_id, embed_variant)
    """)
    print("  ✓ uniqueness is now (chunk_id, model_id, embed_variant)")

    # ---- 4. Convention B renames ----------------------------------------------------
    conn.execute("ALTER TABLE movies           RENAME COLUMN id TO movie_id")
    conn.execute("ALTER TABLE chunks           RENAME COLUMN id TO chunk_id")
    conn.execute("ALTER TABLE chunk_embeddings RENAME COLUMN id TO embedding_id")
    print("  ✓ renamed: movies.movie_id · chunks.chunk_id · chunk_embeddings.embedding_id")
    print("    (foreign keys follow the column automatically — nothing to re-point)")


def phase_2_clean_vectors(conn):
    """Embed the 'clean' (header-free) text of every plot chunk. Resumable."""
    print("\nPHASE 2 — embedding the 'clean' plot variant")
    todo = conn.execute("""
        SELECT c.chunk_id, c.content
        FROM chunks c
        WHERE c.source_field = 'plot'
          AND NOT EXISTS (
              SELECT 1 FROM chunk_embeddings e
              WHERE e.chunk_id = c.chunk_id AND e.model_id = %s AND e.embed_variant = 'clean'
          )
        ORDER BY c.chunk_id
    """, (MODEL_ID,)).fetchall()

    print(f"  {len(todo)} chunks still need a 'clean' vector")
    for done, (chunk_id, content) in enumerate(todo, start=1):
        conn.execute("""
            INSERT INTO chunk_embeddings (chunk_id, embedding, model_id, dimensions, embed_variant)
            VALUES (%(chunk_id)s, %(embedding)s::vector, %(model_id)s, %(dimensions)s, 'clean')
            ON CONFLICT (chunk_id, model_id, embed_variant) DO NOTHING
        """, {"chunk_id": chunk_id, "embedding": str(embed(content)),
              "model_id": MODEL_ID, "dimensions": DIMENSIONS})
        conn.commit()                      # per-row: a throttle cannot cost finished work
        if done % 10 == 0 or done == len(todo):
            print(f"    {done}/{len(todo)}")


def report(conn):
    print("\n" + "=" * 70)
    print("AFTER")
    print("=" * 70)
    for variant, field, n in conn.execute("""
        SELECT e.embed_variant, c.source_field, count(*)
        FROM chunk_embeddings e JOIN chunks c ON c.chunk_id = e.chunk_id
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall():
        print(f"  {variant:15} {field:10} {n:5} vectors")
    total, = conn.execute("SELECT count(*) FROM chunk_embeddings").fetchone()
    print(f"  {'':15} {'TOTAL':10} {total:5}")
    hdr, = conn.execute(
        "SELECT count(*) FROM movies WHERE context_header IS NOT NULL").fetchone()
    print(f"\n  films with a stored context_header: {hdr}")
    leftover, = conn.execute(
        "SELECT count(*) FROM chunks WHERE position(%s in content) > 0", (SEPARATOR,)
    ).fetchone()
    print(f"  chunks still containing an inline header: {leftover}"
          f"{'  <-- expected 0' if leftover else '  ✓'}")


with psycopg.connect(DATABASE_URL) as conn:
    if already_migrated(conn):
        print("Structure already migrated — running phase 2 only.\n")
    else:
        phase_1_structure(conn)
        conn.commit()
        print("\n  ✓ phase 1 committed")
    phase_2_clean_vectors(conn)
    report(conn)

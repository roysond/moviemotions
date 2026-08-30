"""Read-only audit of the MovieMotions database — the ground truth before reorganising.

Prints schema, row counts, constraints, indexes, storage, and data-integrity checks.
Nothing is modified. Run from the repo root:  python experiments/db_audit.py
"""

import os
import sys

import psycopg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import DATABASE_URL  # noqa: E402

def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

with psycopg.connect(DATABASE_URL) as conn:

    rule("1 · TABLES AND COLUMNS")
    rows = conn.execute("""
        SELECT table_name, ordinal_position, column_name, data_type,
               COALESCE(character_maximum_length::text, '') AS len,
               is_nullable, COALESCE(column_default, '') AS dflt
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """).fetchall()
    current = None
    for t, pos, col, typ, ln, nullable, dflt in rows:
        if t != current:
            print(f"\n  ── {t}")
            current = t
        null = "" if nullable == "YES" else "  NOT NULL"
        d = f"  default={dflt[:30]}" if dflt else ""
        print(f"     {pos:2}. {col:18} {typ}{('('+ln+')') if ln else ''}{null}{d}")

    rule("2 · ROW COUNTS AND SIZE ON DISK")
    tables = [r[0] for r in conn.execute("""
        SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename
    """).fetchall()]
    for t in tables:
        n = conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (t,)).fetchone()[0]
        print(f"  {t:20} {n:6} rows   {size:>10}")

    rule("3 · CONSTRAINTS (what the database itself guarantees)")
    for t, name, kind, definition in conn.execute("""
        SELECT rel.relname, con.conname, con.contype, pg_get_constraintdef(con.oid)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = rel.relnamespace
        WHERE n.nspname = 'public'
        ORDER BY rel.relname, con.contype
    """).fetchall():
        label = {"p": "PRIMARY KEY", "u": "UNIQUE", "f": "FOREIGN KEY", "c": "CHECK"}.get(kind, kind)
        print(f"  {t:18} {label:12} {definition}")

    rule("4 · INDEXES (what makes lookups fast)")
    for t, name, definition in conn.execute("""
        SELECT tablename, indexname, indexdef FROM pg_indexes
        WHERE schemaname='public' ORDER BY tablename, indexname
    """).fetchall():
        kind = "VECTOR" if ("hnsw" in definition or "ivfflat" in definition) else ""
        print(f"  {t:18} {name:34} {kind}")
        print(f"      {definition[:110]}")

    rule("5 · CHUNKS BY SOURCE TYPE")
    for field, n, mn, avg, mx in conn.execute("""
        SELECT source_field, count(*), min(length(content)),
               round(avg(length(content))), max(length(content))
        FROM chunks GROUP BY source_field ORDER BY source_field
    """).fetchall():
        print(f"  {field:10} {n:4} rows   length min {mn:5}  avg {int(avg):5}  max {mx:5}")

    rule("6 · EMBEDDINGS BY MODEL AND VARIANT")
    for model, variant, n, dims in conn.execute("""
        SELECT model_id, embed_variant, count(*), max(dimensions) FROM chunk_embeddings
        GROUP BY model_id, embed_variant ORDER BY 1, 2
    """).fetchall():
        print(f"  {model:44} {variant:15} {n:5} vectors  {dims}d")

    rule("7 · DATA INTEGRITY CHECKS")
    checks = [
        ("chunks with no embedding",
         "SELECT count(*) FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id=c.chunk_id WHERE e.embedding_id IS NULL"),
        ("embeddings pointing at a missing chunk",
         "SELECT count(*) FROM chunk_embeddings e LEFT JOIN chunks c ON c.chunk_id=e.chunk_id WHERE c.chunk_id IS NULL"),
        ("movies with zero chunks",
         "SELECT count(*) FROM movies m LEFT JOIN chunks c ON c.movie_id=m.movie_id WHERE c.chunk_id IS NULL"),
        ("empty or whitespace-only content",
         "SELECT count(*) FROM chunks WHERE content IS NULL OR btrim(content)=''"),
        ("duplicate content strings",
         "SELECT count(*) FROM (SELECT content FROM chunks GROUP BY content HAVING count(*)>1) d"),
        ("plot chunks with a header wrongly inlined (want 0)",
         "SELECT count(*) FROM chunks WHERE source_field='plot'"
         " AND position(chr(10)||chr(10) in content) > 0"),
        ("films missing a context_header (want 0)",
         "SELECT count(*) FROM movies WHERE context_header IS NULL OR btrim(context_header)=''"),
        ("plot chunks missing the 'clean' vector (want 0)",
         "SELECT count(*) FROM chunks c WHERE c.source_field='plot' AND NOT EXISTS"
         " (SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id=c.chunk_id"
         "  AND e.embed_variant='clean')"),
        ("plot chunks missing the 'context_header' vector (want 0)",
         "SELECT count(*) FROM chunks c WHERE c.source_field='plot' AND NOT EXISTS"
         " (SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id=c.chunk_id"
         "  AND e.embed_variant='context_header')"),
    ]
    for label, sql in checks:
        n = conn.execute(sql).fetchone()[0]
        flag = "  <-- look at this" if n else ""
        print(f"  {label:46} {n:5}{flag}")

    rule("8 · HEADER STORAGE — deduplicated in migration v2")
    plot_chars, nplot = conn.execute(
        "SELECT sum(length(content)), count(*) FROM chunks WHERE source_field='plot'").fetchone()
    hdr_chars, nfilms = conn.execute(
        "SELECT sum(length(context_header)), count(*) FROM movies"
        " WHERE context_header IS NOT NULL").fetchone()
    if plot_chars and hdr_chars:
        avg_hdr = hdr_chars / nfilms
        would_have = int(avg_hdr * nplot)
        print(f"  plot scene text          {plot_chars:8} chars across {nplot} chunks")
        print(f"  headers stored ONCE      {hdr_chars:8} chars across {nfilms} films")
        print(f"  if inlined per chunk     {would_have:8} chars"
              f"  ({100*would_have//plot_chars}% overhead avoided)")

    rule("9 · SAMPLE ROW (Finding Nemo, first plot chunk)")
    row = conn.execute("""
        SELECT c.chunk_id, c.chunk_index, c.source_field, c.content
        FROM chunks c JOIN movies m USING (movie_id)
        WHERE m.title='Finding Nemo' AND c.source_field='plot'
        ORDER BY c.chunk_index LIMIT 1
    """).fetchone()
    if row:
        cid, idx, field, content = row
        print(f"  id={cid}  chunk_index={idx}  source_field={field}  len={len(content)}")
        print("  ---")
        for line in content.split("\n"):
            print(f"  | {line[:100]}")

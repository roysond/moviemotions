"""Diagnostic: which chunk actually wins, and does the plot corpus ever get seen?

The eval barely moved after loading 136 plot chunks. Before changing anything, find out
where the plot text is being lost. Three questions, answered from the database:

  A. For a query, which source_field wins the per-film DISTINCT ON? (overview/derived/plot)
  B. For one film we care about, how do ALL its chunks rank against that query?
  C. Does the target film even reach the reranker's candidate list (top candidate_k)?
"""

import os
import sys

import psycopg

# this file lives in experiments/, so put the repo root on the import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DATABASE_URL, MODEL_ID, embed  # noqa: E402

CANDIDATE_K = 10

QUERIES = [
    ("a father and son separated and trying to find each other", "Finding Nemo"),
    ("movie that has creatures chasing you and is very intense", "Jurassic Park"),
    ("an underdog who trains hard to win a fight", "The Karate Kid"),
]

BEST_PER_FILM = """
SELECT title, source_field, chunk_index, similarity, content
FROM (
    SELECT DISTINCT ON (m.movie_id)
           m.movie_id, m.title, c.source_field, c.chunk_index, c.content,
           1 - (ce.embedding <=> %(q)s::vector) AS similarity
    FROM chunk_embeddings ce
    JOIN chunks c USING (chunk_id)
    JOIN movies m USING (movie_id)
    WHERE ce.model_id = %(model_id)s
    ORDER BY m.movie_id, ce.embedding <=> %(q)s::vector
) best
ORDER BY similarity DESC
LIMIT %(limit)s
"""

# Report each chunk's RANK WITHIN ITS SOURCE TYPE — that is what the quota filters on,
# so it answers directly: would plot_k=30 have let this chunk through?
ALL_CHUNKS_FOR_FILM = """
SELECT source_field, chunk_index, content, similarity, rank_in_field
FROM (
    SELECT m.title, c.source_field, c.chunk_index, c.content,
           1 - (ce.embedding <=> %(q)s::vector) AS similarity,
           ROW_NUMBER() OVER (PARTITION BY c.source_field
                              ORDER BY ce.embedding <=> %(q)s::vector) AS rank_in_field
    FROM chunk_embeddings ce
    JOIN chunks c USING (chunk_id)
    JOIN movies m USING (movie_id)
    WHERE ce.model_id = %(model_id)s
) ranked
WHERE title = %(title)s
ORDER BY similarity DESC
"""

with psycopg.connect(DATABASE_URL) as conn:
    print("=" * 78)
    print("SANITY — what is actually stored")
    for row in conn.execute(
        "SELECT c.source_field, count(*), min(length(c.content)), max(length(c.content)) "
        "FROM chunks c JOIN chunk_embeddings ce ON ce.chunk_id = c.chunk_id "
        "WHERE ce.model_id = %s GROUP BY 1 ORDER BY 1", (MODEL_ID,)
    ).fetchall():
        print(f"   {row[0]:9} count={row[1]:4}  len min={row[2]} max={row[3]}")

    for query, target in QUERIES:
        vector = str(embed(query))
        params = {"q": vector, "model_id": MODEL_ID}

        print("\n" + "=" * 78)
        print(f"QUERY: {query!r}")
        print(f"TARGET FILM: {target}")

        print(f"\n  A · top {CANDIDATE_K} films by vector (what the reranker receives)")
        rows = conn.execute(BEST_PER_FILM, {**params, "limit": CANDIDATE_K}).fetchall()
        target_in_candidates = False
        for rank, (title, field, idx, sim, content) in enumerate(rows, 1):
            mark = "  <-- TARGET" if title == target else ""
            if title == target:
                target_in_candidates = True
            print(f"   {rank:2}. {sim:.4f}  [{field:8} #{idx}]  {title}{mark}")

        print(f"\n  B · every chunk of {target} — with its rank INSIDE its source type")
        print(f"      (quota admits plot rank<=30, derived/overview rank<=10)")
        for field, idx, content, sim, rank_in_field in conn.execute(
            ALL_CHUNKS_FOR_FILM, {**params, "title": target}
        ).fetchall():
            quota = 30 if field == "plot" else 10
            admitted = "ADMITTED" if rank_in_field <= quota else "  cut   "
            snippet = " ".join(content.split())[:74]
            print(f"      {sim:.4f}  rank {rank_in_field:>3} of {field:8} {admitted}  {snippet}...")

        print(f"\n  C · verdict: target {'IS' if target_in_candidates else 'IS NOT'} "
              f"in the reranker's candidate list")

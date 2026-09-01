"""The vector half: find candidate chunks, rerank them, collapse them to films.

Knows nothing about the knowledge graph, and the graph knows nothing about it. They
share a database URL and nothing else, which is why they are two files.
"""

import psycopg

from backend.config import DATABASE_URL, EVIDENCE_CHARS, MODEL_ID
from backend.models import _hide_vectors, embed, rerank
from backend.tracing import traceable

SEARCH_SQL = """
SELECT movie_id, title, release_date, runtime_minutes, context_header,
       source_field, chunk_index, content, similarity
FROM (
    SELECT m.movie_id, m.title, m.release_date, m.runtime_minutes, m.context_header,
           c.source_field, c.chunk_index, c.content,
           1 - (ce.embedding <=> %(q)s::vector) AS similarity,
           ROW_NUMBER() OVER (PARTITION BY c.source_field
                              ORDER BY ce.embedding <=> %(q)s::vector) AS rank_in_field
    FROM chunk_embeddings ce
    JOIN chunks c USING (chunk_id)
    JOIN movies m USING (movie_id)
    WHERE ce.model_id = %(model_id)s
      -- only plot chunks have a header variant; the others are always 'clean'
      AND ce.embed_variant = CASE WHEN c.source_field = 'plot'
                                  THEN %(variant)s ELSE 'clean' END
      -- HARD CONSTRAINTS. One static query, four optional filters: NULL means "the user
      -- did not ask", so the clause collapses to TRUE and nothing is excluded. The SQL
      -- string is never rebuilt by gluing text together, which is how injection happens.
      -- Applied HERE, inside the inner SELECT, so ROW_NUMBER() ranks only eligible films:
      -- filter before you budget, never after.
      -- A film with an unknown runtime or date fails the comparison and is dropped. That
      -- is deliberate: a hard constraint must be provably true, not merely not-disproved.
      AND (%(max_runtime)s::int IS NULL OR m.runtime_minutes <= %(max_runtime)s::int)
      AND (%(min_runtime)s::int IS NULL OR m.runtime_minutes >= %(min_runtime)s::int)
      AND (%(after_year)s::int  IS NULL
           OR EXTRACT(YEAR FROM m.release_date) >= %(after_year)s::int)
      AND (%(before_year)s::int IS NULL
           OR EXTRACT(YEAR FROM m.release_date) <= %(before_year)s::int)
      -- EXCLUDE A NAMED FILM. "something like Jurassic Park" must not return Jurassic
      -- Park: the user already named it and asked to move PAST it. "is this that film"
      -- has a yes/no test, so it is a hard constraint and belongs here, not in the
      -- embedding. Loose match too, so "Terminator 2" also excludes the full title.
      AND (%(exclude_title)s::text IS NULL
           OR NOT (lower(m.title) = lower(%(exclude_title)s::text)
                   OR lower(m.title) LIKE lower(%(exclude_loose)s::text)))
      -- GRAPH FACTS AS A GATE. Genre, cast and crew are not columns on `movies` — a film
      -- has MANY of each, so they live as edges. That is the only reason they were not
      -- filters until now: the columns were easy and the edges needed a join. An EXISTS
      -- is that join, and it belongs in exactly the same place as the runtime checks
      -- above — inside the inner SELECT, so ROW_NUMBER() budgets its 30/10 quota over
      -- eligible films only. Filter, then budget, then rank.
      AND (%(genre_key)s::text IS NULL OR EXISTS (
             SELECT 1 FROM graph_edges ge
             JOIN graph_nodes gf ON gf.node_key = ge.from_key
             WHERE ge.edge_type = 'HAS_GENRE'
               AND ge.to_key = %(genre_key)s::text
               AND (gf.properties->>'movie_id')::int = m.movie_id))
      AND (%(actor)s::text IS NULL OR EXISTS (
             SELECT 1 FROM graph_edges ge
             JOIN graph_nodes gp ON gp.node_key = ge.from_key
             JOIN graph_nodes gf ON gf.node_key = ge.to_key
             WHERE ge.edge_type = 'ACTED_IN'
               AND lower(gp.name) LIKE lower(%(actor_loose)s::text)
               AND (gf.properties->>'movie_id')::int = m.movie_id))
      AND (%(director)s::text IS NULL OR EXISTS (
             SELECT 1 FROM graph_edges ge
             JOIN graph_nodes gp ON gp.node_key = ge.from_key
             JOIN graph_nodes gf ON gf.node_key = ge.to_key
             WHERE ge.edge_type = 'DIRECTED'
               AND lower(gp.name) LIKE lower(%(director_loose)s::text)
               AND (gf.properties->>'movie_id')::int = m.movie_id))
      -- CORPUS DIAL. NULL = every source type allowed, which is normal operation.
      -- Passing a list restricts the pool. This is an INSTRUMENT, not a feature: it is how
      -- we measure what each corpus is worth — run the golden set with one source type
      -- removed and read the damage. Same placement rule as the hard constraints:
      -- ROW_NUMBER() must rank only rows that are actually eligible.
      AND (%(sources)s::text[] IS NULL OR c.source_field = ANY(%(sources)s::text[]))
) ranked
WHERE (source_field = 'plot'  AND rank_in_field <= %(plot_k)s)
   OR (source_field <> 'plot' AND rank_in_field <= %(other_k)s)
ORDER BY similarity DESC
"""

GET_FILM_SQL = """
SELECT m.title, m.release_date, m.runtime_minutes,
       (SELECT c.content FROM chunks c
        WHERE c.movie_id = m.movie_id AND c.source_field = 'overview'
        ORDER BY c.chunk_index LIMIT 1) AS overview
FROM movies m
WHERE lower(m.title) = lower(%(exact)s)
   OR lower(m.title) LIKE lower(%(loose)s)
ORDER BY (lower(m.title) = lower(%(exact)s)) DESC, length(m.title)
LIMIT 5
"""

@traceable(run_type="retriever", name="retrieval.get_film")
def get_film(title):
    """Look a film up BY NAME. No embedding, no reranker, no vectors at all.

    The counterpart to search(): that one answers "what matches this feeling?", this one
    answers "what do you know about THIS film?". Two questions, two mechanisms —
    semantic similarity cannot do exact identity, and exact identity cannot do mood.

    Matching is forgiving because people type "Terminator 2", not "Terminator 2: Judgment
    Day". An exact case-insensitive hit sorts first; substring matches follow, shortest
    title first, so "Alien" beats a longer title that merely contains it.

    The LIKE pattern is built in PYTHON and passed as a parameter. Never glue it into the
    SQL string — that is both an injection hole and, in psycopg, a placeholder error.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(GET_FILM_SQL, {
            "exact": title, "loose": f"%{title}%",
        }).fetchall()
    return [
        {
            "title": row_title,
            "release_date": str(release_date) if release_date else None,
            "runtime_minutes": runtime,
            "overview": overview,
        }
        for row_title, release_date, runtime, overview in rows
    ]

def _document_for_rerank(chunk, header_at_rerank):
    """The text the cross-encoder reads — not necessarily the text that was embedded.

    The header lives once on the film (movies.context_header), so it can be composed onto
    a chunk at query time for free. Storing it inside 145 chunks cost 31% of the corpus in
    duplication and diluted every vector; composing it here costs nothing and dilutes
    nothing.
    """
    if header_at_rerank and chunk.get("context_header") and chunk["source"] == "plot":
        return f"{chunk['context_header']}\n\n{chunk['content']}"
    return chunk["content"]


def _chunk_count(inputs):
    """~50 chunks with their full text are already in the rerank span above this one.

    Logging them again here would double the payload and tell you nothing new. What is
    worth seeing at this step is how many went in and how many films came out.
    """
    shown = {k: v for k, v in inputs.items() if k != "chunks"}
    shown["chunks"] = f"{len(inputs.get('chunks') or [])} ranked chunk(s)"
    return shown


@traceable(run_type="chain", name="retrieval.collapse_to_films",
           process_inputs=_chunk_count)
def _collapse_to_films(chunks, limit, pool=3):
    """Many ranked chunks in → one row per film out, scored by BREADTH of evidence.

    Not max-pooling. Taking each film's single best chunk rewards one lucky scene:
    Finding Nemo has a barracuda attack, so exactly one chunk looks like an answer to
    "creatures chasing people" — and it then outranks Predator, which is about nothing
    else. A film genuinely about a theme has SEVERAL strong chunks.

    So a film's score sums its top `pool` chunk scores. Later chunks are damped (1, 1/2,
    1/3) so the best chunk still dominates and a film is never rewarded merely for being
    long — depth of the top match first, corroboration second.

    The representative chunk stays the film's best one; only the RANKING changes.
    """
    by_film, order = {}, []
    for chunk in chunks:
        film_id = chunk["movie_id"]
        if film_id not in by_film:
            by_film[film_id] = []
            order.append(film_id)
        by_film[film_id].append(chunk)

    films = []
    for film_id in order:
        found = by_film[film_id]                      # already in rank order
        top = found[:pool]
        aggregate = sum(c["score"] / (rank + 1) for rank, c in enumerate(top))
        best = dict(found[0])
        best["score"] = round(aggregate, 4)
        best["best_chunk_score"] = found[0]["score"]
        best["supporting_chunks"] = len(found) - 1
        films.append(best)

    films.sort(key=lambda f: f["score"], reverse=True)
    return films[:limit]

@traceable(run_type="retriever", name="retrieval.search",
           process_inputs=_hide_vectors)
def search(query, limit=3, use_rerank=True, plot_k=30, other_k=10,
           variant="context_header", header_at_rerank=False, query_vector=None,
           max_runtime=None, min_runtime=None, after_year=None, before_year=None,
           exclude_title=None, genre_key=None, actor=None, director=None, sources=None):
    """Retrieve chunks wide -> rerank -> aggregate per film -> collapse to films.

    DEFAULTS ARE ARM D, CHOSEN BY MEASUREMENT (eval_variants.py, 25-query golden set):

        arm                        recall@3   quiet@3   query-time cost
        A  no header anywhere        75.9%     0.2060   none
        B  header everywhere         89.7%     0.2543   compose onto ~50 candidates
        C  header at rerank only     82.8%     0.2754   compose onto ~50 candidates
        D  header in index only      86.2%     0.2232   none          <-- default

    Isolating each factor: putting the header in the STORED VECTOR is worth +7 to +10
    recall points (A->D, C->B). Adding it at RERANK time is worth less (+3.5 to +6.9)
    and costs the most false confidence (A->C raises quiet@3 by 0.069).

    B leads D by 3.5 points, but with 29 expected answers one answer IS 3.4 points —
    and the entire gap is a single lucky hit on case 21, a query both arms failed
    (1/4 vs 0/4). Not a result. D ties B on every other disagreement, is calmer when
    wrong, and costs nothing at query time.

    plot_k / other_k are per-source-type quotas, not one global top-N: without them an
    abstract query fills the whole pool with short mood chunks and the scene text never
    reaches the judge. Several chunks of the same film compete — that is the point.

    HARD CONSTRAINTS (max_runtime, min_runtime, after_year, before_year) are checked in
    SQL, never by the embedding. Vectors capture topic, not truth value: "under 2 hours"
    embeds as mood and "not a cartoon" embeds NEXT TO "a cartoon", because there is no
    minus sign in vector space. Anything with a yes/no test belongs in a WHERE clause.
    The model's job is to notice the constraint was asked for; the database's job is to
    enforce it.

    sources restricts which corpora may enter the candidate pool (None = all of them).
    Only an experiment should pass it. Leave-one-out ablation is the only honest way to
    answer "is this corpus earning its place?" — adding a corpus and watching the score
    rise says it helped; removing it and watching the score fall says it is still needed.

    query_vector may be passed in already embedded. A query's vector does not depend on
    which arm is being tested, so an experiment that runs four arms embeds each query
    ONCE instead of four times.

    If reranking fails, fall back to vector order and say so.
    """
    if query_vector is None:
        query_vector = str(embed(query))

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(SEARCH_SQL, {
            "q": query_vector, "model_id": MODEL_ID,
            "variant": variant, "plot_k": plot_k, "other_k": other_k,
            "max_runtime": max_runtime, "min_runtime": min_runtime,
            "after_year": after_year, "before_year": before_year,
            "exclude_title": exclude_title,
            "exclude_loose": f"%{exclude_title}%" if exclude_title else None,
            "genre_key": genre_key,
            "actor": actor,
            "actor_loose": f"%{actor}%" if actor else None,
            "director": director,
            "director_loose": f"%{director}%" if director else None,
            "sources": sources,
        }).fetchall()

    chunks = [
        {
            "movie_id": movie_id,
            "title": title,
            "release_date": str(release_date) if release_date else None,
            "runtime_minutes": runtime,
            "context_header": context_header,
            "source": source_field,
            "chunk_index": chunk_index,
            "content": content,
            "vector_similarity": round(similarity, 4),
            "score": round(similarity, 4),
            "method": "vector",
        }
        for (movie_id, title, release_date, runtime, context_header,
             source_field, chunk_index, content, similarity) in rows
    ]

    if use_rerank and chunks:
        documents = [_document_for_rerank(c, header_at_rerank) for c in chunks]
        try:
            ranked = rerank(query, documents, top_n=len(documents))
            chunks = [
                {**chunks[r["index"]], "score": round(r["score"], 4), "method": "rerank"}
                for r in ranked
            ]
        except Exception as error:
            print(f"  [rerank unavailable: {type(error).__name__} — falling back to vector]")

    films = _collapse_to_films(chunks, limit)
    for film in films:
        # KEEP the matched text, trimmed. It used to be discarded, and that quietly made
        # the system ungrounded: the tool reported WHICH film matched and WHERE, but never
        # the words that matched — so anything the model then said about the film came out
        # of its own training, not out of this database. Retrieval you do not pass on is
        # retrieval you did not do.
        film["evidence"] = " ".join((film.pop("content", "") or "").split())[:EVIDENCE_CHARS]
        for key in ("movie_id", "context_header"):
            film.pop(key, None)
    return films

_EXCLUDED_SQL = """
SELECT m.title, m.runtime_minutes, EXTRACT(YEAR FROM m.release_date)::int AS year
FROM movies m
WHERE m.source = 'tmdb'
  AND ((%(max_runtime)s::int IS NOT NULL AND m.runtime_minutes > %(max_runtime)s::int)
    OR (%(min_runtime)s::int IS NOT NULL AND m.runtime_minutes < %(min_runtime)s::int)
    OR (%(after_year)s::int  IS NOT NULL
        AND EXTRACT(YEAR FROM m.release_date) < %(after_year)s::int)
    OR (%(before_year)s::int IS NOT NULL
        AND EXTRACT(YEAR FROM m.release_date) > %(before_year)s::int))
ORDER BY m.title
"""

@traceable(run_type="chain", name="retrieval.excluded_by_filters")
def excluded_by_filters(max_runtime=None, min_runtime=None,
                        after_year=None, before_year=None):
    """Which catalogue films these hard filters keep out of the pool entirely.

    Cheap: one indexed read over 20 rows, no vectors and no model. Returns
    [{title, runtime_minutes, year}, ...], empty when no filter is set.
    """
    if not any((max_runtime, min_runtime, after_year, before_year)):
        return []
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(_EXCLUDED_SQL, {
            "max_runtime": max_runtime, "min_runtime": min_runtime,
            "after_year": after_year, "before_year": before_year}).fetchall()
    return [{"title": title, "runtime_minutes": runtime, "year": year}
            for title, runtime, year in rows]


if __name__ == "__main__":
    # Self-test: real query, real database, real reranker.
    print("=" * 74)
    print("VECTORS — the same catalogue, scored")
    print("=" * 74)
    for film in search("a father and son separated and trying to find each other"):
        print(f"   {film['score']:.3f}  {film['title']:<34} "
              f"matched on its {film['source']} text")
        print(f"          {film['evidence'][:96]}")

    print()
    print("hard filters remove films BEFORE ranking — this is what max_runtime=120 costs:")
    for film in excluded_by_filters(max_runtime=120):
        print(f"   {film['title']:<44} {film['runtime_minutes']} min")

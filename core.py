"""Shared building blocks: embedding, retrieval, reranking."""

import json
import os
import re
import random
import time

import boto3
import httpx
import psycopg
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

DIMENSIONS = 1024
# How much matched text travels back with each film.
#
# Was 320. Measured on the 8-case agent eval: 5 of 23 evidence strings were being cut
# at exactly that limit — roughly a fifth of the evidence stopped mid-sentence. A claim
# can be true, present in the source, and still score 0 for faithfulness because the
# sentence that supported it never reached the judge.
#
# The cost of raising it is real but opposite: more text is more raw material for the
# model to embroider on. Baseline before the change: faithfulness 0.72 over 6 judged
# cases, tool accuracy 8/8.
EVIDENCE_CHARS = 640
MODEL_ID = os.environ["BEDROCK_MODEL_EMBED_NOVA"]
DATABASE_URL = os.environ["DATABASE_URL"]
REGION = os.environ["AWS_REGION"]

# ─── Reranker: OpenRouter (provider-agnostic gateway → Cohere rerank) ───
# One stateless HTTPS call. No AWS Marketplace subscription and no per-region
# model access to clear — the gateway handles the vendor underneath. httpx
# verifies TLS against certifi's CA bundle, so this works on any machine.
RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cohere/rerank-v3.5")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Bulk embedding jobs exceed the free tier's request rate. boto3's "adaptive" mode helps
# but does not hold back far enough on a free plan, so throttling is also handled here.
_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
)

# ─── Tracing (LangSmith) ───────────────────────────────────────────────
# LangChain traces itself automatically, so the agent's model calls and tool calls appear
# in LangSmith with no code at all. But `search()` is PLAIN PYTHON — LangChain knows
# nothing about it — so the whole retrieval pipeline would show up as one opaque box:
# you would see that search_films ran and what it returned, and nothing about the SQL
# pool, the rerank, or the collapse. These decorators open that box.
#
# `traceable` is a no-op unless LANGSMITH_TRACING=true, and the import is optional, so a
# machine that has never heard of LangSmith still runs the pipeline unchanged.
try:
    from langsmith import traceable
except ImportError:                                     # degrade, do not fail
    def traceable(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate(args[0]) if args and callable(args[0]) else decorate


def _hide_vectors(inputs):
    """A 1024-float vector in a trace is noise, not evidence. Log its shape instead."""
    shown = dict(inputs)
    if shown.get("query_vector"):
        shown["query_vector"] = f"<{DIMENSIONS}-dim vector, pre-computed>"
    return shown


def _embedding_shape(output):
    return {"dimensions": len(output) if output else 0}


# Self-pacing throttle control. `_pace` is the delay held between calls; it GROWS when the
# service pushes back and DECAYS while calls succeed, so the script finds the fastest rate
# the account actually allows instead of guessing one. A throttle becomes a pause, never a
# crash — the same "degrade, don't fail" posture used for the reranker.
_pace = 0.0
PACE_MAX = 8.0
THROTTLE_ATTEMPTS = 8


def _invoke_with_backoff(**kwargs):
    """Call Bedrock, backing off exponentially on ThrottlingException."""
    global _pace
    for attempt in range(THROTTLE_ATTEMPTS):
        if _pace:
            time.sleep(_pace)
        try:
            response = _bedrock.invoke_model(**kwargs)
            _pace = max(0.0, _pace * 0.9)          # calm down again once it is flowing
            return response
        except _bedrock.exceptions.ThrottlingException:
            _pace = min(PACE_MAX, max(0.5, _pace * 2))
            wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
            print(f"  [throttled — waiting {wait:.0f}s, pacing at {_pace:.1f}s/call]")
            time.sleep(wait)
    raise RuntimeError(
        f"Bedrock still throttling after {THROTTLE_ATTEMPTS} attempts. "
        "Re-run — finished work is committed and embeddings are cached."
    )

# Retrieve at CHUNK level, STRATIFIED by source type — deliberately not one row per film.
#
# Three measured failures shaped this (experiments/why_chunk.py):
#  1. Collapsing to one chunk per film inside the SQL let a film's short `derived` chunk
#     (~120 chars of mood keywords) beat its own 600-char plot scene on raw cosine, so the
#     plot corpus was discarded before the reranker could vote. Fix: collapse AFTER rerank.
#  2. Widening the pool did not help: for an abstract query EVERY derived/overview chunk
#     out-scored EVERY plot chunk, so the top 30 held zero plot text. Cosine rewards
#     concentration, and short abstract chunks are concentrated by construction. Fix: a
#     per-source-type quota, each type ranked within its own kind.
#  3. A retrieved scene could still lose because chunking severs relationships — the text
#     "Nemo swims to a speedboat and is captured" names no father and no search. Fix: a
#     film-level context header. WHERE that header is applied is now an experiment, not an
#     assumption: `variant` picks which stored vector to search, `header_at_rerank` decides
#     whether the reranker is shown it. See eval_variants.py.
#
# Joins use USING(...) because a key that links two tables carries the SAME column name in
# both — the name itself states the grain.
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


@traceable(run_type="retriever", name="get_film (exact title lookup)")
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


@traceable(run_type="embedding", name="embed", process_outputs=_embedding_shape)
def embed(text):
    """Turn a piece of text into a vector."""
    response = _invoke_with_backoff(
        modelId=MODEL_ID,
        body=json.dumps({
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": DIMENSIONS,
                "text": {"truncationMode": "END", "value": text},
            },
        }),
        accept="application/json",
        contentType="application/json",
    )
    return json.loads(response["body"].read())["embeddings"][0]["embedding"]


@traceable(run_type="tool", name="rerank (cohere via openrouter)")
def rerank(query, documents, top_n):
    """Reorder documents by reading each against the query. Returns [{index, score}].

    Hosted on OpenRouter (Cohere rerank underneath): the cross-encoder reads the
    query and each document together, so it can tell "creatures hunting people"
    apart from "a tiger in the bathroom" — the thing a single vector cannot.
    """
    response = httpx.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return [
        {"index": r["index"], "score": r.get("relevance_score", r.get("relevanceScore"))}
        for r in data["results"]
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


@traceable(run_type="retriever", name="search (retrieve -> rerank -> collapse)",
           process_inputs=_hide_vectors)
def search(query, limit=3, use_rerank=True, plot_k=30, other_k=10,
           variant="context_header", header_at_rerank=False, query_vector=None,
           max_runtime=None, min_runtime=None, after_year=None, before_year=None,
           exclude_title=None, sources=None):
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


# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH — the exact half of the system
#
# Nothing below embeds anything, calls a model, or produces a score. These are
# FACTS: a person either directed a film or did not. Where the vector path
# answers "what feels like this", this answers "what IS this".
#
# The two never mix. A question with a yes/no test must never be sent to the
# reranker — measured on this catalogue, "films starring Arnold Schwarzenegger"
# returns Terminator 2 at 0.618 even though the word "Schwarzenegger" appears
# NOWHERE in the corpus. That score came from the reranker's own training, not
# from this database. Right by accident is still ungrounded.
# ═══════════════════════════════════════════════════════════════════════════

GRAPH_LIMIT = 10          # a factual answer is a list, not a ranking; keep it short

_BY_PERSON_SQL = """
SELECT DISTINCT f.node_key, f.name
FROM graph_edges e
JOIN graph_nodes p ON p.node_key = e.from_key
JOIN graph_nodes f ON f.node_key = e.to_key
WHERE e.edge_type = %(edge_type)s
  AND (lower(p.name) = lower(%(exact)s) OR lower(p.name) LIKE lower(%(loose)s))
"""

_BY_GENRE_SQL = """
SELECT DISTINCT f.node_key, f.name
FROM graph_edges e
JOIN graph_nodes f ON f.node_key = e.from_key
WHERE e.edge_type = 'HAS_GENRE' AND e.to_key = %(genre_key)s
"""

_SIMILAR_SQL = """
SELECT other.node_key, other.name, count(*) AS shared,
       string_agg(kw.name, ', ' ORDER BY kw.name) AS keywords
FROM graph_edges mine
JOIN graph_edges theirs ON theirs.to_key = mine.to_key
                       AND theirs.edge_type = 'HAS_KEYWORD'
JOIN graph_nodes other  ON other.node_key = theirs.from_key
JOIN graph_nodes kw     ON kw.node_key = mine.to_key
WHERE mine.from_key = %(film_key)s AND mine.edge_type = 'HAS_KEYWORD'
  AND theirs.from_key <> mine.from_key
GROUP BY other.node_key, other.name
ORDER BY shared DESC, other.name
"""

_FILM_BY_TITLE_SQL = """
SELECT node_key, name FROM graph_nodes
WHERE node_type = 'film'
  AND (lower(name) = lower(%(exact)s) OR lower(name) LIKE lower(%(loose)s))
ORDER BY (lower(name) = lower(%(exact)s)) DESC, length(name)
LIMIT 1
"""

_FILM_FACTS_SQL = """
SELECT n.node_key, n.name,
       n.properties->>'release_date'    AS release_date,
       n.properties->>'runtime_minutes' AS runtime_minutes,
       -- The film's own words. Without this the caller knows WHICH films matched but
       -- nothing true about them, and a model handed a gap fills it from training.
       (SELECT c.content FROM chunks c
        WHERE c.movie_id = (n.properties->>'movie_id')::int
          AND c.source_field = 'overview'
        ORDER BY c.chunk_index LIMIT 1)  AS overview
FROM graph_nodes n WHERE n.node_key = ANY(%(keys)s)
"""


def graph_genres():
    """Every genre in the catalogue, as the model must spell them. Enumerable, so it
    belongs in the tool description rather than being guessed at."""
    with psycopg.connect(DATABASE_URL) as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM graph_nodes WHERE node_type = 'genre' ORDER BY name")]


@traceable(run_type="retriever", name="graph_find (exact facts, no vectors)")
def graph_find(director=None, actor=None, genre=None, similar_to=None, limit=GRAPH_LIMIT):
    """Exact catalogue lookup over the knowledge graph.

    Every criterion supplied is ANDed — director='Christopher Nolan', genre='Action'
    returns only films satisfying both. Returns a dict:

        {"films": [{title, release_date, runtime_minutes, why: [...], evidence: str}, ...],
         "unknown": {...}}      # a name or genre that matched nothing in the graph

    `unknown` matters. "No Tarantino films" and "Tarantino is not in this catalogue at
    all" are different answers, and a caller that cannot tell them apart will say the
    wrong one.
    """
    sets, why, unknown = [], {}, {}

    with psycopg.connect(DATABASE_URL) as conn:

        for name, edge_type, label in ((director, "DIRECTED", "directed by"),
                                       (actor, "ACTED_IN", "starring")):
            if not name:
                continue
            rows = conn.execute(_BY_PERSON_SQL, {
                "edge_type": edge_type, "exact": name, "loose": f"%{name}%"}).fetchall()
            if not rows:
                unknown["director" if edge_type == "DIRECTED" else "actor"] = name
                return {"films": [], "unknown": unknown}
            sets.append({key for key, _ in rows})
            for key, _ in rows:
                why.setdefault(key, []).append(f"{label} {name}")

        if genre:
            key = "genre:" + re.sub(r"[^a-z0-9]+", "-", genre.strip().lower()).strip("-")
            rows = conn.execute(_BY_GENRE_SQL, {"genre_key": key}).fetchall()
            if not rows:
                unknown["genre"] = genre
                return {"films": [], "unknown": unknown}
            sets.append({k for k, _ in rows})
            for k, _ in rows:
                why.setdefault(k, []).append(f"genre {genre}")

        if similar_to:
            seed = conn.execute(_FILM_BY_TITLE_SQL, {
                "exact": similar_to, "loose": f"%{similar_to}%"}).fetchone()
            if not seed:
                unknown["similar_to"] = similar_to
                return {"films": [], "unknown": unknown}
            rows = conn.execute(_SIMILAR_SQL, {"film_key": seed[0]}).fetchall()
            sets.append({k for k, _, _, _ in rows})
            for k, _, shared, keywords in rows:
                why.setdefault(k, []).append(
                    f"shares {shared} keyword{'s' if shared > 1 else ''} with "
                    f"{seed[1]} ({keywords})")
            # similarity is the only criterion with an order worth keeping
            order = {k: i for i, (k, _, _, _) in enumerate(rows)}
        else:
            order = {}

        if not sets:
            return {"films": [], "unknown": {}}

        keys = list(set.intersection(*sets))
        if not keys:
            return {"films": [], "unknown": {}}

        facts = conn.execute(_FILM_FACTS_SQL, {"keys": keys}).fetchall()

    films = [{
        "_key": key,
        "title": name,
        "release_date": release_date,
        "runtime_minutes": int(runtime) if runtime else None,
        "why": why.get(key, []),
        "evidence": " ".join((overview or "").split())[:EVIDENCE_CHARS],
    } for key, name, release_date, runtime, overview in facts]

    # Only `similar_to` produces a meaningful order (most shared keywords first).
    # Everything else is a set of facts, so it sorts alphabetically.
    films.sort(key=lambda f: (order.get(f["_key"], 9999), f["title"]))
    for film in films:
        film.pop("_key")
    return {"films": films[:limit], "unknown": {}}


if __name__ == "__main__":
    # Self-test: real queries, real database, no model needed for the graph half.
    print("=" * 74)
    print("GRAPH — exact, no scores")
    print("=" * 74)
    print("genres:", ", ".join(graph_genres()))
    for probe in ({"director": "Christopher Nolan"},
                  {"actor": "Arnold Schwarzenegger"},
                  {"genre": "Horror"},
                  {"director": "Christopher Nolan", "genre": "Action"},
                  {"similar_to": "Inception"},
                  {"director": "Quentin Tarantino"}):
        result = graph_find(**probe)
        print(f"\ngraph_find({probe})")
        if result["unknown"]:
            print(f"   not in the catalogue: {result['unknown']}")
        for film in result["films"][:5]:
            print(f"   • {film['title']:<34} {'; '.join(film['why'])[:60]}")
            print(f"     {film['evidence'][:96]}")
        if not result["films"] and not result["unknown"]:
            print("   (no film satisfies all of those at once)")

    print("\n" + "=" * 74)
    print("VECTORS — same catalogue, scored")
    print("=" * 74)
    for film in search("a father and son separated and trying to find each other"):
        print(f"   {film['score']:.3f}  {film['title']}")

"""Split Wikipedia plots into scene-sized chunks — SEMANTIC → RECURSIVE → OVERLAP.

WHY THIS EXISTS
    A TMDB overview is ~350 characters and names no scene, so "creatures chasing people"
    or "a father and son separated" had nothing concrete to match. A Wikipedia plot is
    ~4,000 characters and narrates 20+ scenes. But a single vector over 4,000 characters
    averages every scene into mush — so the plot must be SPLIT, and split in the right
    places.

THE HYBRID STRATEGY — three passes, each covering the previous one's blind spot
    1. SEMANTIC proposes.  Every sentence is embedded; the distance between neighbouring
       sentences is measured; a boundary lands where meaning SHIFTS. Chunking by meaning,
       not by a character count that happened to run out.
       Blind spot: a single scene can run longer than a model's usable context.
    2. RECURSIVE enforces.  Any segment over MAX_CHARS is cut again — at ITS OWN largest
       internal meaning shift — repeating until every piece fits. The cap decides WHETHER
       to cut; meaning still decides WHERE.
       Blind spot: a size-forced cut severs a continuous thought. The piece after the cut
       is an orphan — it opens mid-scene with its setup missing.
    3. OVERLAP heals.  A forced seam gets the tail of the previous chunk copied onto the
       front of the next, so the orphan carries its own context.

    THE REFINEMENT — overlap is applied SELECTIVELY, only at recursion-forced seams.
    At a semantic boundary the meaning genuinely changed, so duplicating text across it
    drags the old scene into the new one and blurs both. Uniform overlap (the common
    default) pays that cost at every boundary. Here the chunker remembers WHY each cut
    was made and only heals the cuts that actually tore something.
        semantic boundary  → no overlap (nothing was torn)
        recursive boundary → overlap    (a thought was severed; stitch it)

WHY A PERCENTILE THRESHOLD, NOT A FIXED ONE
    Nova's vector space is narrow — unrelated text still scores ~0.64 similarity (proved
    in experiments/space_shape.py). "Split below 0.8" would therefore mean something
    different in every document and break entirely on a model swap. Each plot is scored
    against ITSELF: break at the top (100 - BREAK_PERCENTILE)% of its own biggest shifts.

CONTEXTUAL RETRIEVAL — why every chunk carries a header
    Measured failure (experiments/why_chunk.py): for "a father and son separated", Finding
    Nemo's separation scene WAS retrieved (rank 3 of plot) but the reranker scored it below
    Home Alone. Reading the chunk as a cross-encoder does explains why —
        "Nemo defiantly swims to a nearby speedboat ... and is captured by scuba divers"
    contains no "father", no "son", and no search. Two names and an event. Chunking cut the
    scene away from the fact that Marlin IS Nemo's father. Home Alone's text, meanwhile,
    literally says "he and his son are estranged", so it won on words the query used.

    Fix: a short film-level context header, so a chunk can never be read out of context.
    Built from data already held (title + TMDB overview), so it costs no extra model calls.

    WHERE the header applies is no longer assumed. It is stored ONCE on the film
    (movies.context_header) rather than copied into 145 chunks — that copying was 31% of
    the plot corpus and diluted every vector. This script writes TWO vectors per plot
    chunk so the question can be settled by measurement:
        embed_variant='clean'           the scene text alone
        embed_variant='context_header'  header + scene text
    core.py picks which to search, and can also compose the header onto the reranker's
    document only. See eval_variants.py.

SMALL-TO-BIG
    Match at chunk (scene) level, return the parent film. core.py collapses to one row
    per film AFTER reranking, scoring each film by its top 3 chunks.

CHUNK INDEX MAP (chunks.chunk_index is UNIQUE per movie)
    0     = overview  (load_chunks.py)
    1     = derived   (load_derived.py)
    2..N  = plot      (this script)

COST NOTE
    Sentence vectors are cached to data/sentence_vectors.json, keyed by SHA-1 of the text.
    Changing the chunking strategy therefore costs almost nothing: the sentences are
    unchanged, so only the final chunk embeddings are re-computed.
"""

import glob
import hashlib
import json
import math
import os
import re
import time

import psycopg

from core import DATABASE_URL, DIMENSIONS, MODEL_ID, embed

MAX_CHARS = 900              # cap on a chunk's OWN content; overlap may add to this
MIN_CHARS = 250              # below this a chunk is thin — merge forward if it fits
BREAK_PERCENTILE = 80        # break at the top 20% biggest meaning shifts in each plot
OVERLAP_SENTENCES = 1        # how much of the previous chunk to copy onto a forced seam
OVERLAP_AT_SEMANTIC = False  # meaning changed there; copying across it would blur both
CONTEXT_CHARS = 240        # how much film-level context to prepend to each chunk
PLOT_START_INDEX = 2
CACHE_PATH = "data/sentence_vectors.json"
SLEEP = 0.0                  # pacing is now handled adaptively inside core._invoke_with_backoff

INSERT_CHUNK = """
INSERT INTO chunks (movie_id, chunk_index, source_field, content)
VALUES (%(movie_id)s, %(chunk_index)s, 'plot', %(content)s)
ON CONFLICT (movie_id, chunk_index) DO UPDATE SET content = EXCLUDED.content,
                                                  source_field = EXCLUDED.source_field
RETURNING chunk_id
"""

INSERT_EMBEDDING = """
INSERT INTO chunk_embeddings (chunk_id, embedding, model_id, dimensions, embed_variant)
VALUES (%(chunk_id)s, %(embedding)s::vector, %(model_id)s, %(dimensions)s, %(variant)s)
ON CONFLICT (chunk_id, model_id, embed_variant) DO UPDATE SET embedding = EXCLUDED.embedding
"""

SAVE_HEADER = """
UPDATE movies SET context_header = %(header)s WHERE movie_id = %(movie_id)s
"""

DELETE_OLD_PLOTS = "DELETE FROM chunks WHERE movie_id = %(movie_id)s AND chunk_index >= 2"


# ─────────────────────────── embedding cache ───────────────────────────

_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
_cache_dirty = False


def embed_cached(text):
    """Embed, reusing a stored vector when this exact text was embedded before."""
    global _cache_dirty
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    if key not in _cache:
        _cache[key] = embed(text)
        _cache_dirty = True
        time.sleep(SLEEP)
    return _cache[key]


def save_cache():
    if _cache_dirty:
        os.makedirs("data", exist_ok=True)
        with open(CACHE_PATH, "w") as f:
            json.dump(_cache, f)


def load_contexts():
    """Film-level context header, built from the raw TMDB payload already on disk.

    The overview names the relationships a plot chunk assumes you already know —
    Finding Nemo's says "his worrisome father Marlin" — which is exactly the fact the
    reranker needs and the chunk itself lacks.
    """
    contexts = {}
    for path in sorted(glob.glob("data/raw/tmdb_*.json")):
        film = json.load(open(path))
        overview = " ".join((film.get("overview") or "").split())
        if len(overview) > CONTEXT_CHARS:
            overview = overview[:CONTEXT_CHARS].rsplit(" ", 1)[0] + "..."
        year = (film.get("release_date") or "")[:4]
        label = f"{film.get('title')} ({year})" if year else str(film.get("title"))
        contexts[film.get("title")] = f"{label}. {overview}".strip()
    return contexts


# ─────────────────────────── the chunker ───────────────────────────

def split_sentences(text):
    """Sentence split. Crude on purpose — dependency-free, and good enough for prose."""
    flat = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


def cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (na * nb)) if na and nb else 1.0


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] * (high - k) + ordered[high] * (k - low)


def split_recursively(sentences, distances, left_forced):
    """PASS 2 — enforce the cap, cutting at the biggest internal meaning shift.

    Each piece records whether its LEFT seam was created by a size-forced cut
    (left_forced=True) — that is the seam overlap will later need to heal.
    """
    if len(" ".join(sentences)) <= MAX_CHARS or len(sentences) == 1:
        return [{"sentences": sentences, "left_forced": left_forced}]
    best = max(range(len(sentences) - 1), key=lambda i: distances[i])
    left = split_recursively(sentences[: best + 1], distances[:best], left_forced)
    right = split_recursively(sentences[best + 1:], distances[best + 1:], True)
    return left + right


def chunk_plot(plot):
    """Semantic proposes → recursive enforces → overlap heals the forced seams.

    Returns [{content, own_chars, seam}] where seam explains this chunk's left edge:
    'start' | 'semantic' | 'recursive+overlap'.
    """
    sentences = split_sentences(plot)
    if len(sentences) <= 1:
        text = plot.strip()
        return [{"content": text, "own_chars": len(text), "seam": "start"}] if text else []

    vectors = [embed_cached(s) for s in sentences]
    distances = [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]

    # ── PASS 1 · SEMANTIC: break where meaning shifts most, relative to THIS plot
    threshold = percentile(distances, BREAK_PERCENTILE)
    segments, current, current_d = [], [sentences[0]], []
    for i, distance in enumerate(distances):
        if distance >= threshold:
            segments.append((current, current_d))
            current, current_d = [sentences[i + 1]], []
        else:
            current.append(sentences[i + 1])
            current_d.append(distance)
    segments.append((current, current_d))

    # ── PASS 2 · RECURSIVE: enforce the cap, still cutting at meaning
    pieces = []
    for sentence_group, group_distances in segments:
        pieces.extend(split_recursively(sentence_group, group_distances, left_forced=False))

    # merge true runts forward where they fit — removes orphans overlap would only paper over
    merged = []
    for piece in pieces:
        if merged and len(" ".join(piece["sentences"])) < MIN_CHARS:
            combined = merged[-1]["sentences"] + piece["sentences"]
            if len(" ".join(combined)) <= MAX_CHARS:
                merged[-1]["sentences"] = combined
                continue
        merged.append(piece)

    # ── PASS 3 · OVERLAP: stitch only the seams that severed a continuous thought
    out = []
    for index, piece in enumerate(merged):
        own = " ".join(piece["sentences"])
        if index == 0:
            seam = "start"
            content = own
        elif piece["left_forced"] or OVERLAP_AT_SEMANTIC:
            seam = "recursive+overlap"
            carry = merged[index - 1]["sentences"][-OVERLAP_SENTENCES:]
            content = " ".join(carry + piece["sentences"])
        else:
            seam = "semantic"
            content = own
        out.append({"content": content, "own_chars": len(own), "seam": seam})
    return out


# ─────────────────────────── loading ───────────────────────────

def resolve_movie_ids(conn, plots):
    """Map each plot to its movies.id — by identifier where possible, never by guesswork."""
    rows = conn.execute("SELECT movie_id, source_id, title FROM movies").fetchall()
    by_source = {str(source_id): mid for mid, source_id, _ in rows}
    by_title = {title: mid for mid, _, title in rows}

    resolved, missing = {}, []
    for film in plots:
        if str(film["tmdb_id"]) in by_source:
            resolved[film["title"]] = by_source[str(film["tmdb_id"])]
        elif film["title"] in by_title:
            resolved[film["title"]] = by_title[film["title"]]
        else:
            missing.append(film["title"])
    return resolved, missing


def main():
    plots = json.load(open("data/plots.json"))
    contexts = load_contexts()

    with psycopg.connect(DATABASE_URL) as conn:
        movie_ids, missing = resolve_movie_ids(conn, plots)
        if missing:
            print(f"WARNING — no movies row for: {missing}")

        total_chunks = stitched = 0
        for film in plots:
            movie_id = movie_ids.get(film["title"])
            if movie_id is None or not film.get("plot"):
                continue

            chunks = chunk_plot(film["plot"])

            # The header is film-level, so it belongs on the film — stored once, not
            # copied into every chunk. Chunks keep only their own scene text.
            header = contexts.get(film["title"], film["title"])
            conn.execute(SAVE_HEADER, {"header": header, "movie_id": movie_id})

            conn.execute(DELETE_OLD_PLOTS, {"movie_id": movie_id})

            for offset, chunk in enumerate(chunks):
                chunk_id = conn.execute(INSERT_CHUNK, {
                    "movie_id": movie_id,
                    "chunk_index": PLOT_START_INDEX + offset,
                    "content": chunk["content"],
                }).fetchone()[0]

                # two vectors per chunk — the arms of the header experiment
                for variant, text in (
                    ("clean", chunk["content"]),
                    ("context_header", f"{header}\n\n{chunk['content']}"),
                ):
                    conn.execute(INSERT_EMBEDDING, {
                        "chunk_id": chunk_id,
                        "embedding": str(embed_cached(text)),
                        "model_id": MODEL_ID,
                        "dimensions": DIMENSIONS,
                        "variant": variant,
                    })

            conn.commit()          # commit per film: throttling can't cost finished work
            save_cache()           # and keep the vectors we already paid for

            healed = sum(1 for c in chunks if c["seam"] == "recursive+overlap")
            stitched += healed
            total_chunks += len(chunks)
            shape = "/".join(
                f"{c['own_chars']}{'+' if c['seam'] == 'recursive+overlap' else ''}"
                for c in chunks
            )
            print(f"{film['title']:40.40} {len(chunks):2} chunks  {healed} stitched  [{shape}]")

        print(f"\nembedded {total_chunks} plot chunks · {stitched} seams healed by overlap")
        print("  ('+' marks a chunk whose left seam was a size-forced cut, given overlap)")
        for table in ("movies", "chunks", "chunk_embeddings"):
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"  {table}: {count}")
        breakdown = conn.execute(
            "SELECT source_field, count(*) FROM chunks GROUP BY source_field ORDER BY 1"
        ).fetchall()
        print("  by source_field:", dict(breakdown))


if __name__ == "__main__":
    try:
        main()
    finally:
        save_cache()

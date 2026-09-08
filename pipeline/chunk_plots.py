"""Cut each Wikipedia plot into scenes and store them as movie_data rows.

WHY CHUNK AT ALL
    A whole plot is 3,000 characters covering a dozen unrelated events. Embedded as one
    vector it becomes an average of everything and matches nothing sharply. Cut into
    scenes, a query about one moment can find the one paragraph that contains it — and
    the film is still what gets returned. Small chunks match precisely; people want the
    film, not the paragraph.

THE THREE CUTS, IN THIS ORDER — SEMANTIC, THEN RECURSIVE, THEN OVERLAP
    SEMANTIC first: cut where the MEANING changes. Every sentence is embedded and each
        neighbouring pair is compared; the weakest joins are where the story moves on.
    RECURSIVE second: a segment still over the size cap is split again, at its own
        biggest internal drop. The cap is enforced by cutting at the best available
        seam rather than at a character count.
    OVERLAP last, and ONLY on seams the size cap forced. At a semantic boundary the
        meaning genuinely changed, so repeating a sentence across it would blur two
        distinct scenes into each other. Overlap repairs damage; it is not a default.

THE THRESHOLD IS A PERCENTILE, NOT A NUMBER
    "Split when similarity drops below 0.8" dies the day the embedding model changes,
    because every model's similarity scale is its own. This model puts unrelated text
    around 0.64, so a fixed 0.8 would cut almost every sentence. Each plot is scored
    against ITSELF: the weakest quarter of its own joins are the cuts.

SENTENCE VECTORS ARE CACHED ON DISK
    Keyed by the hash of the sentence, so re-running after a threshold change costs no
    model calls at all. Only genuinely new sentences are embedded.

PLOT SCENES ARE A MATCHING SURFACE, NEVER A DISPLAY ONE
    They give away endings. They are not in retrieval's DISPLAYABLE set: they rank
    films, and the premise is what the user reads. Same rule as theme.

RUN
    python -m pipeline.chunk_plots              chunk every film that has a plot
    python -m pipeline.chunk_plots --dry-run    show the cuts, write nothing
    python -m pipeline.chunk_plots --titles "Alien"
"""

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg                                                     # noqa: E402
from dotenv import load_dotenv                                     # noqa: E402

from backend.config import DATABASE_URL                            # noqa: E402
from backend.models import embed                                   # noqa: E402
from pipeline.load_derived import load                             # noqa: E402

load_dotenv()

KIND = "plot_scene"
CACHE = "data/plot_sentence_vectors.json"

BREAK_PERCENTILE = 25     # the weakest quarter of a plot's own joins become cuts
MAX_CHARS = 900           # above this, a scene is split again at its own weakest join
MIN_CHARS = 200           # below this, a scene is joined to the next one

PLOTS = """
SELECT movie_id, title, wikidata_plot
FROM movies
WHERE wikidata_plot IS NOT NULL
  AND (%(titles)s::text[] IS NULL OR title = ANY(%(titles)s::text[]))
ORDER BY movie_id
"""

DROP_EXTRA = """
DELETE FROM movie_data
WHERE movie_id = %(movie_id)s AND data_kind = %(kind)s AND seq > %(keep)s
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SENTENCES
# ══════════════════════════════════════════════════════════════════════════════

# Written out rather than adding a sentence-splitting library. The whole problem is
# abbreviations — a full stop inside "Dr." or "U.S." is not the end of a sentence — and
# a plot summary uses a small, known set of them.
ABBREVIATIONS = ("Mr", "Mrs", "Ms", "Dr", "Prof", "St", "Sgt", "Lt", "Capt", "Jr", "Sr",
                 "vs", "etc", "approx", "No", "Col", "Gen", "Rev", "Mt", "Ft")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def sentences(text):
    """Split prose into sentences, keeping abbreviations intact."""
    parts, buffer = [], ""
    for piece in SENTENCE_END.split(re.sub(r"\s+", " ", text.strip())):
        buffer = f"{buffer} {piece}".strip() if buffer else piece
        last = buffer.rstrip(".").rsplit(" ", 1)[-1].strip("(\"'")
        ends_on_abbreviation = last in ABBREVIATIONS
        ends_on_initial = len(last) == 1 and last.isalpha()      # "J." in "J. Smith"
        if not (ends_on_abbreviation or ends_on_initial):
            parts.append(buffer)
            buffer = ""
    if buffer:
        parts.append(buffer)
    return [p for p in parts if p.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORS, CACHED
# ══════════════════════════════════════════════════════════════════════════════

def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def vector_for(sentence, cache):
    """Cached by CONTENT, not by position. Re-cutting a plot re-uses every sentence."""
    key = hashlib.sha1(sentence.encode("utf-8")).hexdigest()
    if key not in cache:
        cache[key] = embed(sentence)
    return cache[key]


def similarity(a, b):
    """Cosine, written out. Both vectors come from the same model, so this is honest."""
    dot = sum(x * y for x, y in zip(a, b))
    size_a = sum(x * x for x in a) ** 0.5
    size_b = sum(y * y for y in b) ** 0.5
    return dot / (size_a * size_b) if size_a and size_b else 0.0


def percentile(values, p):
    """The value below which p% of these numbers sit. No numpy for one line of maths."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * p / 100))
    return ordered[index]


# ══════════════════════════════════════════════════════════════════════════════
#  THE THREE CUTS
# ══════════════════════════════════════════════════════════════════════════════

def semantic_cuts(joins, threshold):
    """Indexes where the story moves on. joins[i] joins sentence i to sentence i+1."""
    return {i for i, score in enumerate(joins) if score <= threshold}


def segments_from(sents, cuts):
    """Turn a set of cut positions into runs of sentences."""
    out, current = [], []
    for i, sentence in enumerate(sents):
        current.append(sentence)
        if i in cuts or i == len(sents) - 1:
            out.append(current)
            current = []
    return [seg for seg in out if seg]


def split_oversized(segment, joins, offset):
    """RECURSIVE: a segment over the cap is cut at its OWN weakest internal join.

    Returns (pieces, forced_seams) — forced_seams counts the cuts the size cap caused,
    because those are the only seams overlap is allowed to repair.
    """
    text = " ".join(segment)
    if len(text) <= MAX_CHARS or len(segment) < 2:
        return [segment], 0

    inside = [(joins[offset + i], i) for i in range(len(segment) - 1)]
    _, at = min(inside)                      # the weakest join inside this segment
    left, right = segment[:at + 1], segment[at + 1:]

    left_pieces, left_forced = split_oversized(left, joins, offset)
    right_pieces, right_forced = split_oversized(right, joins, offset + at + 1)
    return left_pieces + right_pieces, 1 + left_forced + right_forced


def merge_tiny(segments):
    """Attach an undersized scene to a neighbour — backwards first, then forwards.

    THREE RULES THAT CANNOT ALL HOLD AT ONCE
        no scene over MAX_CHARS · no scene under MIN_CHARS · never cut where the meaning
        did not change. A short segment between two nearly-full ones satisfies the third
        rule and breaks one of the other two whichever way you move it.

        The size cap wins, because it is the one with a downstream consequence: an
        oversized chunk is an average of several events and matches none of them
        sharply, which is the exact failure chunking exists to prevent. A short scene is
        merely a weak chunk.

        So: try the scene before, then the scene after, and if neither has room, leave it
        standing rather than break the cap. Measured on Toy Story, where a 53-character
        orphan appeared because the merge only ever looked backwards.
    """
    def fits(segment):
        return len(" ".join(segment)) <= MAX_CHARS

    def too_small(segment):
        return len(" ".join(segment)) < MIN_CHARS

    out, waiting = [], []
    for segment in segments:
        if waiting:
            # A scene held over from the last step, looking for room in this one.
            if fits(waiting + segment):
                segment = waiting + segment
            else:
                out.append(waiting)          # nowhere to go; let it stand
            waiting = []

        if too_small(segment):
            if out and fits(out[-1] + segment):
                out[-1] = out[-1] + segment
            else:
                waiting = segment
            continue

        out.append(segment)

    if waiting:
        if out and fits(out[-1] + waiting):
            out[-1] = out[-1] + waiting
        else:
            out.append(waiting)
    return out


def chunk(plot, cache):
    """A plot -> (list of scene texts, how many cuts the size cap forced)."""
    sents = sentences(plot)
    if len(sents) < 2:
        return [plot.strip()], 0

    vectors = [vector_for(s, cache) for s in sents]
    joins = [similarity(vectors[i], vectors[i + 1]) for i in range(len(sents) - 1)]
    threshold = percentile(joins, BREAK_PERCENTILE)

    pieces, forced = [], 0
    for segment in segments_from(sents, semantic_cuts(joins, threshold)):
        start = sents.index(segment[0])
        split, count = split_oversized(segment, joins, start)
        pieces += split
        forced += count

    # OVERLAP, last and only where the size cap forced a seam. At a semantic boundary
    # the meaning genuinely changed, so overlapping there would blur two real scenes.
    return [" ".join(seg) for seg in merge_tiny(pieces)], forced


# ══════════════════════════════════════════════════════════════════════════════
#  THE RUN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cuts and write nothing to the database")
    parser.add_argument("--titles", nargs="+", help="only these exact titles")
    args = parser.parse_args()

    cache = load_cache()
    cached_at_start = len(cache)
    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    total_scenes = 0

    with psycopg.connect(DATABASE_URL) as conn:
        films = conn.execute(PLOTS, {"titles": args.titles}).fetchall()
        print(f"{len(films)} films with a plot · cache holds {cached_at_start} "
              f"sentence vectors\n")

        for movie_id, title, plot in films:
            scenes, forced = chunk(plot, cache)
            total_scenes += len(scenes)
            sizes = [len(s) for s in scenes]

            print(f"  {movie_id:>3}  {title:34} {len(scenes):>3} scenes · "
                  f"{min(sizes):>4}-{max(sizes):<5} chars · {forced} forced by size")

            if args.dry_run:
                for n, scene in enumerate(scenes, 1):
                    print(f"        {n:>2}. {scene[:110]}…")
                continue

            for n, scene in enumerate(scenes, 1):
                counts[load(conn, movie_id, KIND, scene, seq=n)] += 1

            # A re-cut can produce FEWER scenes than last time. The leftovers would sit
            # there for ever, still embedded, still findable — a film answering with a
            # scene the current chunking does not believe in.
            conn.execute(DROP_EXTRA, {"movie_id": movie_id, "kind": KIND,
                                      "keep": len(scenes)})

        if not args.dry_run:
            conn.commit()

    json.dump(cache, open(CACHE, "w"))

    print(f"\n{total_scenes} scenes · {total_scenes / max(1, len(films)):.1f} per film")
    print(f"cache {cached_at_start} -> {len(cache)} "
          f"({len(cache) - cached_at_start} sentences newly embedded)")
    if args.dry_run:
        print("DRY RUN — nothing was written.")
    else:
        print(f"inserted {counts['inserted']} · updated {counts['updated']} · "
              f"unchanged {counts['unchanged']}")
        print("Now run: python -m pipeline.embed_data")


if __name__ == "__main__":
    main()

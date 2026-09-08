"""Write each film's mood, theme and premise. Save to disk for review, not to the DB.

WHAT THIS IS
    Three short pieces of text per film, written once, offline, by a model reading the
    Wikipedia plot. They are the only text in the corpus that speaks about FEELING and
    MEANING — a plot scene says what happens, this says what it is like.

WHY ONE CALL FOR THREE FIELDS
    They change for the same reason: Royson decided the text should say something
    different. Three files and three calls would cost 3x per run and buy independent
    tuning that is not yet wanted. Split premise out the day it needs its own model —
    not before. (Decided 5 Sep.)

WHY IT WRITES A FILE AND NOT THE DATABASE
    So the text can be READ before it is believed. A bad derivation that goes straight
    into vectors is invisible; a bad derivation sitting in data/derived.json is
    obvious. The loader is a separate script for the same reason fetch and load are
    separate: a step that talks to a model and a step that writes rows fail in
    different ways and should not share a failure.

WHICH MODEL
    backend/config.py -> DERIVE_PROVIDER / DERIVE_MODEL. NOT the agent's model: this
    job is offline and only quality matters, the agent's is live and speed matters.
    Nothing in this file names a vendor.

RUN
    python -m pipeline.derive_corpus                        all films -> data/derived.json
    python -m pipeline.derive_corpus --limit 5              first 5   -> data/derived.sample.json
    python -m pipeline.derive_corpus --titles "Alien" ...   named     -> data/derived.sample.json

A PARTIAL RUN NEVER WRITES data/derived.json. Half a corpus in the real file looks
exactly like a whole one, and the loader downstream cannot tell the difference. Samples
go to their own file so the real one is always all-or-nothing.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg                                                     # noqa: E402
from dotenv import load_dotenv                                     # noqa: E402

from backend.config import DATABASE_URL, DERIVE_MODEL, DERIVE_PROVIDER   # noqa: E402
from backend.models import derive_model                            # noqa: E402

load_dotenv()

OUT_FULL = "data/derived.json"
OUT_SAMPLE = "data/derived.sample.json"


# ══════════════════════════════════════════════════════════════════════════════
#  THE PROMPT — five blocks. To change what a field means, edit ONE block.
# ══════════════════════════════════════════════════════════════════════════════

SOURCES_RULE = """YOUR SOURCES
- The PLOT is your primary source. Trust it.
- The OVERVIEW is marketing copy. Use it only as a weak second signal. Where the
  overview's tone disagrees with the plot's, trust the plot."""

MOOD_FEEL_RULE = """"mood_feel" — what it FEELS LIKE to watch, in one or two sentences.

  START WITH THE FEELING. The FIRST WORD is a plain everyday feeling word someone
  would actually type into a search box — warm, lonely, uneasy, giddy, frantic,
  restless. Not "a sense of", not "a feeling of", not "an atmosphere of". The word
  itself, first.

  ONE feeling, not two. Never join two different feelings with "and" or "but": a
  sentence that leans two ways leans strongly at neither, and a search for either
  one misses.

  NO PLOT. Do not continue the sentence with "as ...", "while ...", "when ..." or any
  other clause describing what happens or who it happens to. "Uneasy, as a man
  discovers his hosts are hiding something" is plot wearing a feeling as a hat, and it
  puts plot words inside the one field that exists to hold none.
  Fill the rest of the sentence from the WATCHING instead: the pace, the quiet or the
  noise, how close or far the camera holds you, whether you know more than the people
  on screen or less, what the place feels like to sit inside.
  Name no character. Describe no event."""

THEME_RULE = """"theme" — what the film is ABOUT UNDERNEATH the plot, in one sentence.
  Not the subject. Jurassic Park's subject is dinosaurs; its theme is the illusion
  of control over what we create.

  State the idea directly. Never open with "The film ..." — those words are identical
  across every film, so they describe none of them."""

PREMISE_RULE = """"premise" — the SETUP, in one or two sentences: who, where, and what
  goes wrong to start the story. First act only."""

SPOILER_RULE = """SPOILER RULE — applies to all three fields
  Never state how the film ends.
  Never state who dies, survives, betrays, or is revealed to be someone else.
  Never name a twist, a reveal, or a final confrontation.

  AND NEVER NAME THE DESTINATION. Not only the events — the place the film arrives at.
  Do not say what the experience builds toward, what is finally earned, released,
  escaped or overcome, what a character learns or becomes, or what the story turns out
  to have really been about. "Builds toward a feeling of earned liberation" names the
  ending without naming an event. So does "learns to care for the boy". Both are
  spoilers under this rule.
  Describe the film from where a first-time viewer stands at the start, never from
  where the film leaves them.
  Write only what a trailer would be allowed to show."""

SPECIFICITY_RULE = """SPECIFICITY RULE
  Use words that fit THIS film and few others.
  Never use: tense, uplifting, bleak, coming of age, gripping, heartwarming.
  If the plot does not support something, leave it out. Do not guess."""

SYSTEM_PROMPT = f"""You read a film's Wikipedia plot and describe the film three ways.

Return only JSON, no other text.

{SOURCES_RULE}

THE THREE FIELDS

{MOOD_FEEL_RULE}

{THEME_RULE}

{PREMISE_RULE}

{SPOILER_RULE}

{SPECIFICITY_RULE}

Shape:
{{"mood_feel": "", "theme": "", "premise": ""}}"""

FIELDS = ("mood_feel", "theme", "premise")


# ══════════════════════════════════════════════════════════════════════════════
#  READING THE MODEL'S ANSWER
# ══════════════════════════════════════════════════════════════════════════════

def text_of(reply):
    """Pull the words out of a reply. Some models return a string; some a list.

    Gemini answers in a list of content blocks. `str()` on a list gives Python's repr —
    single quotes, list brackets — which is not JSON, and json.loads then fails at
    character 2 as though the MODEL had misbehaved. It had not. (Cost an hour, 3 Sep.)
    """
    content = reply.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def unwrap(text):
    """Parse the answer. Returns (fields, was_fenced).

    A model that wraps JSON in ```json fences has not returned JSON, it has returned a
    document containing JSON. The fence is stripped AND recorded, because a model that
    needs unwrapping today will change the shape of its wrapper one day.
    """
    raw = text.strip()
    fenced = raw.startswith("```")
    if fenced:
        body = raw.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
        raw = body.strip()
    return json.loads(raw), fenced


# ══════════════════════════════════════════════════════════════════════════════
#  THE RUN
# ══════════════════════════════════════════════════════════════════════════════

def films(titles=None, limit=None):
    """Plot first, overview second — the order the prompt asks the model to weigh them."""
    sql = """
        SELECT movie_id, title, wikidata_plot, tmdb_overview
        FROM movies
        WHERE (%(titles)s::text[] IS NULL OR title = ANY(%(titles)s::text[]))
        ORDER BY movie_id
    """
    if limit:
        sql += " LIMIT %(limit)s"
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute(sql, {"titles": titles, "limit": limit}).fetchall()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int,
                        help="derive only the first N films (writes the sample file)")
    parser.add_argument("--titles", nargs="+",
                        help="derive only these exact titles (writes the sample file)")
    args = parser.parse_args()

    rows = films(args.titles, args.limit)
    partial = bool(args.limit or args.titles)
    out = OUT_SAMPLE if partial else OUT_FULL

    if args.titles:
        found = {r[1] for r in rows}
        for wanted in args.titles:
            if wanted not in found:
                print(f"  !!  no film titled {wanted!r} — check the exact spelling")
    llm = derive_model()

    print(f"{len(rows)} films · {DERIVE_PROVIDER} · {DERIVE_MODEL}\n")

    derived, fenced_count, skipped = [], 0, 0

    for movie_id, title, plot, overview in rows:
        # No plot is a normal outcome, not an error. Deriving mood from a 350-character
        # marketing blurb is exactly the weakness this rewrite exists to fix, so a film
        # without a plot is skipped rather than derived badly.
        if not plot:
            skipped += 1
            print(f"  --  {title}  (no plot — skipped)")
            continue

        message = f"PLOT:\n{plot}\n\nOVERVIEW:\n{overview or '(none)'}"
        fields, fenced = unwrap(text_of(llm.invoke(
            [("system", SYSTEM_PROMPT), ("human", message)])))
        fenced_count += fenced

        derived.append({"movie_id": movie_id, "title": title,
                        **{f: fields.get(f, "") for f in FIELDS}})

        print(f"\n  {title}")
        for f in FIELDS:
            print(f"    {f:10} {fields.get(f, '')}")

    with open(out, "w") as handle:
        json.dump(derived, handle, indent=2, ensure_ascii=False)

    print(f"\n{len(derived)} films -> {out} · {skipped} skipped · {fenced_count} fenced")
    if partial:
        print("PARTIAL RUN — data/derived.json is untouched.")
    print("READ THE FILE before loading it. That is what this step is for.")


if __name__ == "__main__":
    main()

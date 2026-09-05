"""Which model should write the mood text? Settle it by measurement, not by taste.

THE QUESTION
    pipeline/derive_corpus.py asks a model to read a film's OVERVIEW and write back
    how the film feels. That text becomes a second chunk per film, it gets embedded,
    and it is the ONLY text in the corpus that speaks in mood language. A plot chunk
    says what happens. This says how it feels. So a mood query — "a warm, comforting
    film for a rainy evening" — has nowhere else to land.

THE OBSERVED DEFECT (3 Sep 2026, measured on the live data/derived.json)
    The prompt says   "feel" is ONE SENTENCE          ->  0 of 20 films comply
    The prompt says   never use "tense"/"uplifting"   ->  6 of 20 films break it
    Nova Micro is ignoring two explicit instructions on a third of the corpus.

WHY THIS JOB IS A FAIR PLACE FOR A SLOW MODEL
    It runs offline, once, over 20 films. Nobody is waiting. The 57s-per-answer that
    disqualified Gemini as the live AGENT (measured 2 Sep) is irrelevant here. This
    is the one job in the app where quality is the only axis that matters.

THE ARMS
    champion    the live data/derived.json, written by Nova Micro. NOT re-generated
                — the baseline must be the text actually in the database.
    challengers three Vertex models, chosen from Agent Studio's model list:
                newest Flash / the quality tier / the cheap floor.

WHAT IS MEASURED — three things, and only the third decides anything
    1. compliance   does the model do what the prompt said? Counted, not judged.
                    Free to compute. Diagnostic only: a model can follow every
                    instruction and still write text that retrieves nothing.
    2. clean JSON   did it answer in JSON, or wrap it in markdown fences? A model
                    that needs unwrapping is a model that will break the pipeline
                    one day when the fence changes shape.
    3. achievable@3 THE HEADLINE. Of the mood answers that could fit in a top 3,
                    how many did this model's text actually surface?
                    Ground truth is data/golden_set_mood.json — five queries and the
                    films Royson said feel that way, written by hand on 24 Aug. Not
                    an LLM judge. Which films feel warm is taste, and taste is his.

WHAT THIS DOES *NOT* MEASURE
    Only the mood chunk is searched here — no plot chunks, no reranker, no graph.
    That is deliberate: the mood text is the ONE variable, so nothing else may move.
    It answers "does this text pull the right film up", not "is the whole app better".
    End-to-end is evals/eval_agent.py, and it only runs on the winner.

THE NOISE FLOOR — read this before reading any number below
    5 queries, ceiling 15 reachable answers. ONE film moving in or out is 6.7 points.
    A gap smaller than that is not a result. It is the same number twice.

COST CONTROL
    A query's vector does not depend on the arm, so each of the 5 mood queries is
    embedded ONCE and reused across all four arms — 5 embeddings, not 20.
    Derived text is cached to data/mood_bakeoff/<arm>.json. Re-running scores the
    cached text and costs nothing; delete a file to force one arm to regenerate.

USAGE
    python -m experiments.mood_bakeoff --probe   one film, every model, printed. Do
                                                 this FIRST. It is the cheap test
                                                 that could kill the idea before the
                                                 expensive one that might support it.
    python -m experiments.mood_bakeoff           the full run.
"""

import argparse
import ast
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg                                                     # noqa: E402
from dotenv import load_dotenv                                     # noqa: E402

from backend.config import (DATABASE_URL, GCP_LOCATION,            # noqa: E402
                            GCP_PROJECT, REGION)
from backend.models import embed                                   # noqa: E402

load_dotenv()

GOLDEN = "data/golden_set_mood.json"
LIVE_DERIVED = "data/derived.json"
CACHE_DIR = "data/mood_bakeoff"
TOP_N = 3
BANNED = ("tense", "coming of age", "uplifting", "bleak")
SENTENCE_WORDS = 6          # below this, "feel" is a fragment, not a sentence

# derive_corpus.py caps Nova at 300, which is generous for a model that answers
# immediately. A reasoning model spends tokens THINKING before it writes, and those
# count against the same cap — gemini-2.5-pro was truncated mid-sentence at 300 on
# 3 Sep. The cap is a safety rail against a runaway, not a length instruction: the
# prompt asks for one sentence and 2-4 words per list, and that is what bounds the
# ANSWER. Raising it changes what fits, not what is asked for.
MAX_TOKENS = 2048

# (label, provider, model_id). provider "live" means: read the file, call nothing.
ARMS = [
    ("champion  nova (live)", "live",   None),
    ("gemini-3.8-flash",      "vertex", "gemini-3.8-flash"),
    ("gemini-2.5-pro",        "vertex", "gemini-2.5-pro"),
    ("gemini-2.5-flash-lite", "vertex", "gemini-2.5-flash-lite"),
]


def system_prompt():
    """Read SYSTEM_PROMPT out of pipeline/derive_corpus.py rather than copying it.

    A copy would drift, and a drifted prompt makes the whole comparison a lie — the
    arms would differ by model AND by instructions, and no number could be attributed
    to either. derive_corpus.py cannot be imported (it connects to the database at
    import time), so the source is parsed instead. One prompt exists, by construction.
    """
    tree = ast.parse(open("pipeline/derive_corpus.py", encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "SYSTEM_PROMPT" for t in node.targets):
            return node.value.value
    raise RuntimeError("SYSTEM_PROMPT not found in pipeline/derive_corpus.py")


def films():
    """The same query derive_corpus.py runs — overview only, never the plot."""
    with psycopg.connect(DATABASE_URL) as conn:
        return conn.execute(
            "SELECT movie_id, title, overview FROM movies ORDER BY movie_id"
        ).fetchall()


def writer_for(model_id):
    """Build one Vertex chat model and return a function that derives one film.

    The vendor class is named here and nowhere else in this file. Same reasoning as
    backend/models.chat_model(): the import is inside the function so a Bedrock-only
    environment never needs the Google packages to run the champion arm.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=model_id, vertexai=True, project=GCP_PROJECT,
                                 location=GCP_LOCATION, temperature=0,
                                 max_output_tokens=MAX_TOKENS)
    prompt = system_prompt()

    def derive(overview):
        return unwrap(text_of(llm.invoke([("system", prompt), ("human", overview)])))

    return derive


def text_of(reply):
    """Pull the words out of a reply. Some models return a string; some a list.

    WHY THIS EXISTS — it did not, and that was a bug on 3 Sep. The first version did
    `str(reply.content)`, which on a list produces Python's repr:

        [{'type': 'text', 'text': '{"feel": ...

    Single quotes, and a list wrapper. Not JSON. json.loads failed at character 2 and
    it looked like the MODEL had misbehaved when the fault was entirely local.

    This is the same shape as the reasoning-block bug in backend/agent.py: two halves
    that are each correct, and no test of the join. A model that answers in blocks is
    normal, not exotic, and every caller of .content has to expect either shape.
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
    """Parse the model's answer. Returns (tags, was_fenced).

    A model that wraps JSON in ```json fences has not returned JSON, it has returned
    a document containing JSON. derive_corpus.py would crash on it. That is worth
    counting rather than silently forgiving — so the fence is stripped AND recorded.
    """
    raw = text.strip()
    fenced = raw.startswith("```")
    if fenced:
        body = raw.split("```")[1]
        if body.startswith("json"):
            body = body[4:]
        raw = body.strip()
    return json.loads(raw), fenced


def derive_all(model_id, rows):
    """Derive every film with one model, printing as it goes so it can be watched."""
    write = writer_for(model_id)
    out, fences = [], 0
    for movie_id, title, overview in rows:
        tags, fenced = write(overview)
        fences += fenced
        out.append({"id": movie_id, "title": title, **tags})
        print(f"    {title:32} {tags.get('feel', '')[:60]}")
        time.sleep(0.1)
    return out, fences


def cached(label, model_id, rows):
    """Derived text for one arm — from disk if present, from the model if not."""
    if model_id is None:
        return json.load(open(LIVE_DERIVED)), 0

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{model_id}.json")
    if os.path.exists(path):
        saved = json.load(open(path))
        print(f"    (cached — delete {path} to regenerate)")
        return saved["films"], saved["fences"]

    print(f"  deriving 20 films with {model_id} ...")
    entries, fences = derive_all(model_id, rows)
    json.dump({"model": model_id, "fences": fences, "films": entries},
              open(path, "w"), indent=2)
    return entries, fences


def mood_text(entry):
    """The EXACT string pipeline/load_derived.py stores and embeds.

    If this drifts from load_derived.py, the bake-off scores text the app would never
    hold, and the winner would not reproduce once promoted.
    """
    return (f"{entry.get('feel', '')}. "
            f"Moods: {', '.join(entry.get('moods', []))}. "
            f"Themes: {', '.join(entry.get('themes', []))}.")


def compliance(entries):
    """Count the two instructions the prompt states outright. No judgement involved."""
    banned = sum(
        any(word in (e.get("feel", "") + " " + " ".join(e.get("moods", []))).lower()
            for word in BANNED)
        for e in entries)
    sentences = sum(len(e.get("feel", "").split()) >= SENTENCE_WORDS for e in entries)
    return banned, sentences


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def achievable(entries, cases, query_vectors):
    """Of the answers that COULD fit in a top 3, how many did this text surface?

    Only TOP_N films fit in a top-N list, so a case naming 6 films can never be fully
    satisfied. Scoring against 6 would make "impossible" and "failed" look identical,
    so the denominator is what was actually reachable.
    """
    vectors = {e["title"]: embed(mood_text(e)) for e in entries}
    hits = ceiling = 0
    per_case = {}
    for case in cases:
        ranked = sorted(vectors, key=lambda t: cosine(query_vectors[case["id"]],
                                                      vectors[t]), reverse=True)
        top = ranked[:TOP_N]
        found = sum(1 for want in case["expect"] if want in top)
        hits += found
        ceiling += min(len(case["expect"]), TOP_N)
        per_case[case["id"]] = (found, min(len(case["expect"]), TOP_N), top)
    return hits, ceiling, per_case


def probe(rows):
    """The cheap test that could kill the idea. One film, every model, printed raw."""
    movie_id, title, overview = rows[0]
    print(f"\nPROBE — one film, every challenger, no scoring\n\n  {title}\n"
          f"  overview: {overview[:200]}...\n")
    live = {e["title"]: e for e in json.load(open(LIVE_DERIVED))}.get(title, {})
    print(f"  {'champion  nova (live)':24} {json.dumps(live, indent=2)}\n")
    for label, provider, model_id in ARMS:
        if provider != "vertex":
            continue
        try:
            tags, fenced = writer_for(model_id)(overview)
            fence = "  [WRAPPED IN FENCES]" if fenced else ""
            print(f"  {label:24} {json.dumps(tags, indent=2)}{fence}\n")
        except Exception as broke:                    # a model that errors is a result
            print(f"  {label:24} FAILED: {type(broke).__name__}: {broke}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                        help="one film through every model, printed, no scoring")
    args = parser.parse_args()

    rows = films()
    if args.probe:
        return probe(rows)

    cases = json.load(open(GOLDEN))["cases"]
    print(f"{len(rows)} films · {len(cases)} mood queries · "
          f"{sum(min(len(c['expect']), TOP_N) for c in cases)} reachable answers\n")

    print("embedding each mood query once (reused across all arms)...")
    query_vectors = {c["id"]: embed(c["query"]) for c in cases}

    results = {}
    for label, provider, model_id in ARMS:
        print(f"\narm {label}")
        entries, fences = cached(label, model_id, rows)
        banned, sentences = compliance(entries)
        hits, ceiling, per_case = achievable(entries, cases, query_vectors)
        results[label] = {"banned": banned, "sentences": sentences, "fences": fences,
                          "hits": hits, "ceiling": ceiling, "per_case": per_case,
                          "achievable": hits / ceiling if ceiling else 0.0,
                          "films": len(entries)}

    floor = 1 / next(iter(results.values()))["ceiling"] * 100

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    print(f"  {'arm':24} {'achievable@3':>13} {'hits':>8}   "
          f"{'banned words':>13} {'sentences':>10} {'fenced':>7}")
    print(f"  {'-'*24} {'-'*13} {'-'*8}   {'-'*13} {'-'*10} {'-'*7}")
    best = max(r["achievable"] for r in results.values())
    for label, r in results.items():
        star = "  <-- best" if r["achievable"] == best else ""
        print(f"  {label:24} {r['achievable']*100:12.1f}% {r['hits']:4}/{r['ceiling']:<3}"
              f"   {r['banned']:6}/{r['films']:<6} {r['sentences']:5}/{r['films']:<4}"
              f" {r['fences']:7}{star}")

    print(f"\n  NOISE FLOOR: one film moving is {floor:.1f} points. A gap smaller "
          f"than that is NOT a result.")
    spread = (best - min(r['achievable'] for r in results.values())) * 100
    print(f"  Spread between best and worst arm: {spread:.1f} points — "
          f"{'ABOVE' if spread > floor else 'BELOW'} the floor.")
    print("  banned words / sentences are DIAGNOSTIC. They say the model followed the")
    print("  brief. Only achievable@3 says the corpus got better.")

    print("\n" + "=" * 78)
    print("WHERE THE ARMS DISAGREE")
    print("=" * 78)
    labels = list(results)
    for case in cases:
        scores = [results[l]["per_case"][case["id"]][0] for l in labels]
        if len(set(scores)) == 1:
            continue
        print(f"\n  [{case['id']}] \"{case['query']}\"")
        print(f"       expect: {', '.join(case['expect'])}")
        for l in labels:
            found, reach, top = results[l]["per_case"][case["id"]]
            print(f"       {l:24} {found}/{reach}   {', '.join(top)}")

    out = {l: {k: v for k, v in r.items() if k != "per_case"} for l, r in results.items()}
    json.dump(out, open("data/mood_bakeoff/results.json", "w"), indent=2)
    print("\nsaved data/mood_bakeoff/results.json")


if __name__ == "__main__":
    main()

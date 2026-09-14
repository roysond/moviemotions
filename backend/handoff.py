"""THE CONTRACT BETWEEN THE REASONER AND THE WRITER.

WHY A CONTRACT AND NOT A CONVERSATION
    Two models handing prose to each other is how a fact gets invented in the gap.
    The second one reads a sentence, likes the shape of it, and finishes the thought
    with something nobody retrieved — and because the output is prose either way,
    nothing downstream can tell the difference.

    So the handoff is a RECORD, not a paragraph:

        {"films": [{"title": "Predator", "year": 1987,
                    "why": ["creatures hunting people in a jungle"]}],
         "nothing_found": false}

    The writer receives that and the user's question, and NOTHING else. No premise, no
    scores, no tool output, no history. It cannot smuggle in a fact it was never given,
    because it was never given any facts beyond these — the guarantee is structural
    rather than a rule in a prompt, and a rule in a prompt is a request.

WHY THIS FILE HAS NO MODEL IN IT
    Everything here is a pure function over text. Parsing and validation are decisions
    CODE is allowed to make, so they are made where they can be tested without a
    network, a key, or a bill. `backend/agent.py` owns the calls; this file owns the
    shape.

THE ONE CHECK THAT MATTERS
    `validate` drops any film whose title and year do not appear in the evidence the
    tools actually returned. That is the hallucination gate, and it is mechanical: a
    film the catalogue never mentioned cannot reach the writer, whatever the reasoner
    believed about it.
"""

import json
import re

# At most three films, because the answer format is at most three. A reasoner that
# names five has misread the floor, and truncating is a kinder failure than refusing.
MAX_FILMS = 3

# At most two reasons each, and each one short. A "why" is a PHRASE, not a sentence —
# "creatures hunting people in a jungle", not "This film is a great match because...".
# The writer builds the sentence; if the reasoner writes it first, the writer has
# nothing to do but copy, which is exactly the behaviour being designed out.
MAX_REASONS = 2
MAX_REASON_CHARS = 120

# The tool prints each film as `1. Predator (1987) · 107 min · relevance 0.711`.
# Title and year together, because a title alone is not an identity: remakes exist.
TITLE_IN_EVIDENCE = re.compile(r"^\s*\d+\.\s+(.+?)\s+\((\d{4})\)", re.M)

# ```json ... ``` — small models fence their JSON even when told not to. Stripping the
# fence in code is cheaper than a fourth attempt at asking them to stop.
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


def films_in_evidence(evidence):
    """Every film the tools actually returned, as {(lowercased title, year): title}.

    The returned title is the ORIGINAL spelling, so a downstream answer can print the
    catalogue's version rather than the model's approximation of it.
    """
    return {(title.strip().lower(), int(year)): title.strip()
            for title, year in TITLE_IN_EVIDENCE.findall(evidence)}


def extract(text):
    """Pull the JSON object out of a reply. Returns a dict, or None.

    Tolerant on purpose. A reasoner told to reply with JSON and nothing else will
    still occasionally wrap it in a fence or add a line of throat-clearing, and
    failing the whole turn over a stray backtick would be a self-inflicted outage.
    What is NOT tolerated is guessing: if there is no object here, this returns None
    and the caller decides what to do about it.
    """
    if not text:
        return None
    cleaned = FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start:end + 1])
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _clean_reasons(raw):
    """Reasons as a list of short, non-empty phrases."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        phrase = str(item).strip().rstrip(".")
        if phrase:
            out.append(phrase[:MAX_REASON_CHARS])
    return out[:MAX_REASONS]


def validate(payload, evidence):
    """Return (verdict, problems).

    `verdict` is always usable — a valid record, possibly empty. `problems` is a list
    of plain sentences describing what was thrown away and why.

    NOTHING IS EVER REPAIRED BY GUESSING. A film that does not match the evidence is
    DROPPED, never corrected to the nearest title: a near-miss correction turns a
    visible error into an invisible one, and an invisible one ships.
    """
    problems = []
    known = films_in_evidence(evidence)

    films = payload.get("films") if isinstance(payload, dict) else None
    if not isinstance(films, list):
        films = []
        problems.append("The verdict carried no list of films.")

    kept, seen = [], set()
    for entry in films:
        if not isinstance(entry, dict):
            problems.append("A film was not a record and was dropped.")
            continue
        title = str(entry.get("title", "")).strip()
        try:
            year = int(entry.get("year"))
        except (TypeError, ValueError):
            problems.append(f"{title or 'A film'} had no usable year and was dropped.")
            continue

        identity = (title.lower(), year)
        if identity not in known:
            # THE GATE. The reasoner named something the tools never returned.
            problems.append(f"{title} ({year}) is not in what the tools returned "
                            f"and was dropped.")
            continue
        if identity in seen:
            problems.append(f"{title} ({year}) was named twice; the repeat was dropped.")
            continue
        seen.add(identity)

        reasons = _clean_reasons(entry.get("why"))
        if not reasons:
            problems.append(f"{title} ({year}) came with no reason and was dropped. "
                            f"A recommendation with no reason is a name, not an answer.")
            continue

        kept.append({"title": known[identity], "year": year, "why": reasons})

    if len(kept) > MAX_FILMS:
        problems.append(f"{len(kept)} films were named; kept the first {MAX_FILMS}.")
        kept = kept[:MAX_FILMS]

    # `nothing_found` is DERIVED, never trusted. A reasoner that says it found nothing
    # and then lists two films has contradicted itself, and the list is the evidence.
    return {"films": kept, "nothing_found": not kept}, problems


def summarise(verdict):
    """The verdict as one line, for a trace or a console. Not for a user."""
    if verdict["nothing_found"]:
        return "nothing found"
    return " | ".join(f"{f['title']} ({f['year']}): {'; '.join(f['why'])}"
                      for f in verdict["films"])


def for_writer(verdict):
    """The verdict as the writer sees it. This is the writer's ENTIRE world.

    Deliberately not JSON. The writer is a language model being asked for language;
    handing it braces invites it to answer in braces. The structure has already done
    its job by the time it reaches here — it survived `validate`, so every title on
    this page came from the catalogue.
    """
    blocks = []
    for film in verdict["films"]:
        reasons = "\n".join(f"  - {why}" for why in film["why"])
        blocks.append(f"{film['title']} ({film['year']})\n{reasons}")
    return "\n\n".join(blocks)

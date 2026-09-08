"""What the agent is allowed to do.

THE DOCSTRING IS THE INTERFACE
    Only a tool's NAME, ARGUMENTS and DESCRIPTION travel to the model. The code never
    leaves this machine. So everything the model must know — when to use a tool, when
    NOT to, what a number means, how to phrase an argument — has to be written in prose
    here, and nowhere else. A rule that lives in a comment is a rule the model cannot
    read.

TWO TOOLS, AND WHY NOT THREE
    A "similar to X" search was nearly its own tool. It is an ARGUMENT instead, because
    "similar to Alien, but funnier" needs a comparison film AND a mood in the SAME call.
    Two separate tools cannot express that sentence at all — the model would have to
    pick one half and throw the other away. Where two tools would have to be called
    together, they were always one tool.

    The two that remain key off different things in the sentence, so there is no
    judgement call: search_films answers "find me something", lookup_film answers
    "tell me about this one".

WHAT A SCORE IS NOT
    Closeness, never correctness. Measured 7 Sep 2026: unrelated text scores about 0.64
    in this embedding space and nothing ever scores below 0.6, so a fixed cut-off cannot
    work. The GAP line in each result is the honest signal, and it is reported rather
    than acted on — refusing is the agent's judgement to make.

RUN
    python -m backend.tools        the spec the model receives, then real calls
"""

import psycopg
from langchain_core.tools import tool

from backend.config import DATABASE_URL
from backend.retrieval import DISPLAYABLE, resolve_title, search

# How to say a data_kind out loud, for the one line the model reads.
KIND_NAME = {"mood_feel": "feel", "premise": "premise", "theme": "theme",
             "plot_scene": "plot"}

def render(results, notes):
    """One block of text for the model. Evidence beside every claim.

    Every film carries the sentence it MATCHED on and its spoiler-free premise, because
    a tool that returns a film and no words leaves a gap, and a model fills gaps from
    its training. That was learned twice on this project, a month apart.

    TWO CONSTRAINTS CAN MATCH THE SAME SENTENCE. "Like Alien" and "funny" both read a
    film's one mood_feel row, so printing per constraint showed the identical sentence
    twice and dropped the two DIFFERENT scores — the only part that was not identical.
    The text is printed once, with every score that was measured against it.
    """
    lines = []
    if not results:
        lines.append("No films matched.")
    else:
        # "best first" is a CLAIM. It is true of a ranked result and false of a listed
        # one, and a header that contradicts the note below it is the same failure as a
        # trace describing a pipeline that no longer runs.
        if results[0].get("listed_only"):
            lines.append(f"{len(results)} films carrying that fact, in no "
                         f"meaningful order.")
        else:
            lines.append(f"{len(results)} films, best first.")
        for n, film in enumerate(results, 1):
            lines.append("")
            # rank is what ORDERED the films; raw and the margin are what say
            # whether any of them is actually good. Both are printed because rank
            # alone always makes the winner look perfect.
            head = (f'{n}. {film["title"]} ({film["year"]}) · '
                    f'{film["runtime_minutes"]} min')
            if film.get("listed_only"):
                # No score at all, deliberately. A number here would invent a
                # confidence nobody measured.
                lines.append(head)
                shown = film.get("show", {})
                if shown.get("mood_feel"):
                    lines.append(f'   FEELS   {shown["mood_feel"]}')
                if shown.get("premise"):
                    lines.append(f'   PREMISE {shown["premise"]}')
                continue
            if "rerank" in film:
                # The number that actually decided this order. Shown FIRST because a
                # reader — model or person — takes the first figure as the verdict, and
                # after a rerank the vector numbers are history, not the decision.
                head += f' · relevance {film["rerank"]:.3f}'
                head += f' (vector had it #{film["vector_place"]})'
            else:
                head += f' · rank {film["score"]:.2f}'
                if "over_floor" in film:
                    head += f' · {film["over_floor"]:+.3f} over its band'
            lines.append(head)

            scores = film.get("per_constraint", {})

            def measured(label):
                hit = scores[label]
                return (f'rank {hit["rank"]:.2f} among text of that kind, '
                        f'raw {hit["raw"]:.3f}')

            for label, kind in film.get("matched_kind", {}).items():
                where = KIND_NAME.get(kind, kind)
                lines.append(f"   MATCHED {label} — on its {where}, {measured(label)}")
            for label in film.get("matched_unseen", []):
                lines.append(
                    f"   MATCHED {label} — {measured(label)} — on text that is "
                    f"withheld because it gives away endings. Rank on it. Do not "
                    f"describe the film from it and do not invent what it said.")

            # BOTH surfaces, every time, whichever constraint did the matching. A film
            # returned with only its premise leaves nothing to say except the plot,
            # and "say why it fits, do not retell the plot" is then an impossible
            # instruction. FEELS is what an answer is written from; PREMISE is what
            # keeps it true.
            shown = film.get("show", {})
            if shown.get("mood_feel"):
                lines.append(f'   FEELS   {shown["mood_feel"]}')
            if shown.get("premise"):
                lines.append(f'   PREMISE {shown["premise"]}')

    if notes:
        lines.append("")
        lines.append("NOTES")
        lines += [f"- {note}" for note in notes]
    return "\n".join(lines)


def search_films(mood: str = "", situation: str = "", theme: str = "",
                 similar_to: str = "", director: str = "", actor: str = "",
                 genre: str = "", max_runtime: int | None = None,
                 min_year: int | None = None, max_year: int | None = None) -> str:
    """Find films by how they FEEL, what happens in them, or by naming one the user liked.

    Use this whenever the user is asking to be given a film. Fill in only the arguments
    their sentence actually supports and leave the rest empty — an argument you invent
    is a requirement they never stated, and every one of them narrows the search.

    mood
        How they want the film to FEEL, in FEELING WORDS. "Warm and comforting, cosy",
        not "a warm gentle feel-good film for a rainy evening" — the stored text is one
        tight sentence about a feeling, so words like "film", "movie" and "evening"
        dilute the query rather than sharpen it. Expand a one-word request into a few
        feeling words. Do not pad it into a paragraph.

    situation
        What actually HAPPENS. "A group trapped somewhere with no way out." Use this
        when they describe events rather than a feeling. It searches both a film's
        opening setup and its individual scenes, so it can find a moment from the middle
        of a film, not only films that begin that way. Some of what it matches on cannot
        be shown to you, because a scene from late in a film gives the ending away.

    theme
        What the film is about UNDERNEATH, when they ask for that specifically: "films
        about the illusion of control". Leave empty otherwise. It is a weak signal on
        its own — abstract sentences resemble each other — so never set it as a guess.

    similar_to
        The EXACT TITLE of a film they named as a reference point. This searches by what
        that film feels like, and excludes the film itself and its sequels.
        A sentence with both a film and a description — "like Alien, but funnier" — sets
        similar_to AND mood, in one call. Both must be satisfied, so expect lower scores
        and a smaller field; that is the request being hard, not the search failing.

    director, actor, genre
        A FACT, not a feeling. Set these when the user names a person or a genre, and
        never as a guess. They are checked against a knowledge graph, so the answer is
        yes or no — a director either made a film or did not — and films without that
        edge are removed before anything is ranked.
        Never put a person's name or a genre into `mood` or `situation` instead. A vector
        would return films that FEEL like that person's work, which is a different
        question answered confidently and wrongly.
        A name that is not in the catalogue is reported as a plain fact. That is not a
        weak match to work around — the name is simply not here, so say so.

    max_runtime, min_year, max_year
        Only when stated out loud. These are enforced exactly and delete films that do
        not qualify, so a limit you assumed can remove the one right answer and leave no
        trace. "Nothing too long" is not a number. Ask rather than guess.

    Returns each film with the sentence it matched on, its spoiler-free premise, and a
    NOTES block. Read the NOTES: they say what the filters removed and how far the top
    film stood above the rest of the field. A small gap means nothing genuinely stood
    out, and saying so plainly is a better answer than the least-bad film.

    Do NOT use this to look up a film the user already named and simply wants details
    about — that is lookup_film.
    """
    results, notes = search(
        mood=mood or None, situation=situation or None, theme=theme or None,
        similar_to=similar_to or None, director=director or None,
        actor=actor or None, genre=genre or None, max_runtime=max_runtime,
        min_year=min_year, max_year=max_year, limit=5)
    return render(results, notes)


def lookup_film(title: str) -> str:
    """Tell the user about ONE film they have already named. No searching, no ranking.

    Use this when the sentence is ABOUT a film — "what is Alien about", "how long is
    Titanic" — rather than a request to be given one.

    If the title is not in the catalogue, or matches several films, that is said plainly.
    Neither is an error and neither is answered by guessing: ask which one they meant.

    Do NOT use this to find films LIKE the one named. That is search_films with
    similar_to.
    """
    return _one_film(title)


ONE = """
SELECT m.title, EXTRACT(YEAR FROM m.release_date)::int, m.runtime_minutes,
       d.data_kind, d.content
FROM movies m
LEFT JOIN movie_data d ON d.movie_id = m.movie_id AND d.seq = 0
WHERE lower(m.title) = lower(%(title)s)
"""


def _one_film(title):
    """The film's own facts and its spoiler-free text. Never its theme — see retrieval.

    DISPLAYABLE is imported rather than re-listed. Two files that each decide for
    themselves what may be shown will eventually disagree, and the one that gets it
    wrong prints a spoiler.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        found, problem = resolve_title(conn, title)
        if problem:
            return problem
        rows = conn.execute(ONE, {"title": found[1]}).fetchall()

    if not rows:
        return f"{title!r} is not in the catalogue."

    name, year, runtime, _, _ = rows[0]
    text = {kind: content for _, _, _, kind, content in rows
            if kind in DISPLAYABLE}
    lines = [f"{name} ({year}) · {runtime} min"]
    if "premise" in text:
        lines.append(f"   PREMISE {text['premise']}")
    if "mood_feel" in text:
        lines.append(f"   FEELS   {text['mood_feel']}")
    return "\n".join(lines)


# ── TRANSPORTS ────────────────────────────────────────────────────────────────
# The two functions above are PLAIN PYTHON. They do not import LangChain, they do not
# know what MCP is, and they return a string. Each way of reaching them wraps them here.
#
# That is the whole point: `backend/mcp_server.py` exposes the SAME functions to any
# MCP client without a second definition. A tool defined twice is a tool that will
# eventually behave two ways, and the divergence shows up in whichever transport nobody
# is testing that week.
SEARCH_FILMS = tool(search_films)
LOOKUP_FILM = tool(lookup_film)

TOOLS = [SEARCH_FILMS, LOOKUP_FILM]


if __name__ == "__main__":
    print("=" * 78)
    print("THE SPEC THE MODEL RECEIVES — this text, and nothing else, is what it reads")
    print("=" * 78)
    for spec in TOOLS:
        print(f"\n\n### {spec.name}{spec.args}\n")
        print(spec.description)

    print("\n\n" + "=" * 78)
    print("REAL CALLS")
    print("=" * 78)
    for call in [
        {"mood": "warm and comforting, cosy"},
        {"similar_to": "Toy Story"},
        {"similar_to": "Alien", "mood": "funny and light-hearted"},
        {"mood": "funny and chaotic", "max_runtime": 100},

        # SITUATION — the only calls that can reach a plot scene. Without these the
        # self-test never touches 148 of the 208 rows in the table, and adding a whole
        # corpus that nothing exercises is how you get a feature nobody notices is dead.
        {"situation": "a group trapped somewhere with no way out"},
        {"situation": "someone hunted through a jungle"},
        {"situation": "a child left alone in a house"},
        {"situation": "a prisoner digging a tunnel"},

        # THE GRAPH — questions a vector cannot answer at all, only a fact can.
        {"genre": "Science Fiction", "mood": "uneasy, dread"},
        {"director": "Nolan"},
        {"actor": "Schwarzenegger"},
        {"director": "Kurosawa"},          # not in the catalogue — must say so plainly
    ]:
        print(f"\n\n--- search_films({call})\n")
        print(search_films(**call))

    for title in ["Alien", "alien", "Karate", "Jaws"]:
        print(f"\n\n--- lookup_film({title!r})\n")
        print(lookup_film(title))

"""The tool registry — what the agent is allowed to do.

A TOOL IS TWO THINGS
    1. A normal Python function. Ordinary code, nothing special.
    2. A description written FOR THE MODEL. The model never sees the code — it reads
       the name, the docstring and the argument types, then replies asking for the tool
       by name. Our code decides whether to honour that request and runs it locally.
       Nothing about the database, the credentials or the SQL ever leaves this machine.

SO THE DOCSTRING IS THE INTERFACE
    It is prompt engineering aimed at a function signature. A vague description means a
    tool that is never called, or called with nonsense. Each description here states:
        - what it does
        - WHEN to use it        (so it gets picked at the right moment)
        - WHEN NOT to use it    (this prevents more errors than the previous line)
        - what comes back       (so the model knows how to read the result)

TWO KINDS OF ARGUMENT, AND THE WHOLE DESIGN IS IN THE SPLIT
    `query` is SOFT MEANING — mood, situation, plot. No exact test exists for "tense", so
    it goes to the embedding and the reranker, which judge by reading.

    The rest are HARD CONSTRAINTS — runtime and year. Each has a yes/no test, so each goes
    to a SQL WHERE clause and is enforced exactly. They are never embedded, because vectors
    capture topic, not truth value: "under 2 hours" embeds as mood, and "not a cartoon"
    embeds NEXT TO "a cartoon" — there is no minus sign in vector space.

        the LLM EXTRACTS the constraint   (fuzzy: "I've only got 90 minutes" -> 90)
        the DATABASE ENFORCES it          (exact: runtime_minutes <= 90)

    Never ask the fuzzy machine a question the exact machine can answer.
"""

from typing import Optional

from langchain_core.tools import tool

from core import get_film, graph_find, search

MAX_RESULTS = 5


@tool
def search_films(
    query: str,
    max_runtime: Optional[int] = None,
    min_runtime: Optional[int] = None,
    after_year: Optional[int] = None,
    before_year: Optional[int] = None,
    exclude_title: Optional[str] = None,
) -> str:
    """Find films whose plot, mood or situation matches a natural-language description.

    Use this whenever the user describes what they want to watch — a feeling
    ("something cosy"), a situation ("creatures chasing people"), or a plot
    ("a man wrongly imprisoned").

    EXPAND what the user said into a fuller description before calling. Never send a
    single word or a bare keyword — short queries score flat and rank nothing. Measured
    on this catalogue:
        query="cosy"                                  -> top score 0.08, spread 0.007
                                                         (noise: Hangover, Predator)
        query="a warm gentle feel-good film for a
               rainy evening at home"                 -> top score 0.37
                                                         (Finding Nemo, Crazy Stupid Love)
    Same request, same corpus, four times the signal. The only difference was wording.

    Aim for a full sentence naming the FEELING and the KIND OF STORY. Keep everything
    the user said and add the words they implied. You may call this more than once with
    different wordings if the first results look wrong.

    Do NOT use this when the user NAMES a film ("tell me about Predator", "how long
    is Titanic?") — use `lookup_film` for that. Do NOT use it to answer a question
    about a film already discussed in this conversation, or for anything unrelated
    to films — answer those directly instead.

    ARGUMENTS
    query: the mood, situation or plot ONLY, as a full descriptive phrase — never a
        single word. Strip out any length or date wording and put it in the arguments
        below instead, but do NOT strip anything else:
            "something tense with creatures under two hours"
              -> query="a tense, frightening film where dangerous creatures hunt
                        and kill people", max_runtime=120
    max_runtime / min_runtime: length in MINUTES. Translate the user's phrasing —
        "under two hours" -> 120, "I've only got 90 minutes" -> 90, "nothing short"
        -> min_runtime=100. Leave out entirely when the user says nothing about length.
    after_year / before_year: four-digit years, inclusive. "something modern" or
        "from the 90s" -> after_year=1990, before_year=1999. Leave out if not asked.
    exclude_title: a film to keep OUT of the results. Set it whenever the user says
        "like <film>", "similar to <film>", or "another <film>" — they already know
        that one and are asking to move past it.

    "SOMETHING LIKE <FILM>, BUT ..." — read this before writing the query.
    Describe what the film IS LIKE TO WATCH, not what it contains. A film's subject
    matter is unique to it, so searching for the subject can only find that one film
    back. Its EXPERIENCE is shared with others, and others is what the user wants.

        "like Jurassic Park but more intense, with gore"
          WRONG  query="a thrilling intense film with dinosaur attacks and gore"
                 -> dinosaurs exist in exactly one film here. Top result: Jurassic
                    Park, the film they asked to move past. Everything else scores
                    near zero because nothing else has dinosaurs.
          RIGHT  query="a tense, frightening film where people are hunted by
                        dangerous creatures and barely escape being killed",
                 exclude_title="Jurassic Park"

    Only keep the subject matter when the USER asked for it — "another film with
    dinosaurs" means dinosaurs; "something like Jurassic Park" means the feeling.

    These are enforced exactly, so a film that comes back ALWAYS satisfies them —
    never re-check or apologise for them, and never mention a film the tool did not
    return just because you believe it would fit.

    This tool has no genre, cast or director filter. If the user names a GENRE, an
    ACTOR or a DIRECTOR, that is `find_films_by_fact`, not this — those are facts with
    exact answers and this tool can only guess at them.

    Each result carries the QUOTED TEXT that matched. Base everything you say about a
    film on that quote. If the quote does not support a claim, do not make the claim —
    naming the right film for an invented reason is still wrong.

    Returns up to 5 films with relevance scores. Judge the results by the TOP
    score and by the gap below it, not by how many rows came back — this tool
    always returns something, so a list is not evidence of a match.
        above ~0.40   a genuine match. Recommend it plainly.
        0.25 to 0.40  UNCERTAIN. Do NOT refuse, and do NOT oversell. Offer the top
                      one or two tentatively and say they are not a strong match:
                      "nothing here is exactly that, but the closest is X".
        below ~0.25   nothing in the catalogue fits. Say so plainly instead of
                      offering weak suggestions.
    A steep drop after the first result (e.g. 0.58 then 0.19) means only the
    first one is real.
    """
    filters = {
        "max_runtime": max_runtime, "min_runtime": min_runtime,
        "after_year": after_year, "before_year": before_year,
    }
    asked = {k: v for k, v in filters.items() if v is not None}
    films = search(query, limit=MAX_RESULTS, **filters)

    if not films:
        # A filter emptying the pool and a description matching nothing are DIFFERENT
        # failures and need different replies, so the tool result says which one happened.
        if asked:
            return ("No films at all satisfy those hard constraints "
                    f"({', '.join(f'{k}={v}' for k, v in asked.items())}). "
                    "The limits ruled everything out, not the description — tell the user "
                    "plainly and offer to relax the limit. Do not search again unchanged.")
        return "No films found."

    lines = []
    if asked:
        lines.append("filters enforced: " + ", ".join(f"{k}={v}" for k, v in asked.items()))
    for rank, film in enumerate(films, start=1):
        year = (film["release_date"] or "----")[:4]
        runtime = f"{film['runtime_minutes']} min" if film["runtime_minutes"] else "? min"
        lines.append(
            f"{rank}. {film['title']} ({year}) · {runtime} · score {film['score']:.3f}"
            f" · matched on its {film['source']} text"
        )
        if film.get("evidence"):
            lines.append(f'   "{film["evidence"]}"')
    return "\n".join(lines)


# Everything the agent may do. Adding a capability = adding to this list, nowhere else.
@tool
def lookup_film(title: str) -> str:
    """Look up ONE film the user has named, and return its facts.

    Use this whenever the user mentions a film BY NAME — "tell me about Predator",
    "how long is Titanic?", "is Alien in your catalogue?", "what year was Rocky?".
    A partial name is fine: "Terminator 2" finds "Terminator 2: Judgment Day".

    Do NOT use this when the user describes what they want without naming it
    ("something tense with creatures") — that is `search_films`. The rule is simple:
        did the user say a TITLE?  -> lookup_film
        did the user say a MOOD, SITUATION or PLOT?  -> search_films

    This is an exact database lookup, not a similarity search, so anything it returns
    IS in the catalogue and anything it does not return is NOT. If it comes back empty,
    the film genuinely is not here — say so plainly and do not search for it instead.
    """
    films = get_film(title)
    if not films:
        return (f"'{title}' is not in the catalogue. This was an exact lookup, not a "
                f"guess, so the film really is absent — tell the user plainly. Do not "
                f"call search_films to look for it.")
    lines = []
    for film in films:
        year = (film["release_date"] or "????")[:4]
        lines.append(f"{film['title']} ({year}) · {film['runtime_minutes']} min")
        if film["overview"]:
            lines.append(f"    {film['overview']}")
    return "\n".join(lines)


@tool
def find_films_by_fact(
    director: Optional[str] = None,
    actor: Optional[str] = None,
    genre: Optional[str] = None,
    similar_to: Optional[str] = None,
) -> str:
    """Look films up by FACT — who made them, who is in them, what genre they are, or
    what else is like a named film. Exact database lookup, no scores, no guessing.

    Use this whenever the user names a PERSON, a GENRE, or an existing film to be
    compared against:
        "anything by Christopher Nolan"        -> director="Christopher Nolan"
        "films with Arnold Schwarzenegger"     -> actor="Arnold Schwarzenegger"
        "show me a horror film"                -> genre="Horror"
        "something like Inception"             -> similar_to="Inception"
        "a Nolan action film"                  -> director="Christopher Nolan", genre="Action"

    Every argument you supply is combined with AND, so supplying two narrows the result.
    Supply only what the user actually said.

    WHEN NOT TO USE IT. If the user describes a FEELING, a MOOD or a PLOT rather than
    naming something — "something tense", "a film about a wrongly imprisoned man" — that
    is `search_films`. If the user names ONE film and wants ITS details — "how long is
    Titanic?" — that is `lookup_film`. The rule:
        a description        -> search_films
        one title, its facts -> lookup_film
        a name or a category -> this tool

    genre MUST be one of these exactly, and there are no others in this catalogue:
        Action · Adventure · Animation · Comedy · Crime · Drama · Family
        Fantasy · Horror · Mystery · Romance · Science Fiction · Thriller
    If the user asks for a genre not on that list, say plainly that the catalogue does
    not have it. Do not substitute a near-miss.

    WHAT COMES BACK, and how to read it. Each film carries the QUOTED TEXT of its own
    description. Everything you say about a film must come from ITS quote. Do not describe
    a film from your own knowledge, however sure you are — if the quote does not support
    the sentence you want to write, write a different sentence or say nothing about it.
    There are no relevance scores here because
    there is nothing to be unsure about — a person either directed a film or did not.
    Everything returned satisfies every filter exactly, so state it plainly and never
    hedge. Three distinct empty answers, which must NOT be reported the same way:
        "not in this catalogue"   the person, genre or film is absent entirely. Say so.
        "no film matches all"     each filter exists, but nothing satisfies them together.
                                  Say which combination failed and offer to drop one.
        a list                    these are facts. Do not re-check them, do not apologise
                                  for them, and do not add films the tool did not return.
    Never call `search_films` to double-check this result. This IS the exact answer.
    """
    asked = {k: v for k, v in (("director", director), ("actor", actor),
                               ("genre", genre), ("similar_to", similar_to)) if v}
    if not asked:
        return ("No filter was given. Supply at least one of director, actor, genre or "
                "similar_to — or use search_films if the user described a mood or plot.")

    result = graph_find(**asked)

    if result["unknown"]:
        field, value = next(iter(result["unknown"].items()))
        noun = {"director": "director", "actor": "actor",
                "genre": "genre", "similar_to": "film"}[field]
        return (f"'{value}' is not in this catalogue as a {noun}. This was an exact "
                f"lookup, not a guess, so it genuinely is absent — tell the user plainly. "
                f"Do not call search_films to look for it.")

    if not result["films"]:
        return ("Each of those exists in the catalogue, but no single film satisfies all "
                f"of them at once ({', '.join(f'{k}={v}' for k, v in asked.items())}). "
                "Tell the user which combination came up empty and offer to drop one.")

    lines = ["exact match on: " + ", ".join(f"{k}={v}" for k, v in asked.items())]
    for film in result["films"]:
        year = (film["release_date"] or "----")[:4]
        runtime = f"{film['runtime_minutes']} min" if film["runtime_minutes"] else "? min"
        lines.append(f"{film['title']} ({year}) · {runtime} · {'; '.join(film['why'])[:120]}")
        if film.get("evidence"):
            lines.append(f'   "{film["evidence"]}"')
    return "\n".join(lines)


TOOLS = [search_films, lookup_film, find_films_by_fact]


if __name__ == "__main__":
    # Self-test: no agent, no LLM. Prove the tool works and show what the model will read.
    print("=" * 74)
    print("THE SPEC THE MODEL RECEIVES  (this is all it knows about your code)")
    print("=" * 74)
    for t in TOOLS:
        print(f"\nname:        {t.name}")
        print(f"input schema: {t.args}")
        print("description:")
        for line in t.description.split("\n"):
            print(f"    {line}")

    print("\n" + "=" * 74)
    print("CALLING IT DIRECTLY  (exactly what the agent will do on your behalf)")
    print("=" * 74)
    probes = [
        {"query": "a father and son separated and trying to find each other"},
        {"query": "tense, creatures hunting people", "max_runtime": 120},
        {"query": "anything at all", "max_runtime": 30},          # filter empties the pool
    ]
    for kwargs in probes:
        print(f"\ncall: search_films({kwargs})")
        print(search_films.invoke(kwargs))

    for probe in ["Predator", "Terminator 2", "The Godfather"]:
        print(f"\ncall: lookup_film({{'title': '{probe}'}})")
        print(lookup_film.invoke({"title": probe}))

    for kwargs in ({"director": "Christopher Nolan"},
                   {"actor": "Arnold Schwarzenegger"},
                   {"genre": "Horror"},
                   {"director": "Christopher Nolan", "genre": "Comedy"},
                   {"similar_to": "Inception"},
                   {"director": "Quentin Tarantino"}):
        print(f"\ncall: find_films_by_fact({kwargs})")
        print(find_films_by_fact.invoke(kwargs))

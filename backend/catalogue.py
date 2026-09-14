"""WHAT IS IN THE CATALOGUE — plain facts, read from the database.

THE ONE JOB
    Answer "does this exist, and what is it?" Nothing here ranks, scores, embeds,
    reranks or renders. A function in this file returns DATA — a dict, a list, a
    key — never a sentence meant for a person and never a number meant as a verdict.

WHY IT WAS PULLED OUT
    Three layers were each opening their own connection and writing their own SQL:
    `retrieval.py` (which is allowed to — searching is its job), `tools.py`, and
    `api.py`. The web layer reaching past the domain to the database is the classic
    version of this mistake, and it had already produced the classic consequence:
    "the facts about a film, by title" existed as `_one_film` in tools.py AND as
    `film_facts` in api.py, two queries against the same table, written twice.

    So the direction of dependency is now one way and only one way:

        api.py  ─┐
        tools.py ─┼──▶  catalogue.py  ──▶  the database
        retrieval.py ─┘

    Nothing in here imports anything above it. That is what makes it the bottom.

WHY EVERY FUNCTION TAKING A CONNECTION DOES SO EXPLICITLY
    `resolve_title` and `resolve_node` are called from inside retrieval's own
    transaction — it resolves a name, then runs the search against the state that
    name was resolved in. Opening a second connection here would break that: the
    lookup and the search could see different data. So the caller passes the
    connection it is already holding, and the ones that need no transaction
    (`all_titles`, `facts_for`, `one_film`) open and close their own.
"""

import psycopg

from backend.config import DATABASE_URL

# WHICH STORED TEXT MAY BE SHOWN TO A PERSON.
#
# `theme` is abstract enough to be useless as a reason and `plot_scene` can come from
# the final act, so both are matched on and never printed. This lives HERE, at the
# bottom, because it is a fact about what the movie_data table holds — and because
# two files that each decide for themselves what is safe to show will eventually
# disagree, and the one that gets it wrong prints an ending.
DISPLAYABLE = {"premise", "mood_feel"}

BY_TITLE = """
SELECT movie_id, title,
       tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies
WHERE lower(title) = lower(%(title)s)
"""

LIKE_TITLE = """
SELECT movie_id, title,
       tmdb_raw_payload -> 'belongs_to_collection' ->> 'name' AS series
FROM movies
WHERE title ILIKE %(pattern)s
ORDER BY title
"""

BY_NAME = """
SELECT node_key, name
FROM graph_nodes
WHERE node_type = %(type)s AND lower(name) = lower(%(name)s)
"""

LIKE_NAME = """
SELECT node_key, name
FROM graph_nodes
WHERE node_type = %(type)s AND name ILIKE %(pattern)s
ORDER BY name
"""

ALL_TITLES = "SELECT title FROM movies ORDER BY title"

FACTS = """
SELECT title,
       EXTRACT(YEAR FROM release_date)::int          AS year,
       runtime_minutes,
       tmdb_raw_payload ->> 'poster_path'            AS poster_path
FROM movies
WHERE title = ANY(%(titles)s)
"""

ONE = """
SELECT m.title, EXTRACT(YEAR FROM m.release_date)::int, m.runtime_minutes,
       d.data_kind, d.content
FROM movies m
LEFT JOIN movie_data d ON d.movie_id = m.movie_id AND d.seq = 0
WHERE lower(m.title) = lower(%(title)s)
"""


def resolve_title(conn, title):
    """A written title -> one film. Returns (row, note). Exactly one of them is None.

    Nought matches and several matches are both NORMAL outcomes of a person typing a
    film's name from memory, not errors. Neither is answered by guessing: the note comes
    back so the caller can ask one short question instead.
    """
    row = conn.execute(BY_TITLE, {"title": title}).fetchone()
    if row:
        return row, None

    near = conn.execute(LIKE_TITLE, {"pattern": f"%{title}%"}).fetchall()
    if len(near) == 1:
        return near[0], None
    if not near:
        return None, f"No film called {title!r} is in the catalogue."
    names = ", ".join(r[1] for r in near[:6])
    return None, f"{title!r} matches several films here: {names}. Which one?"


def resolve_node(conn, node_type, name):
    """A written name -> node keys. Returns (keys, note); exactly one is None.

    Exact first, then a contains-match, because people type "Nolan" and the graph holds
    "Christopher Nolan". SEVERAL matches are kept, not narrowed — two actors sharing a
    surname is a real thing, and returning both films is a better answer than silently
    picking one. NO match is reported and stops the search: a filter that matches nothing
    would otherwise delete the entire catalogue and report an empty result as though the
    question had been understood.
    """
    rows = conn.execute(BY_NAME, {"type": node_type, "name": name}).fetchall()
    if not rows:
        rows = conn.execute(LIKE_NAME, {"type": node_type,
                                        "pattern": f"%{name}%"}).fetchall()
    if not rows:
        return None, (f"No {node_type} called {name!r} appears in this catalogue, so "
                      f"nothing was searched. This is a fact, not a weak match — the "
                      f"name is simply not here.")
    return [key for key, _ in rows], None


def all_titles():
    """Every title in the catalogue, alphabetically.

    Used to check a written answer against what actually exists — so it must be the
    whole list, not a page of it.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        return [row[0] for row in conn.execute(ALL_TITLES).fetchall()]


def facts_for(titles):
    """{title: {title, year, runtime_minutes, poster_path}} for the titles given.

    Missing titles are simply absent from the result. A film that is not in the
    catalogue has no facts, and inventing a placeholder for it would put an empty
    card on the page rather than no card.
    """
    if not titles:
        return {}
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(FACTS, {"titles": list(titles)}).fetchall()
    return {row[0]: {"title": row[0], "year": row[1], "runtime_minutes": row[2],
                     "poster_path": row[3]} for row in rows}


def one_film(title):
    """One named film's facts and its showable text. Returns (film, note).

    Exactly one of the two is None, the same contract as `resolve_title` — because
    the reasons it can fail are the same reasons, and a caller should not have to
    learn two shapes for "I could not find that".

    `text` carries only DISPLAYABLE kinds. A theme or a late plot scene is matched on
    and never printed, and enforcing that here rather than at each call site means a
    new caller cannot forget.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        found, problem = resolve_title(conn, title)
        if problem:
            return None, problem
        rows = conn.execute(ONE, {"title": found[1]}).fetchall()

    if not rows:
        return None, f"{title!r} is not in the catalogue."

    name, year, runtime, _, _ = rows[0]
    return {"title": name, "year": year, "runtime_minutes": runtime,
            "text": {kind: content for _, _, _, kind, content in rows
                     if kind in DISPLAYABLE}}, None

"""The exact half: facts, relationships, and where a film can be watched.

No vectors and no model reach this file. A person either directed a film or did not,
and a film is either on a service or is not — questions with a yes/no answer belong to
SQL, never to an embedding.
"""

import re

import psycopg

from backend import providers
from backend.config import DATABASE_URL, EVIDENCE_CHARS, REGION
from backend.tracing import traceable

GRAPH_LIMIT = 10

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

@traceable(run_type="tool", name="graph.genres")
def graph_genres():
    """Every genre in the catalogue, as the model must spell them. Enumerable, so it
    belongs in the tool description rather than being guessed at."""
    with psycopg.connect(DATABASE_URL) as conn:
        return [r[0] for r in conn.execute(
            "SELECT name FROM graph_nodes WHERE node_type = 'genre' ORDER BY name")]

@traceable(run_type="retriever", name="graph.find")
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

_FILM_FOR_AVAILABILITY_SQL = """
SELECT node_key, name,
       properties->>'release_date'    AS release_date,
       properties->>'runtime_minutes' AS runtime_minutes,
       properties->>'poster_path'     AS poster_path
FROM graph_nodes
WHERE node_type = 'film'
  AND (lower(name) = lower(%(exact)s) OR lower(name) LIKE lower(%(loose)s))
ORDER BY (lower(name) = lower(%(exact)s)) DESC, length(name)
LIMIT 1
"""

_AVAILABILITY_SQL = """
SELECT e.edge_type, p.node_key, p.name,
       p.properties->>'logo_path' AS logo_path,
       e.properties->>'link'      AS link
FROM graph_edges e
JOIN graph_nodes p ON p.node_key = e.to_key
WHERE e.from_key  = %(film_key)s
  AND e.source    = %(source)s
  AND e.edge_type = ANY(%(types)s)
"""

@traceable(run_type="retriever", name="graph.availability")
def availability(title):
    """Where one film can be watched, priced and grouped, cheapest band first.

    Returns:
        {"found": bool,          # is the film in this catalogue at all
         "has_listing": bool,    # do we hold ANY offer for this country
         "title", "release_date", "runtime_minutes", "poster_path",
         "region", "checked_on", "stale_days", "link",
         "offers": [ ... ]}      # already sorted; each carries price_text

    `found` and `has_listing` are deliberately separate. "I don't know" and
    "it isn't available" are different sentences, and a caller that cannot tell
    them apart will confidently say the wrong one.
    """
    empty = {"found": False, "has_listing": False, "title": title,
             "release_date": None, "runtime_minutes": None, "poster_path": None,
             "region": providers.REGION,
             "checked_on": providers.PRICES_CHECKED_ON.isoformat(),
             "stale_days": providers.staleness_days(),
             "link": None, "offers": []}

    with psycopg.connect(DATABASE_URL) as conn:
        film = conn.execute(_FILM_FOR_AVAILABILITY_SQL, {
            "exact": title, "loose": f"%{title}%"}).fetchone()
        if not film:
            return empty                       # not in the catalogue

        film_key, name, release_date, runtime, poster = film
        rows = conn.execute(_AVAILABILITY_SQL, {
            "film_key": film_key,
            "source": providers.SOURCE,
            "types": list(providers.BAND_FROM_EDGE)}).fetchall()

    offers, link = [], None
    for edge_type, provider_key, provider_name, logo_path, deep_link in rows:
        slug = provider_key.split(":", 1)[1]
        offer = providers.describe(slug, provider_name, edge_type)
        offer["logo_path"] = logo_path
        offers.append(offer)
        link = link or deep_link

    offers.sort(key=providers.sort_key)

    return {**empty,
            "found": True,
            "has_listing": bool(offers),
            "title": name,
            "release_date": release_date,
            "runtime_minutes": int(runtime) if runtime else None,
            "poster_path": poster,
            "link": link,
            "offers": offers}

@traceable(run_type="tool", name="graph.film_titles")
def graph_film_titles():
    """Every film title in the catalogue, longest first.

    Longest first matters: "Terminator 2: Judgment Day" must be tested before
    "Terminator 2", or the shorter name matches and the longer one never gets a turn.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT name FROM graph_nodes WHERE node_type = 'film'").fetchall()
    return sorted((r[0] for r in rows), key=len, reverse=True)


if __name__ == "__main__":
    # Self-test: exact facts, no model involved anywhere in this file.
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
        if not result["films"] and not result["unknown"]:
            print("   (no film satisfies all of those at once)")

    print("\n" + "=" * 74)
    print("AVAILABILITY — the hardest constraint, straight from the graph")
    print("=" * 74)
    for probe in ("Predator", "Alien", "Terminator 2", "The Seventh Seal"):
        found = availability(probe)
        if not found["found"]:
            print(f"\n{probe}: not in this catalogue"); continue
        if not found["has_listing"]:
            print(f"\n{found['title']}: here, but no {found['region']} listing held"); continue
        print(f"\n{found['title']} — {len(found['offers'])} ways to watch in "
              f"{found['region']}  (checked {found['checked_on']}, "
              f"{found['stale_days']}d ago)")
        band = None
        for offer in found["offers"][:7]:
            if offer["band"] != band:
                band = offer["band"]
                print(f"   {providers.BAND_LABEL[band]}")
            flag = "" if offer["verified"] else "   ?"
            print(f"      {offer['display']:<26} {offer['price_text']:<22}{flag}")
        if len(found["offers"]) > 7:
            print(f"      … and {len(found['offers']) - 7} more")

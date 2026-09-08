"""Derive the knowledge graph from the TMDB payloads already in the database.

WHAT A GRAPH IS FOR, AND WHAT IT IS NOT FOR
    "Anything by Nolan" is a yes/no question about an edge. It has no degrees, so it must
    never touch a vector — a vector would return films that FEEL like Nolan films, which
    is a different question answered confidently and wrongly.

    So the graph is a FILTER, never a ranker, and it returns no scores. A person either
    directed a film or did not, and attaching a confidence to a fact only invites the
    caller to doubt it.

DERIVED FROM movies.tmdb_raw_payload, NEVER FROM data/raw/
    Reading the files would let the graph describe films that are not in `movies`.
    Reading the stored payload makes that drift impossible. This is the same reason the
    payload is kept in the first place: keep the raw thing, derive everything else.

    Which also means: dropping both tables and re-running this is always safe, and is
    the normal way to change the graph's shape.

WHAT IS KEPT, AND WHY SO LITTLE
    Top 10 cast, and crew filtered to directors. One film in this catalogue lists 148
    crew. Noise in a graph is not free — it costs you at every traversal, forever, and a
    hundred camera assistants make "who worked on both of these" meaningless.

HOW A NODE IS NAMED
    `film:27205`, `person:525`, `genre:878` — built from what the thing IS, so an edge
    can be written without looking anything up first, and so a rebuild produces byte-
    identical keys. Films are keyed by TMDB id, which is the natural key; joining to
    `movies` is `'film:' || movies.tmdb_id`.

RUN
    python -m pipeline.build_graph            build or refresh
    python -m pipeline.build_graph --status   counts by type, writes nothing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg                                                     # noqa: E402
from dotenv import load_dotenv                                     # noqa: E402

from backend.config import DATABASE_URL                            # noqa: E402

load_dotenv()

TOP_CAST = 10
DIRECTING_JOBS = ("Director",)

PAYLOADS = """
SELECT movie_id, title, tmdb_id, tmdb_raw_payload
FROM movies
ORDER BY movie_id
"""

NODE = """
INSERT INTO graph_nodes (node_key, node_type, name, properties)
VALUES (%(key)s, %(type)s, %(name)s, %(properties)s)
ON CONFLICT (node_key) DO UPDATE
    SET name = EXCLUDED.name, properties = EXCLUDED.properties
"""

EDGE = """
INSERT INTO graph_edges (from_key, to_key, edge_type)
VALUES (%(from_key)s, %(to_key)s, %(edge_type)s)
ON CONFLICT (from_key, to_key, edge_type, source) DO NOTHING
"""

STATUS_NODES = """
SELECT node_type, count(*) FROM graph_nodes GROUP BY node_type ORDER BY node_type
"""

STATUS_EDGES = """
SELECT edge_type, count(*) FROM graph_edges GROUP BY edge_type ORDER BY edge_type
"""

SHARED = """
SELECT n.node_type, n.name, count(DISTINCT e.from_key) AS films
FROM graph_edges e
JOIN graph_nodes n ON n.node_key = e.to_key
GROUP BY n.node_type, n.name
HAVING count(DISTINCT e.from_key) > 1
ORDER BY films DESC, n.name
LIMIT 12
"""


def facts(payload):
    """One payload -> the nodes and edges it justifies. Pure: no database, no network.

    Separated from the writing so the extraction can be read, and reasoned about, without
    a connection open. It is also the only place that knows TMDB's field names.
    """
    nodes, edges = [], []
    film = f"film:{payload['id']}"
    nodes.append((film, "film", payload.get("title") or "?", {}))

    for genre in payload.get("genres") or []:
        key = f"genre:{genre['id']}"
        nodes.append((key, "genre", genre["name"], {}))
        edges.append((film, key, "HAS_GENRE"))

    for keyword in (payload.get("keywords") or {}).get("keywords") or []:
        key = f"keyword:{keyword['id']}"
        nodes.append((key, "keyword", keyword["name"], {}))
        edges.append((film, key, "HAS_KEYWORD"))

    credits = payload.get("credits") or {}

    # Top billing only. TMDB orders cast by `order`, which is the billing order, so the
    # first ten are the people someone would actually name when describing the film.
    cast = sorted(credits.get("cast") or [], key=lambda p: p.get("order", 999))
    for person in cast[:TOP_CAST]:
        key = f"person:{person['id']}"
        nodes.append((key, "person", person["name"], {}))
        edges.append((film, key, "ACTED_IN"))

    for person in credits.get("crew") or []:
        if person.get("job") not in DIRECTING_JOBS:
            continue
        key = f"person:{person['id']}"
        nodes.append((key, "person", person["name"], {}))
        edges.append((film, key, "DIRECTED"))

    return nodes, edges


def status(conn):
    print("nodes")
    for node_type, count in conn.execute(STATUS_NODES).fetchall():
        print(f"  {node_type:10} {count:>5}")
    print("edges")
    for edge_type, count in conn.execute(STATUS_EDGES).fetchall():
        print(f"  {edge_type:14} {count:>5}")

    # THE ONLY NUMBER THAT SAYS WHETHER THE GRAPH IS WORTH HAVING.
    # A graph's value is entirely in what is SHARED — a person, genre or keyword
    # attached to exactly one film connects nothing and can answer no question.
    shared = conn.execute(SHARED).fetchall()
    print(f"\nthings connecting more than one film — this is where a graph earns its keep")
    if not shared:
        print("  none. Every node touches one film, so the graph connects nothing yet.")
    for node_type, name, films in shared:
        print(f"  {films:>2} films  {node_type:8} {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true",
                        help="report what is in the graph and write nothing")
    args = parser.parse_args()

    with psycopg.connect(DATABASE_URL) as conn:
        if args.status:
            status(conn)
            return

        films = conn.execute(PAYLOADS).fetchall()
        print(f"{len(films)} films\n")

        seen_nodes = set()
        written_edges = 0

        for movie_id, title, tmdb_id, payload in films:
            nodes, edges = facts(payload)

            for key, node_type, name, properties in nodes:
                if key in seen_nodes:
                    continue                       # already written this run
                conn.execute(NODE, {"key": key, "type": node_type, "name": name,
                                    "properties": psycopg.types.json.Json(properties)})
                seen_nodes.add(key)

            for from_key, to_key, edge_type in edges:
                conn.execute(EDGE, {"from_key": from_key, "to_key": to_key,
                                    "edge_type": edge_type})
                written_edges += 1

            print(f"  {movie_id:>3}  {title:34} {len(nodes):>3} nodes · "
                  f"{len(edges):>3} edges")

        conn.commit()
        print(f"\n{len(seen_nodes)} distinct nodes · {written_edges} edges offered\n")
        status(conn)


if __name__ == "__main__":
    main()

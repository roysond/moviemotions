"""
build_graph.py — derive a knowledge graph from the film payloads already in Postgres.

The graph is DERIVED data, exactly like embeddings. The source of truth is
movies.raw_payload; this script only reshapes it. Deleting the whole graph and
rebuilding it must always be safe, and running this twice must change nothing.

    python build_graph.py              build (idempotent)
    python build_graph.py --status     what is in there now
    python build_graph.py --remove     delete every node and edge

Nodes    film · person · genre · keyword
Edges    person -ACTED_IN->  film
         person -DIRECTED->  film
         film   -HAS_GENRE-> genre
         film   -HAS_KEYWORD-> keyword
"""

import os
import re
import sys

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Json

load_dotenv()

# How many billed actors per film become ACTED_IN edges. TMDB lists everyone
# down to "Man in Diner"; past the top ten they are noise, not signal.
CAST_TOP_N = 10

SOURCE = "tmdb"          # every edge records where the claim came from
CONFIDENCE = 1.0         # TMDB is a primary source, so we assert it plainly


# ---------------------------------------------------------------- helpers

def slug(text: str) -> str:
    """'Science Fiction' -> 'science-fiction'. Node keys stay readable."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


class Graph:
    """Collects nodes and edges in memory, de-duplicating as it goes."""

    def __init__(self):
        self.nodes = {}          # node_key -> (node_type, name, properties)
        self.edges = {}          # (from, to, type) -> properties
        self.slug_owner = {}     # slug -> original name, to catch collisions

    def node(self, node_type, key_part, name, **properties):
        key = f"{node_type}:{key_part}"
        # A genre and a keyword may share a name; they are still different nodes
        # because the type is part of the key.
        if node_type in ("genre", "keyword"):
            seen = self.slug_owner.get(key)
            if seen and seen != name:
                print(f"  ! slug collision on {key}: '{seen}' vs '{name}'")
            self.slug_owner[key] = name
        self.nodes.setdefault(key, (node_type, name, properties))
        return key

    def edge(self, from_key, to_key, edge_type, **properties):
        # Same person credited twice in one film collapses to one edge, which
        # is what the UNIQUE constraint in the table would do anyway.
        self.edges.setdefault((from_key, to_key, edge_type), properties)


# ---------------------------------------------------------------- extract

def build_from_payloads(rows) -> Graph:
    g = Graph()

    for movie_id, source_id, payload in rows:
        film_key = g.node(
            "film", source_id, payload["title"],
            movie_id=movie_id,                 # the join back to the relational side
            tmdb_id=payload["id"],
            release_date=payload.get("release_date"),
            runtime_minutes=payload.get("runtime"),
        )

        for genre in payload.get("genres", []):
            gk = g.node("genre", slug(genre["name"]), genre["name"],
                        tmdb_id=genre["id"])
            g.edge(film_key, gk, "HAS_GENRE")

        for kw in payload.get("keywords", {}).get("keywords", []):
            kk = g.node("keyword", slug(kw["name"]), kw["name"], tmdb_id=kw["id"])
            g.edge(film_key, kk, "HAS_KEYWORD")

        credits = payload.get("credits", {})

        for member in credits.get("cast", [])[:CAST_TOP_N]:
            pk = g.node("person", member["id"], member["name"],
                        known_for=member.get("known_for_department"))
            g.edge(pk, film_key, "ACTED_IN",
                   character=member.get("character"),
                   billing=member.get("order"))

        for member in credits.get("crew", []):
            if member.get("job") != "Director":
                continue
            pk = g.node("person", member["id"], member["name"],
                        known_for=member.get("known_for_department"))
            g.edge(pk, film_key, "DIRECTED")

    return g


# ---------------------------------------------------------------- load

INSERT_NODE = """
INSERT INTO graph_nodes (node_key, node_type, name, properties)
VALUES (%s, %s, %s, %s)
ON CONFLICT (node_key) DO NOTHING
"""

INSERT_EDGE = """
INSERT INTO graph_edges (from_key, to_key, edge_type, properties, source, confidence)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (from_key, to_key, edge_type, source) DO NOTHING
"""


def report(conn, heading):
    print(f"\n{heading}")
    print("  nodes")
    for row in conn.execute(
        "SELECT node_type, count(*) FROM graph_nodes GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"    {row[0]:<10} {row[1]:>5}")
    print("  edges")
    for row in conn.execute(
        "SELECT edge_type, count(*) FROM graph_edges GROUP BY 1 ORDER BY 2 DESC"
    ):
        print(f"    {row[0]:<12} {row[1]:>5}")
    totals = conn.execute(
        "SELECT (SELECT count(*) FROM graph_nodes), (SELECT count(*) FROM graph_edges)"
    ).fetchone()
    print(f"  TOTAL       {totals[0]} nodes · {totals[1]} edges")


def main():
    flag = sys.argv[1] if len(sys.argv) > 1 else ""

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:

        if flag == "--status":
            report(conn, "current graph")
            return

        if flag == "--remove":
            # Edges first: they hold the foreign keys.
            edges = conn.execute("DELETE FROM graph_edges").rowcount
            nodes = conn.execute("DELETE FROM graph_nodes").rowcount
            conn.commit()
            print(f"removed {nodes} nodes and {edges} edges")
            return

        rows = conn.execute(
            "SELECT movie_id, source_id, raw_payload FROM movies "
            "WHERE source = 'tmdb' ORDER BY movie_id"
        ).fetchall()
        print(f"read {len(rows)} film payloads from movies.raw_payload")

        g = build_from_payloads(rows)
        print(f"derived {len(g.nodes)} nodes and {len(g.edges)} edges in memory")

        conn.cursor().executemany(INSERT_NODE, [
            (key, ntype, name, Json(props))
            for key, (ntype, name, props) in g.nodes.items()
        ])
        conn.cursor().executemany(INSERT_EDGE, [
            (frm, to, etype, Json(props), SOURCE, CONFIDENCE)
            for (frm, to, etype), props in g.edges.items()
        ])
        conn.commit()

        report(conn, "after load")


if __name__ == "__main__":
    main()

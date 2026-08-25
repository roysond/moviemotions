"""
graph_vs_vector.py — the same question asked two ways, side by side.

WHY THIS EXISTS
    "A knowledge graph answers things a vector search cannot" is a claim. This turns it
    into evidence you can look at. Three factual questions, each sent to both machines.

    The graph queries below are HARD-CODED, and that is the point. Turning English into
    a graph query is the language model's job, and that is exactly what the next step
    builds. Here we skip the model so nothing fuzzy can be blamed for the result.

    Read the two columns and ask one question: which one is a FACT, and which one is a
    GUESS that happens to have a number next to it?

    python experiments/graph_vs_vector.py
"""

import os
import sys

import psycopg
from dotenv import load_dotenv

# experiments/ is a subfolder, so the project root is not on the import path by
# default. Same line every script in here carries.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import search

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

# Each case: the English question, the graph query, and its parameters.
CASES = [
    (
        "films directed by Christopher Nolan",
        """SELECT f.name
           FROM graph_edges e
           JOIN graph_nodes p ON p.node_key = e.from_key
           JOIN graph_nodes f ON f.node_key = e.to_key
           WHERE e.edge_type = 'DIRECTED' AND p.name = %(name)s
           ORDER BY f.name""",
        {"name": "Christopher Nolan"},
    ),
    (
        "horror films",
        """SELECT f.name
           FROM graph_edges e
           JOIN graph_nodes f ON f.node_key = e.from_key
           WHERE e.edge_type = 'HAS_GENRE' AND e.to_key = %(genre)s
           ORDER BY f.name""",
        {"genre": "genre:horror"},
    ),
    (
        "films starring Arnold Schwarzenegger",
        """SELECT f.name
           FROM graph_edges e
           JOIN graph_nodes p ON p.node_key = e.from_key
           JOIN graph_nodes f ON f.node_key = e.to_key
           WHERE e.edge_type = 'ACTED_IN' AND p.name = %(name)s
           ORDER BY f.name""",
        {"name": "Arnold Schwarzenegger"},
    ),
]

WIDTH = 78


def ask_graph(conn, sql, params):
    return [row[0] for row in conn.execute(sql, params)]


def ask_vectors(question):
    films = search(question, limit=3)
    return [(f["title"], f["score"]) for f in films]


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        for question, sql, params in CASES:
            print("\n" + "=" * WIDTH)
            print(f"  {question}")
            print("=" * WIDTH)

            graph_answer = ask_graph(conn, sql, params)
            vector_answer = ask_vectors(question)

            print("\n  GRAPH — exact, no scores, same answer every run")
            if graph_answer:
                for title in graph_answer:
                    print(f"      • {title}")
            else:
                print("      (nothing — and 'nothing' is a real answer here)")

            print("\n  VECTOR SEARCH — ranked by similarity of meaning")
            for title, score in vector_answer:
                print(f"      {score:.3f}  {title}")

            hit = {t for t in graph_answer}
            top = vector_answer[0][0] if vector_answer else None
            verdict = "agrees" if top in hit else "DISAGREES"
            print(f"\n  top vector result vs graph truth: {verdict}")

    print("\n" + "-" * WIDTH)
    print("  The graph column has no scores because there is nothing to be unsure about.")
    print("  The vector column always returns three films — including when the honest")
    print("  answer is 'that is not a question I can answer'.")
    print("-" * WIDTH)


if __name__ == "__main__":
    main()

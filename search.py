"""Search the corpus from the command line."""

import sys

from core import search

query = " ".join(sys.argv[1:]) or "creatures chasing you, very intense"

print(f'\nQUERY: "{query}"\n')
for position, film in enumerate(search(query, limit=5), start=1):
    print(f"{position}. {film['score']:.4f}  [{film['method']:>6}]  {film['title']}")
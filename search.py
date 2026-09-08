"""Search the corpus from the command line. No agent, no model choosing anything.

WHY THIS EXISTS SEPARATELY FROM THE AGENT
    When an answer is wrong there are two suspects: retrieval found the wrong films, or
    the model wrote badly about the right ones. This tells them apart in one command,
    because it shows exactly what retrieval returned before any model touched it.

RUN
    python search.py warm and cosy
    python search.py --situation a group trapped with no way out
    python search.py --like "Toy Story"
"""

import sys

from backend.retrieval import search

args = sys.argv[1:]
kind = "mood"
if args and args[0].startswith("--"):
    kind, args = args[0][2:], args[1:]
query = " ".join(args) or "warm and comforting, cosy"

field = {"mood": "mood", "situation": "situation", "theme": "theme",
         "like": "similar_to"}.get(kind)
if field is None:
    sys.exit(f"--{kind} is not a search field. Use mood, situation, theme or like.")

results, notes = search(**{field: query}, limit=5)

print(f'\n{field}: "{query}"\n')
for position, film in enumerate(results, start=1):
    print(f"{position}. rank {film['score']:.2f} · {film['over_floor']:+.3f} over its "
          f"band   {film['title']} ({film['year']})")
    for label, hit in film["per_constraint"].items():
        print(f"      {label}: raw {hit['raw']:.3f}")
if not results:
    print("  (nothing)")
print()
for note in notes:
    print(f"  · {note}")

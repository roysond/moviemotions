"""Move the fetched plots off disk and into movies.wikidata_plot.

WHY THIS IS A SEPARATE SCRIPT FROM fetch_plots.py
    Fetching talks to two APIs over the network and can half-fail. Loading reads a file
    that is already on disk and cannot. Keeping them apart means a failed load never
    costs an API call, and re-running the load is free.

MATCHED ON tmdb_id, NEVER ON TITLE
    plots.json carries the TMDB id precisely so the plot can be returned to the right
    film. Matching on title would give the 2010 Karate Kid the 1984 one's plot.

ON THE COLUMN NAME
    The text is Wikipedia's; Wikidata is only the directory used to find the article.
    The column is called wikidata_plot by choice — recorded here so the next reader is
    not misled into thinking Wikidata serves prose.

SAFE TO RE-RUN
    UPDATE, not INSERT. Running twice writes the same text twice and changes nothing.

RUN
    python -m pipeline.load_plots
"""

import json
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

UPDATE = """
UPDATE movies SET wikidata_plot = %(plot)s
WHERE tmdb_id = %(tmdb_id)s
RETURNING movie_id, title
"""


def main():
    plots = json.load(open("data/plots.json"))

    loaded = empty = unmatched = 0

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        for entry in plots:
            title = entry.get("title", "?")
            plot = entry.get("plot")

            # A film with no plot is a normal outcome, not an error — some articles have
            # no Plot section. Say so and move on rather than writing an empty string,
            # which would be indistinguishable from "we have not tried yet".
            if not plot:
                empty += 1
                print(f"  --   {title}  (no plot fetched)")
                continue

            row = conn.execute(UPDATE, {
                "plot": plot,
                "tmdb_id": str(entry["tmdb_id"]),
            }).fetchone()

            if row is None:
                unmatched += 1
                print(f"  !!   {title}  (tmdb_id {entry['tmdb_id']} not in movies)")
            else:
                loaded += 1
                print(f"  {row[0]:>3}  {row[1]:34} {len(plot):>6} chars")

        conn.commit()

    print(f"\nloaded {loaded} · no plot {empty} · unmatched {unmatched}")
    if unmatched:
        print("Unmatched means plots.json holds a film movies does not. Check before rerunning.")


if __name__ == "__main__":
    main()

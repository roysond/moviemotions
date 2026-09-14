"""Load only the hand-picked test corpus from disk into Postgres."""

import json
import os

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Json

load_dotenv()

# Column names carry their source. `source`/`source_id` were dropped on 5 Sep: with
# TMDB the only catalogue, `source` was a constant, and `source_id` said less than
# `tmdb_id` does. The plot arrives from Wikipedia and lands in wikidata_plot, so a
# column named plainly `overview` would no longer say whose overview it is.
#
# ON CONFLICT makes this script safe to run twice: the second run skips rows already
# present instead of failing on the unique key.
INSERT = """
INSERT INTO movies (tmdb_id, title, release_date,
                    runtime_minutes, tmdb_overview, tmdb_raw_payload)
VALUES (%(tmdb_id)s, %(title)s, %(release_date)s,
        %(runtime_minutes)s, %(tmdb_overview)s, %(tmdb_raw_payload)s)
ON CONFLICT (tmdb_id) DO NOTHING
RETURNING movie_id
"""

with open("data/test_corpus_ids.json") as handle:
    wanted_ids = json.load(handle)

inserted = skipped = 0

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    for tmdb_id in wanted_ids:
        with open(f"data/raw/tmdb_{tmdb_id}.json") as handle:
            film = json.load(handle)

        row = conn.execute(
            INSERT,
            {
                "tmdb_id": str(film["id"]),
                "title": film["title"],
                "release_date": film["release_date"] or None,
                "runtime_minutes": film.get("runtime"),
                "tmdb_overview": film.get("overview"),
                "tmdb_raw_payload": Json(film),
            },
        ).fetchone()

        if row:
            inserted += 1
            print(f"{row[0]:>3}  {film['title']}")
        else:
            skipped += 1
            print(f"  -  {film['title']}  (already present)")

    conn.commit()

print(f"\ninserted {inserted}, skipped {skipped}")

"""Load only the hand-picked test corpus from disk into Postgres."""

import json
import os

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Json

load_dotenv()

INSERT = """
INSERT INTO movies (source, source_id, title, release_date,
                    runtime_minutes, overview, raw_payload)
VALUES (%(source)s, %(source_id)s, %(title)s, %(release_date)s,
        %(runtime_minutes)s, %(overview)s, %(raw_payload)s)
ON CONFLICT (source, source_id) DO NOTHING
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
                "source": "tmdb",
                "source_id": str(film["id"]),
                "title": film["title"],
                "release_date": film["release_date"] or None,
                "runtime_minutes": film.get("runtime"),
                "overview": film.get("overview"),
                "raw_payload": Json(film),
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
"""Move the derived text off disk and into movie_data.

WHY THIS IS A SEPARATE SCRIPT FROM derive_corpus.py
    Deriving talks to a model, costs money, and can half-fail. Loading reads a file that
    is already on disk and cannot. Keeping them apart means a failed load never costs a
    model call, and re-running the load is free. Same reason fetch and load are separate.

WHAT IT WRITES
    Three rows per film, all at seq 0, because mood_feel, theme and premise are
    one-per-film. Plot scenes will arrive later as seq 1..n under the same movie_id.

MATCHED ON movie_id, VERIFIED BY TITLE
    derived.json carries both. The id is what the row is written against; the title is
    checked against the database and a mismatch STOPS the run. If movies were ever
    reloaded, an id could point at a different film, and the failure would be silent —
    every row lands, every row is wrong.

CHANGED TEXT DROPS ITS VECTORS
    A vector is the numbers for a specific sentence. Change the sentence and the numbers
    describe text that no longer exists. Nothing errors: search simply returns films for
    reasons you cannot read any more. So an UPDATE here deletes that row's vectors, and
    the embedder rebuilds them. A missing vector is visible. A stale one is not.

SAFE TO RE-RUN
    Unchanged text is left alone and counted as unchanged. Re-running after a prompt
    change updates only what the model actually rewrote.

RUN
    python -m pipeline.load_derived                       data/derived.json
    python -m pipeline.load_derived --file data/derived.sample.json
"""

import argparse
import json
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

FIELDS = ("mood_feel", "theme", "premise")
DEFAULT_FILE = "data/derived.json"

TITLES = "SELECT movie_id, title FROM movies"

EXISTING = """
SELECT data_id, content
FROM movie_data
WHERE movie_id = %(movie_id)s AND data_kind = %(data_kind)s AND seq = %(seq)s
"""

INSERT = """
INSERT INTO movie_data (movie_id, data_kind, seq, content)
VALUES (%(movie_id)s, %(data_kind)s, %(seq)s, %(content)s)
RETURNING data_id
"""

UPDATE = "UPDATE movie_data SET content = %(content)s WHERE data_id = %(data_id)s"

DROP_VECTORS = "DELETE FROM movie_vectors WHERE data_id = %(data_id)s"


def load(conn, movie_id, data_kind, content, seq=0):
    """Write one row of movie_data. Returns 'inserted', 'updated' or 'unchanged'.

    SHARED WITH THE PLOT CHUNKER, which is why `seq` is an argument. Writing a row and
    dropping its stale vectors is one rule, and a rule implemented twice is a rule that
    will eventually be two different rules — the one that drifts being the one nobody
    reads. Derived text is seq 0; plot scenes are seq 1..n; the rule is identical.

    Three plain statements rather than one clever ON CONFLICT ... RETURNING. The clever
    version needs a Postgres internal column to tell an insert from an update, and a line
    of code you cannot read is a line of code you cannot check.
    """
    row = conn.execute(EXISTING, {"movie_id": movie_id, "data_kind": data_kind,
                                  "seq": seq}).fetchone()

    if row is None:
        conn.execute(INSERT, {"movie_id": movie_id, "data_kind": data_kind,
                              "content": content, "seq": seq})
        return "inserted"

    data_id, stored = row
    if stored == content:
        return "unchanged"

    conn.execute(UPDATE, {"content": content, "data_id": data_id})
    conn.execute(DROP_VECTORS, {"data_id": data_id})
    return "updated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help=f"which derived file to load (default {DEFAULT_FILE})")
    args = parser.parse_args()

    derived = json.load(open(args.file))
    print(f"{len(derived)} films from {args.file}\n")

    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    blank = wrong_title = 0

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        titles = dict(conn.execute(TITLES).fetchall())

        for entry in derived:
            movie_id = entry["movie_id"]
            title = entry.get("title", "?")

            # An id that is not in movies, or is in movies under a different name, means
            # the file and the table disagree about which film this is. Stop — do not
            # write a guess.
            if titles.get(movie_id) != title:
                wrong_title += 1
                print(f"  !!   {title}  (movie_id {movie_id} is "
                      f"{titles.get(movie_id, 'not in movies')!r})")
                continue

            results = []
            for field in FIELDS:
                content = (entry.get(field) or "").strip()
                if not content:
                    blank += 1
                    print(f"  --   {title:34} {field} is empty — not written")
                    continue
                outcome = load(conn, movie_id, field, content)
                counts[outcome] += 1
                results.append(f"{field}:{outcome}")

            print(f"  {movie_id:>3}  {title:34} {' · '.join(results)}")

        if wrong_title:
            print("\nSTOPPING — the file and the table disagree about a film. "
                  "Nothing has been written.")
            conn.rollback()
            return

        conn.commit()

    print(f"\ninserted {counts['inserted']} · updated {counts['updated']} · "
          f"unchanged {counts['unchanged']} · blank {blank}")
    if counts["updated"]:
        print(f"{counts['updated']} rows changed — their vectors were deleted and "
              "must be rebuilt before searching.")


if __name__ == "__main__":
    main()

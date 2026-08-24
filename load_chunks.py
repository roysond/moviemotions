"""Write the chosen model's vectors into chunks + chunk_embeddings."""

import json
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

MODEL_KEY = "nova"
MODEL_ID = os.environ["BEDROCK_MODEL_EMBED_NOVA"]

INSERT_CHUNK = """
INSERT INTO chunks (movie_id, chunk_index, source_field, content)
VALUES (%(movie_id)s, 0, 'overview', %(content)s)
ON CONFLICT (movie_id, chunk_index) DO UPDATE SET content = EXCLUDED.content
RETURNING chunk_id
"""

INSERT_EMBEDDING = """
INSERT INTO chunk_embeddings (chunk_id, embedding, model_id, dimensions, embed_variant)
VALUES (%(chunk_id)s, %(embedding)s::vector, %(model_id)s, %(dimensions)s, 'clean')
ON CONFLICT (chunk_id, model_id, embed_variant) DO NOTHING
"""

data = json.load(open("data/embeddings.json"))
dimensions = data["dimensions"]

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    overviews = dict(conn.execute("SELECT movie_id, overview FROM movies").fetchall())

    for film in data["films"]:
        chunk_id = conn.execute(
            INSERT_CHUNK,
            {"movie_id": film["id"], "content": overviews[film["id"]]},
        ).fetchone()[0]

        conn.execute(
            INSERT_EMBEDDING,
            {
                "chunk_id": chunk_id,
                "embedding": str(film["vectors"][MODEL_KEY]),
                "model_id": MODEL_ID,
                "dimensions": dimensions,
            },
        )

    conn.commit()

    for table in ("chunks", "chunk_embeddings"):
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")
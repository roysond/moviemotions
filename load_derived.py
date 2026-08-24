"""Store derived moods and themes as a second chunk per film, embedded."""

import json

import psycopg

from core import DATABASE_URL, DIMENSIONS, MODEL_ID, embed

INSERT_CHUNK = """
INSERT INTO chunks (movie_id, chunk_index, source_field, content)
VALUES (%(movie_id)s, 1, 'derived', %(content)s)
ON CONFLICT (movie_id, chunk_index) DO UPDATE SET content = EXCLUDED.content
RETURNING chunk_id
"""

INSERT_EMBEDDING = """
INSERT INTO chunk_embeddings (chunk_id, embedding, model_id, dimensions, embed_variant)
VALUES (%(chunk_id)s, %(embedding)s::vector, %(model_id)s, %(dimensions)s, 'clean')
ON CONFLICT (chunk_id, model_id, embed_variant) DO UPDATE SET embedding = EXCLUDED.embedding
"""

derived = json.load(open("data/derived.json"))

with psycopg.connect(DATABASE_URL) as conn:
    for film in derived:
        content = (
            f"{film.get('feel', '')}. "
            f"Moods: {', '.join(film.get('moods', []))}. "
            f"Themes: {', '.join(film.get('themes', []))}."
        )

        chunk_id = conn.execute(
            INSERT_CHUNK, {"movie_id": film["id"], "content": content}
        ).fetchone()[0]

        conn.execute(
            INSERT_EMBEDDING,
            {
                "chunk_id": chunk_id,
                "embedding": str(embed(content)),
                "model_id": MODEL_ID,
                "dimensions": DIMENSIONS,
            },
        )
        print(film["title"])

    conn.commit()

    for table in ("chunks", "chunk_embeddings"):
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")
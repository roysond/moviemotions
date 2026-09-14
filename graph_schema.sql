-- Knowledge graph: nodes are things, edges are facts connecting two things.
--
-- STATUS 13 Sep 2026 — THESE TABLES EXIST AND ARE POPULATED.
-- 566 nodes, 634 edges. This file now describes what is in the database rather
-- than what was once planned for it; the previous version still carried a banner
-- saying the tables did not exist, three days after they were built.
--
-- Additive only. Nothing here touches movies, movie_data or movie_vectors. The graph
-- is DERIVED entirely from movies.tmdb_raw_payload, so dropping both tables and
-- re-running this file plus pipeline/build_graph.py is always safe, and is the normal
-- way to change its shape.

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_key    TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL
                CHECK (node_type IN ('film', 'person', 'genre', 'keyword')),
    name        TEXT NOT NULL,
    properties  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- WHY 'provider' IS NOT IN THAT LIST ANY MORE
-- The five AVAILABLE_* edge types below it described a streaming-availability layer
-- that was removed in the RAG rebuild and has not come back. A CHECK constraint that
-- permits values nothing writes is not harmless documentation: it is the schema
-- claiming a capability the application does not have.

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id     BIGSERIAL PRIMARY KEY,
    from_key    TEXT NOT NULL REFERENCES graph_nodes(node_key) ON DELETE CASCADE,
    to_key      TEXT NOT NULL REFERENCES graph_nodes(node_key) ON DELETE CASCADE,
    edge_type   TEXT NOT NULL
                CHECK (edge_type IN ('ACTED_IN', 'DIRECTED',
                                     'HAS_GENRE', 'HAS_KEYWORD')),
    properties  JSONB NOT NULL DEFAULT '{}'::jsonb,
    source      TEXT NOT NULL DEFAULT 'tmdb',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- source is part of the key so the same fact from two providers is two rows,
    -- which is what lets one be removed without taking the other with it.
    UNIQUE (from_key, to_key, edge_type, source)
);

-- WHY THERE IS NO `confidence` COLUMN
-- There was one, defaulting to 1.0. Every edge here comes from a structured TMDB
-- payload: a person directed a film or did not. A column that must never vary from
-- 1.0 is not a measurement, it is an invitation to start guessing later and to have
-- somewhere to put the guess.

-- Traversal runs in BOTH directions, so it needs an index at both ends.
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges (from_key, edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges (to_key,   edge_type);


-- Knowledge graph: nodes are things, edges are facts connecting two things.
-- Additive only. Nothing here touches movies, chunks or chunk_embeddings.

CREATE TABLE IF NOT EXISTS graph_nodes (
    node_key    TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL
                CHECK (node_type IN ('film','person','genre','keyword')),
    name        TEXT NOT NULL,
    properties  JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS graph_edges (
    edge_id     BIGSERIAL PRIMARY KEY,
    from_key    TEXT NOT NULL REFERENCES graph_nodes(node_key) ON DELETE CASCADE,
    to_key      TEXT NOT NULL REFERENCES graph_nodes(node_key) ON DELETE CASCADE,
    edge_type   TEXT NOT NULL
                CHECK (edge_type IN ('ACTED_IN','DIRECTED','HAS_GENRE','HAS_KEYWORD')),
    properties  JSONB NOT NULL DEFAULT '{}'::jsonb,
    source      TEXT NOT NULL DEFAULT 'tmdb',
    confidence  REAL NOT NULL DEFAULT 1.0
                CHECK (confidence >= 0 AND confidence <= 1),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (from_key, to_key, edge_type, source)
);

-- Traversal runs in BOTH directions, so it needs an index at both ends.
CREATE INDEX IF NOT EXISTS idx_graph_nodes_type ON graph_nodes (node_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges (from_key, edge_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to   ON graph_edges (to_key,   edge_type);
SQL
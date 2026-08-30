"""Two files that must agree about the same set of names, checked by a machine.

pipeline/build_graph.py writes edges. graph_schema.sql has a CHECK constraint listing the
edge types it will accept. Nothing has ever verified that those two lists match.
If they drift, the failure arrives at 2am as a constraint violation in the middle
of a load — not at review time, which is when it is cheap.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = (ROOT / "graph_schema.sql").read_text()


def allowed(column):
    """Pull the quoted values out of  CHECK (<column> IN ('a','b',...))."""
    match = re.search(rf"CHECK\s*\(\s*{column}\s+IN\s*\((.*?)\)\s*\)",
                      SCHEMA, re.S)
    assert match, f"no CHECK constraint found for {column}"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_every_edge_type_build_graph_writes_is_allowed_by_the_schema():
    from pipeline import build_graph
    written = set(build_graph.OFFER_EDGE.values()) | {
        "ACTED_IN", "DIRECTED", "HAS_GENRE", "HAS_KEYWORD"}
    assert written <= allowed("edge_type"), (
        f"build_graph writes types the schema rejects: {written - allowed('edge_type')}")


def test_the_schema_allows_nothing_build_graph_never_writes():
    # Drift in the other direction: a type nobody produces is dead config, and dead
    # config is read as documentation by the next person.
    from pipeline import build_graph
    written = set(build_graph.OFFER_EDGE.values()) | {
        "ACTED_IN", "DIRECTED", "HAS_GENRE", "HAS_KEYWORD"}
    assert allowed("edge_type") <= written, (
        f"schema allows types nothing writes: {allowed('edge_type') - written}")


def test_node_types_match():
    expected = {"film", "person", "genre", "keyword", "provider"}
    assert allowed("node_type") == expected


def test_every_offer_category_maps_to_a_band():
    from pipeline import build_graph
    from backend import providers
    for edge_type in build_graph.OFFER_EDGE.values():
        assert edge_type in providers.BAND_FROM_EDGE, (
            f"{edge_type} is written to the graph but providers.py cannot band it")


def test_the_schema_file_has_no_stray_heredoc_terminator():
    # This actually happened: the word SQL sat as the last line for weeks and made
    # `psql -f graph_schema.sql` impossible. Cheap to check, so check it.
    for line in SCHEMA.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("--"):
            assert not re.fullmatch(r"[A-Z]{2,10}", stripped), (
                f"stray heredoc terminator in graph_schema.sql: {stripped!r}")

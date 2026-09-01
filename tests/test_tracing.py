"""The pipeline must be visible in LangSmith, and stay visible.

WHAT HAPPENED
    `backend/tracing.py` exists so that plain-Python steps — the SQL search, the
    embedding call, the reranker, the graph queries — appear in a trace instead of
    being swallowed into one opaque box between two LangGraph nodes. Its docstring says
    exactly that.

    It was imported by three modules and applied to NOTHING. Zero decorators. For a week
    every trace showed the agent's nodes and the Bedrock calls and nothing in between,
    and no check noticed, because an unused import is not an error.

WHY THIS TEST READS THE SOURCE
    Whether `@traceable` actually wraps a function at run time depends on whether the
    langsmith package is installed, so an attribute check would pass or fail for the
    wrong reason. The decorator being written down is the thing that must not be lost —
    so that is what is asserted, from the AST.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The steps a person debugging a bad recommendation needs to see, one line each.
MUST_BE_TRACED = {
    "backend/models.py":    ["embed", "rerank"],
    "backend/retrieval.py": ["search", "get_film", "_collapse_to_films",
                             "excluded_by_filters"],
    "backend/graph.py":     ["graph_find", "availability", "graph_genres",
                             "graph_film_titles"],
}


def decorators_on(path):
    """{function name: [decorator names]} for one module."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = []
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Attribute):
                names.append(target.attr)
        found[node.name] = names
    return found


def test_every_step_worth_seeing_is_traced():
    for path, functions in MUST_BE_TRACED.items():
        decorators = decorators_on(path)
        for function in functions:
            assert function in decorators, f"{path} has no function named {function}"
            assert "traceable" in decorators[function], (
                f"{path}::{function} is not decorated with @traceable — it will be "
                f"invisible in LangSmith, which is how it was for a week")


def test_tracing_is_never_imported_without_being_used():
    """The precise smell that hid the bug: the import present, the decorator absent."""
    for path in MUST_BE_TRACED:
        source = (ROOT / path).read_text(encoding="utf-8")
        if "from backend.tracing import traceable" in source:
            assert "@traceable" in source, (
                f"{path} imports traceable and never applies it")

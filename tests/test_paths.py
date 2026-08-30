"""Every path built from __file__ must land on the repository root.

WHY THIS FILE EXISTS
    On 30 Aug api.py moved from the root into backend/. Its one line of path arithmetic —
    `os.path.dirname(os.path.abspath(__file__))` — moved with it and kept meaning "the
    folder this file is in", which used to be the root and now is not. Every page served
    from disk answered 500.

    Nothing caught it. The file parsed, the imports resolved, 37 tests passed and all four
    CI jobs were green, because a path assembled at run time out of strings is invisible to
    a static check. Only running the app found it, and only a person can do that.

    So this is the rule, enforced: a file that reaches for the repository root must count
    the levels correctly for where it actually sits. Move the file, and this test fails
    before the browser does.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
FOLDERS = ["", "backend", "pipeline", "evals", "scripts", "tests"]


def dirname_depth(node):
    """How many os.path.dirname() calls wrap this expression, walking outward is hard —
    so walk INWARD instead: count the chain from the outermost call down to abspath."""
    depth = 0
    while (isinstance(node, ast.Call)
           and isinstance(node.func, ast.Attribute)
           and node.func.attr == "dirname"
           and node.args):
        depth += 1
        node = node.args[0]
    return depth, node


def mentions_file(node):
    """Is this the `os.path.abspath(__file__)` at the bottom of the chain?"""
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "abspath"
            and any(isinstance(a, ast.Name) and a.id == "__file__" for a in node.args))


def python_files():
    for folder in FOLDERS:
        base = ROOT / folder if folder else ROOT
        for path in sorted(base.glob("*.py")):
            yield path


def test_every_root_path_counts_its_own_depth_correctly():
    checked = 0
    for path in python_files():
        levels_below_root = len(path.relative_to(ROOT).parts)   # backend/api.py -> 2
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # ast.walk visits dirname(dirname(x)) twice — once as the whole thing, once as
        # the inner half. Judging both would report a correct two-level chain as a
        # one-level mistake, which is how the first version of this test failed.
        # So: note every node that is already the ARGUMENT of a dirname, and skip those.
        nested = {id(n.args[0]) for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "dirname" and n.args}
        for node in ast.walk(tree):
            if id(node) in nested:
                continue
            depth, inner = dirname_depth(node)
            if not depth or not mentions_file(inner):
                continue
            checked += 1
            assert depth == levels_below_root, (
                f"{path.relative_to(ROOT)} wraps os.path.abspath(__file__) in {depth} "
                f"dirname() call(s), which reaches "
                f"{'the repository root' if depth == levels_below_root else 'the wrong folder'}"
                f" — a file {levels_below_root} level(s) below the root needs exactly "
                f"{levels_below_root}")
    assert checked, "found no path arithmetic at all — has this test stopped looking?"


def test_the_api_can_find_the_page_it_serves():
    # The concrete case, stated plainly. static/index.html is committed; static/app/ is
    # build output and deliberately absent from a clean checkout, so it is NOT asserted.
    from backend import api
    assert pathlib.Path(api.ROOT) == ROOT
    assert (pathlib.Path(api.STATIC) / "index.html").is_file()

"""The container installs a shorter list than the laptop. It must still be complete.

WHY THIS EXISTS
    requirements.txt is the development freeze — 100+ packages including pandas,
    pyarrow, datasets and ragas, none of which the running application imports.
    Shipping all of it means a slower build and a much larger image for no benefit,
    so the container installs requirements-runtime.txt instead.

    Two files listing overlapping versions is the exact shape that rots. One gets a
    security bump, the other does not, and the bug appears only in production — which
    is the one place nobody is watching a test run. So the drift is checked here.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# import name -> distribution name, where they differ. Same map repo_check uses.
ALIAS = {"dotenv": "python-dotenv", "yaml": "pyyaml", "sklearn": "scikit-learn",
         "PIL": "pillow"}

# Imported as `langgraph.x` but shipped as separate distributions, so scanning imports
# alone can never find them. Listed once, here, where the reason is written down.
SUBPACKAGES = {"langgraph-checkpoint", "langgraph-prebuilt"}

# Nothing imports the web server; the web server imports everything else.
UNIMPORTED = {"uvicorn"}

# Never imported by name and absolutely required: psycopg-binary is the compiled libpq
# driver that psycopg loads at connect time. Without it the container needs Postgres
# client libraries installed at the OS level, and the failure arrives as a connection
# error rather than an import error — much later, and much less obvious.
BINARY_WHEELS = {"psycopg-binary"}


def pins(filename):
    """{distribution: full pinned line} for a requirements file."""
    out = {}
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[=<>!\[]", line, maxsplit=1)[0].strip()
        out[name.lower().replace("_", "-")] = line
    return out


def third_party_imports_in_backend():
    stdlib = set(sys.stdlib_module_names)
    local = {"backend", "pipeline", "evals", "scripts", "tests", "experiments"}
    found = set()
    for path in sorted((ROOT / "backend").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name not in stdlib and name not in local:
                    found.add(ALIAS.get(name, name).lower().replace("_", "-"))
    return found


def test_the_container_installs_everything_the_app_imports():
    runtime = pins("requirements-runtime.txt")
    for dist in sorted(third_party_imports_in_backend()):
        assert dist in runtime, (
            f"backend/ imports '{dist}' and requirements-runtime.txt does not install "
            f"it — the container will start and then fail on the first request")


def test_the_two_files_never_disagree_about_a_version():
    full, runtime = pins("requirements.txt"), pins("requirements-runtime.txt")
    for dist, line in runtime.items():
        assert dist in full, f"{dist} is in the runtime file but not the full freeze"
        assert line == full[dist], (
            f"{dist}: runtime says '{line}', requirements.txt says '{full[dist]}' — "
            f"the container would run a different version than anything you tested")


def test_nothing_is_carried_that_nothing_needs():
    """The reverse drift: a package kept in the image long after its import went away."""
    runtime = set(pins("requirements-runtime.txt"))
    needed = (third_party_imports_in_backend() | SUBPACKAGES | UNIMPORTED
              | BINARY_WHEELS)
    # botocore arrives with boto3 and is pinned deliberately, so it is imported anyway.
    extra = runtime - needed
    assert not extra, (
        f"requirements-runtime.txt installs {sorted(extra)}, which backend/ never "
        f"imports. Either something needs it (say so here) or it is dead weight")

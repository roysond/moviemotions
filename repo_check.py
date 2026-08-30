"""
repo_check.py — the checks a machine can run without credentials.

WHY THIS EXISTS
    On 25 Aug a manual review of this repository found five real defects: a .gitignore
    rule silently excluding graph_schema.sql, three environment variables the code read
    but .env.example never mentioned, stale metrics in the README, dead scripts, and a
    doc pointing at a file that did not exist.

    Every one of those was findable by a script. A review that only happens when someone
    remembers to do it is not a control. This turns that review into a gate.

WHAT IT CANNOT CHECK, AND WHY THAT IS FINE
    Nothing here touches Postgres, Bedrock or OpenRouter. CI has no credentials and
    should not have any — a workflow that needs your AWS keys is a workflow that can leak
    them. So this checks STRUCTURE, and the evals check BEHAVIOUR on your machine.
    Two different jobs. Do not try to make this one do the other.

    python repo_check.py           # human-readable, exits non-zero on failure
"""

import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

failures, warnings = [], []

# Files that failed to parse. Later checks must SKIP these rather than crash on them:
# one broken file should produce one clear failure, not a traceback that hides the
# other five checks. Found by the negative test — the gate exited 1 for the wrong
# reason, which reads exactly like working.
unparseable = set()


def fail(check, detail):
    failures.append((check, detail))


def warn(check, detail):
    warnings.append((check, detail))


def py_files():
    return sorted(glob.glob("*.py")) + sorted(glob.glob("experiments/*.py"))


# ── 1 · every module parses ────────────────────────────────────────────────
def check_syntax():
    for path in py_files():
        try:
            ast.parse(open(path, encoding="utf-8").read(), filename=path)
        except SyntaxError as error:
            unparseable.add(path)
            fail("syntax", f"{path}:{error.lineno} {error.msg}")


# ── 2 · every third-party import is pinned ─────────────────────────────────
# Distribution names use hyphens, import names use underscores. Comparing the two
# naively produced a false positive during the manual review; normalise both.
LOCAL = {os.path.splitext(os.path.basename(p))[0] for p in py_files()}


def check_requirements():
    if not os.path.exists("requirements.txt"):
        return fail("requirements", "requirements.txt is missing")
    pinned = {
        re.split(r"[=<>!\[]", line, maxsplit=1)[0].strip().lower().replace("_", "-")
        for line in open("requirements.txt", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    }
    alias = {"dotenv": "python-dotenv", "yaml": "pyyaml", "sklearn": "scikit-learn",
             "psycopg": "psycopg", "PIL": "pillow"}
    stdlib = set(sys.stdlib_module_names)
    for path in py_files():
        if path in unparseable:
            continue                      # already reported by the syntax check
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name in stdlib or name in LOCAL:
                    continue
                dist = alias.get(name, name).lower().replace("_", "-")
                if dist not in pinned:
                    fail("requirements",
                         f"{path} imports '{name}' — '{dist}' is not in requirements.txt")


# ── 3 · .env.example matches what the code actually reads ──────────────────
ENV_READ = re.compile(r"""os\.(?:environ\[|environ\.get\(|getenv\()\s*["']([A-Z][A-Z0-9_]*)["']""")
# Read by a library from the environment, never by our code. Declaring them is correct.
IMPLICIT = {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "LANGSMITH_API_KEY"}


def check_env_example():
    if not os.path.exists(".env.example"):
        return fail("env", ".env.example is missing")
    declared = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", open(".env.example", encoding="utf-8").read(),
                              re.MULTILINE))
    read = {}
    for path in py_files():
        for name in ENV_READ.findall(open(path, encoding="utf-8").read()):
            read.setdefault(name, path)
    for name, path in sorted(read.items()):
        if name not in declared:
            fail("env", f"{name} is read in {path} but not declared in .env.example")
    for name in sorted(declared - set(read) - IMPLICIT):
        warn("env", f"{name} is declared in .env.example but no code reads it")


# ── 4 · no doc points at a file that does not exist ────────────────────────
REFERENCE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|md|sql|html|json|txt|yml|yaml))`")


def check_doc_references():
    docs = ["README.md"] + sorted(glob.glob("docs/*.md"))
    for doc in docs:
        if not os.path.exists(doc):
            continue
        # session notes are a historical log; they are SUPPOSED to mention deleted files
        if os.path.basename(doc) in {"session-notes.md", "roadmap.md", "groundwork.md"}:
            continue
        for ref in sorted(set(REFERENCE.findall(open(doc, encoding="utf-8").read()))):
            if any(os.path.exists(p) for p in (ref, f"docs/{ref}", f"experiments/{ref}")):
                continue
            fail("docs", f"{doc} references '{ref}', which does not exist")


# ── 5 · secrets never reach the repository ─────────────────────────────────
# Report the LOCATION, never the value. A check that prints the secret it found has
# leaked it into the CI log, which is public on a public repository.
SECRET_SHAPES = [
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("OpenRouter key", re.compile(r"\bsk-or-v1-[A-Za-z0-9]{20,}")),
    ("LangSmith key", re.compile(r"\blsv2_[A-Za-z0-9_]{20,}")),
    ("postgres URL with a password",
     re.compile(r"postgres(?:ql)?://([^:\s]+):([^@\s]+)@")),
]

# Documentation MUST show the shape of a connection string. These are the words people
# use when they mean "put yours here" — a scanner that cannot tell an example from a
# credential gets switched off, and a switched-off scanner catches nothing.
PLACEHOLDERS = {"user", "username", "pass", "password", "passwd", "youruser",
                "yourpassword", "your-user", "your-password", "postgres", "changeme",
                "xxx", "***", "user_name", "my_user", "my_password"}


def is_placeholder(match):
    parts = [g for g in (match.groups() or ()) if g]
    if not parts:
        return False
    return any(p.strip("<>{}[]").lower() in PLACEHOLDERS or p.startswith(("<", "{", "$"))
               for p in parts)
SCAN_GLOBS = ["*.py", "*.md", "*.sql", "*.txt", "*.yml", "*.html",
              "docs/*.md", "docs/*.html", "experiments/*.py", ".env.example"]


def check_no_secrets():
    for pattern in SCAN_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for label, shape in SECRET_SHAPES:
                for match in shape.finditer(text):
                    if is_placeholder(match):
                        continue
                    line = text[:match.start()].count("\n") + 1
                    fail("secrets", f"{path}:{line} looks like a {label} — value not shown")


# ── 6 · .gitignore protects secrets and does not swallow source ────────────
def check_gitignore():
    if not os.path.exists(".gitignore"):
        return fail("gitignore", ".gitignore is missing")
    rules = [ln.strip() for ln in open(".gitignore", encoding="utf-8")
             if ln.strip() and not ln.startswith("#")]
    if ".env" not in rules:
        fail("gitignore", ".env is not ignored — credentials would be committed")
    negations = {r.lstrip("!") for r in rules if r.startswith("!")}
    # A blanket *.sql once hid graph_schema.sql for a whole session. Any schema file
    # must be explicitly un-ignored, or it never reaches anyone who clones the repo.
    if "*.sql" in rules:
        for path in glob.glob("*.sql"):
            if path not in negations:
                fail("gitignore",
                     f"'*.sql' ignores {path} and nothing un-ignores it — "
                     f"add '!{path}' or the schema never ships")


CHECKS = [
    ("syntax", check_syntax),
    ("requirements", check_requirements),
    ("env", check_env_example),
    ("docs", check_doc_references),
    ("secrets", check_no_secrets),
    ("gitignore", check_gitignore),
]

if __name__ == "__main__":
    print("=" * 74)
    print("REPO CHECK — structure only. Behaviour is checked by the evals, on your machine.")
    print("=" * 74)
    for name, run in CHECKS:
        before = len(failures)
        run()
        added = len(failures) - before
        print(f"  {'FAIL' if added else ' ok ':4}  {name:<14} "
              f"{f'{added} problem(s)' if added else 'clean'}")

    if warnings:
        print("\nwarnings — not failures, but worth a look:")
        for check, detail in warnings:
            print(f"  · [{check}] {detail}")

    if failures:
        print("\n" + "=" * 74)
        print(f"{len(failures)} PROBLEM(S)")
        print("=" * 74)
        for check, detail in failures:
            print(f"  [{check}] {detail}")
        sys.exit(1)

    print("\nall structural checks passed.")

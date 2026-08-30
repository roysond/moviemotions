"""Shared setup for every test.

WHY THIS FILE EXISTS AT ALL
    core.py and agent.py read os.environ["..."] at IMPORT time. That is a good choice
    in an application: it fails loudly on a misconfigured machine instead of halfway
    through a run. It is awkward in a test, because merely importing the module demands
    credentials that CI must never hold.

    So we plant harmless placeholders first. Nothing connects to anything — these tests
    exercise arithmetic, sorting and string matching, and never open a socket.

WHY THE LIST IS SCANNED AND NOT TYPED OUT
    It was typed out once, and it was wrong: it said AWS_DEFAULT_REGION where the code
    says AWS_REGION. The tests still passed on the laptop, because a developer shell has
    already exported the real values from .env — and then failed the moment CI ran them
    on a clean machine with nothing set.

    A hand-written list is a second copy of a fact the code already states, and second
    copies drift. So this reads the code and derives the list. Add a new required
    variable tomorrow and this keeps working with no edit here.

    Same principle as REGION living only in providers.py: one definition, no drift.
"""

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # so `import core` works from the repo root

# Values that must look real enough for a library to accept them at import time.
# Everything else gets a plainly fake string, so a placeholder can never be mistaken
# for a working credential in a log.
REALISTIC = {
    "AWS_REGION": "us-east-1",
    "AWS_DEFAULT_REGION": "us-east-1",
    "DATABASE_URL": "postgresql://placeholder:placeholder@localhost:5432/placeholder",
    "LANGSMITH_TRACING": "false",
}

REQUIRED = re.compile(r"""os\.environ\[\s*['"]([A-Z0-9_]+)['"]\s*\]""")


def required_variables():
    """Every name the application demands outright, read from the source."""
    names = set()
    for path in ROOT.glob("*.py"):                 # the app; experiments are not tested
        names |= set(REQUIRED.findall(path.read_text(encoding="utf-8")))
    return names


for name in sorted(required_variables()):
    os.environ.setdefault(name, REALISTIC.get(name, f"placeholder-{name.lower()}"))

# agent.py reads BEDROCK_MODEL_AGENT with a fallback rather than outright, so the scan
# above will not see it. Named here on purpose, with the reason.
os.environ.setdefault("BEDROCK_MODEL_AGENT", "placeholder-bedrock-model-agent")

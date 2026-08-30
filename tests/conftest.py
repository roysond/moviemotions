"""Shared setup for every test.

WHY THIS FILE EXISTS AT ALL
    core.py reads os.environ["DATABASE_URL"] and os.environ["BEDROCK_MODEL_EMBED_NOVA"]
    at IMPORT time. That is convenient in an application — it fails loudly on a machine
    that is misconfigured, rather than halfway through a run. It is inconvenient in a
    test, because merely importing the module demands credentials that CI must never have.

    So we plant harmless placeholder values before anything is imported. Nothing connects
    to anything: these tests only exercise pure functions — arithmetic, sorting, string
    matching — and never open a socket.

    Worth noticing as a design lesson: reading the environment at import time makes code
    harder to test. Not wrong, but it has a cost, and this file is the cost.
"""

import os
import sys

PLACEHOLDERS = {
    "DATABASE_URL": "postgresql://placeholder/placeholder",
    "BEDROCK_MODEL_EMBED_NOVA": "placeholder",
    "BEDROCK_MODEL_AGENT": "placeholder",
    "OPENROUTER_API_KEY": "placeholder-not-a-real-key",
    "AWS_DEFAULT_REGION": "us-east-1",
    "LANGSMITH_TRACING": "false",
}
for name, value in PLACEHOLDERS.items():
    os.environ.setdefault(name, value)

# so `import core` works when pytest is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

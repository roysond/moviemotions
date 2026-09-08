"""What a vector IS in this system: which run produced it, and how it is written down.

WHY THIS FILE EXISTS AT ALL
    Two files need these two facts — the script that WRITES vectors and the code that
    READS them — and they must agree exactly. A variant string typed out twice is a
    variant string that will one day disagree with itself, and the failure is silent:
    the writer stores vectors under one name, the reader looks for another, and search
    quietly returns nothing at all. Same lesson as REGION being defined in two files.

    It is a tiny module on purpose. One reason to change: how a vector is identified.
"""

from backend.config import DIMENSIONS, MODEL_ID

# An embedding is identified by TEXT x MODEL x WHAT-WAS-EMBEDDED. The first is the row;
# these are the other two.
#
#   "plain" — the stored content, embedded as-is with nothing added.
#
# A later experiment that prepends a film-level header would use a different word, and
# both sets of vectors could then live in the table at once and be compared, instead of
# one overwriting the other and taking the evidence with it.
EMBED_VARIANT = f"{MODEL_ID}@{DIMENSIONS}/plain"


def as_literal(vector):
    """A list of floats, in the text form Postgres accepts for a vector column.

    psycopg has no idea what a pgvector is, so the value is sent as '[0.1,0.2,...]' and
    cast in the SQL with ::vector. Written out rather than installing the pgvector
    adapter package: one dependency avoided, and these two lines say exactly what
    crosses the wire.
    """
    return "[" + ",".join(str(number) for number in vector) + "]"

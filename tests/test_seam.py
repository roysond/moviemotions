"""backend/models.py must stay the only file in backend/ that names a model vendor.

WHAT HAPPENED
    The README claimed for weeks that models.py was the only file naming a vendor.
    It was not. `backend/agent.py` built ChatBedrockConverse directly, in the middle
    of the file, so the agent's own model — the one doing the reasoning — sat outside
    the seam while the embedder and the reranker sat inside it.

    Nobody noticed because nothing checked. A claim in a docstring is a hope.

WHY AN AST WALK AND NOT A GREP
    A grep over the whole file matches the word "bedrock" inside a comment, and this
    repository's comments discuss vendors constantly and should keep doing so. Only
    an IMPORT actually couples a file to a vendor, so only imports are inspected.

WHY THE LAZY IMPORTS INSIDE chat_model() ARE FINE
    They are still inside models.py, which is the file allowed to have them. The
    laziness is about keeping ~60MB of Google packages out of a Bedrock-only image,
    not about hiding from this test.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Import a module from this list and you have named a model vendor.
VENDOR_MODULES = {
    "boto3", "botocore",
    "langchain_aws",
    "langchain_google_genai", "langchain_google_vertexai",
    "vertexai", "google.cloud.aiplatform", "openai", "anthropic", "cohere",
}

ALLOWED = {"models.py"}


def vendor_imports(path):
    """Every vendor module this file imports, at any nesting depth."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            # "google.cloud.aiplatform" must match, and so must "boto3.session".
            root = name.split(".")[0]
            if name in VENDOR_MODULES or root in VENDOR_MODULES:
                found.add(name)
    return found


def test_only_models_names_a_vendor():
    offenders = {}
    for path in sorted((ROOT / "backend").glob("*.py")):
        if path.name in ALLOWED:
            continue
        found = vendor_imports(path)
        if found:
            offenders[path.name] = sorted(found)
    assert not offenders, (
        f"these files import a model vendor and should go through "
        f"backend/models.py instead: {offenders}")


def test_models_actually_holds_the_vendors():
    """The mirror image. If models.py imports nothing, the seam is empty and the
    test above passes for the wrong reason."""
    assert vendor_imports(ROOT / "backend" / "models.py")


def test_chat_model_covers_every_declared_provider():
    """Every provider named in config must be one chat_model() can build.

    The two lists are written in different files and drift silently: adding a
    provider to config's docstring without a branch here fails at the first
    question, not at startup.
    """
    source = (ROOT / "backend" / "models.py").read_text(encoding="utf-8")
    for provider in ("bedrock", "vertex"):
        assert f'LLM_PROVIDER == "{provider}"' in source, provider

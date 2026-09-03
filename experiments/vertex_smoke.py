"""The smallest thing that can prove Vertex AI works. Nothing else imports this.

WHY A THROWAWAY SCRIPT AND NOT A CHANGE TO backend/
    Three things have to be true before Gemini can answer: the API enabled, the
    credentials readable by a library rather than by the shell, and the model id
    correct. If all three are wired into the agent at once, a failure could be any of
    them — plus the agent's own prompt, its tools, and LangGraph's state.

    So this file asks one question: does a request reach Gemini and come back?
    It lives in experiments/, which is the folder meant to be deleted.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

PROJECT = os.environ.get("GCP_PROJECT")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
MODEL = os.environ.get("VERTEX_MODEL_AGENT", "gemini-3.8-flash")

if not PROJECT:
    sys.exit("GCP_PROJECT is not set in .env")

print(f"project  {PROJECT}")
print(f"location {LOCATION}")
print(f"model    {MODEL}")
print()

from langchain_google_vertexai import ChatVertexAI      # noqa: E402

llm = ChatVertexAI(model_name=MODEL, project=PROJECT, location=LOCATION,
                   temperature=0)

reply = llm.invoke("Reply with exactly these five words: Vertex AI is reachable now.")
print("REPLY:", reply.content)

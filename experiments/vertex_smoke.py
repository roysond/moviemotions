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

# ChatGoogleGenerativeAI, NOT ChatVertexAI.
#
# LangChain deprecated ChatVertexAI in 3.2.0 and removes it in 4.0.0. Google is
# consolidating two SDKs into one, and the runtime says so on every call.
#
# WHY THE ARGUMENTS MATTER MORE THAN THE CLASS NAME
#     This class can reach Gemini two ways. Given an api_key it uses Google AI
#     Studio — simple, and a long-lived secret in a file. Given `vertexai=True`
#     with a project it uses Vertex AI and authenticates with Application
#     Default Credentials, which live outside this project and rotate.
#
#     Same model either way. Only one of them keeps a key out of the repository,
#     so the arguments below are the security decision, not the import line.
from langchain_google_genai import ChatGoogleGenerativeAI      # noqa: E402

llm = ChatGoogleGenerativeAI(model=MODEL, vertexai=True, project=PROJECT,
                             location=LOCATION, temperature=0)

reply = llm.invoke("Reply with exactly these five words: Vertex AI is reachable now.")
print("REPLY:", reply.content)

"""HTTP interface over the agent — the same graph, driven by a browser instead of a terminal.

WHAT THIS IS NOT
    It is not a second copy of the agent. There is no logic here: no prompts, no tools, no
    loop. It imports the SAME compiled graph that agent.py runs and pushes the same state
    through it. If the two ever disagree, one of them is a bug.

THE INTERESTING PART — a pause that survives an HTTP request
    In the terminal, `interrupt()` pauses and a Python while-loop resumes it moments later.
    Over HTTP the pause has to survive the request ENDING. It does, and nothing had to be
    added to make that work:

        POST /api/ask     graph runs -> hits interrupt() -> state written to checkpointer
                          -> HTTP 200 returns {state: "review", draft, thread_id}
                          -> the request is over. the server is idle. nothing is held open.

        POST /api/resume  Command(resume=...) with the SAME thread_id
                          -> checkpoint reloaded, review node re-entered, loop continues

    That is the checkpointer earning its place. A blocking prompt could never do this —
    an HTTP handler cannot sit and wait for a human to make up their mind.

    InMemorySaver keeps threads in this process, so a server restart forgets them. Swapping
    in a Postgres saver is the only change needed to survive one.

RUN
    uvicorn api:app --reload --port 8000     then open http://127.0.0.1:8000
"""

import json
import os
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from agent import AGENT_MODEL, MAX_PASSES, graph, split_content
from core import search

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="MovieMotions")

# "1. Predator (1987) · 107 min · score 0.614 · matched on its plot text"
FILM_LINE = re.compile(
    r"^\s*(\d+)\.\s+(.*?)\s+\((\d{4}|----)\)\s+·\s+(.*?)\s+·\s+score\s+([\d.]+)"
    r"\s+·\s+matched on its (\w+) text"
)


def parse_films(text):
    """Turn a tool's plain-text result into rows a page can draw.

    The tool answers a MODEL, so its output is prose — that is the right call and it stays
    that way. Parsing for display happens here, at the edge, rather than bending the tool's
    contract to suit a screen.
    """
    films, current = [], None
    for line in text.splitlines():
        match = FILM_LINE.match(line)
        if match:
            rank, title, year, runtime, score, source = match.groups()
            current = {"rank": int(rank), "title": title, "year": year, "runtime": runtime,
                       "score": float(score), "source": source, "evidence": ""}
            films.append(current)
        elif current is not None and line.strip().startswith('"'):
            current["evidence"] = line.strip().strip('"')
    return films


def trace_of(messages):
    """Flatten a LangGraph message list into display steps."""
    steps = []
    for message in messages:
        kind = message.__class__.__name__
        if kind == "HumanMessage":
            steps.append({"kind": "human", "text": str(message.content)})
        elif kind == "ToolMessage":
            body = str(message.content)
            steps.append({"kind": "tool_result", "tool": getattr(message, "name", ""),
                          "text": body, "films": parse_films(body)})
        else:
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                steps.append({"kind": "tool_call", "tool": call["name"], "args": call["args"]})
            visible, reasoning = split_content(message)
            if visible:
                steps.append({"kind": "ai", "text": visible, "reasoning": reasoning})
    return steps


def config_for(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": MAX_PASSES * 3,
        "run_name": "moviemotions-web",
        "metadata": {"agent_model": AGENT_MODEL, "surface": "web"},
    }


def advance(payload, thread_id):
    """Push the graph forward one leg and describe where it stopped."""
    result = graph.invoke(payload, config_for(thread_id))
    messages = result.get("messages", [])
    if result.get("__interrupt__"):
        return {"thread_id": thread_id, "state": "review",
                "draft": (result["__interrupt__"][0].value or {}).get("draft", ""),
                "trace": trace_of(messages)}
    return {"thread_id": thread_id, "state": "done",
            "answer": split_content(messages[-1])[0] if messages else "",
            "trace": trace_of(messages)}


class Ask(BaseModel):
    question: str
    thread_id: str


class Resume(BaseModel):
    thread_id: str
    action: str          # approve | edit | revise
    text: str = ""
    note: str = ""


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/meta")
def meta():
    return {"model": AGENT_MODEL, "max_passes": MAX_PASSES,
            "tracing": os.environ.get("LANGSMITH_TRACING", "") == "true",
            "project": os.environ.get("LANGSMITH_PROJECT", "")}


@app.post("/api/ask")
def ask(request: Ask):
    return advance({"messages": [HumanMessage(request.question)]}, request.thread_id)


@app.post("/api/resume")
def resume(request: Resume):
    decision = {"action": request.action}
    if request.action == "edit":
        decision["text"] = request.text
    if request.action == "revise":
        decision["note"] = request.note
    return advance(Command(resume=decision), request.thread_id)


@app.get("/api/eval")
def last_eval():
    path = os.path.join(HERE, "data", "agent_eval.json")
    if not os.path.exists(path):
        return {"cases": []}
    return {"cases": json.load(open(path))}


class Retrieve(BaseModel):
    query: str
    limit: int = 5


@app.post("/api/search")
def raw_search(request: Retrieve):
    """Retrieval with no agent at all — for comparing what the model was given."""
    return {"query": request.query, "results": search(request.query, request.limit)}

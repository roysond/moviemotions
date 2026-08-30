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
import providers
from core import availability, graph_film_titles, search

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
        # x4, not x3: the critic adds a node to every lap, and the limit counts NODE
        # EXECUTIONS rather than laps. Left at x3 a long conversation would hit the
        # backstop and look like a runaway loop when nothing is wrong.
        "recursion_limit": MAX_PASSES * 4,
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


# ─────────────────────────────────────────────────────────────────────────────
# THE RIGHT-HAND PANEL
#
# The chat produces prose. The panel needs rows. Rather than make the agent emit
# JSON — which would bend the tool contract to suit a screen — the answer is
# matched against the catalogue HERE, at the edge. Same reasoning as parse_films.
#
# Image URLs are built here, not in the browser, so TMDB's URL shape lives in one
# place. If it ever changes, one file changes.
# ─────────────────────────────────────────────────────────────────────────────

TMDB_IMAGE = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"
LOGO_SIZE = "w45"


# Words that turn a mention into a REJECTION. "…but are not Jurassic Park" names
# the film in order to rule it out, and a panel that cannot tell the difference
# proudly recommends the one film the agent just excluded.
NEGATIONS = ("not ", "n't ", "other than", "besides", "except", "excluding",
             "rather than", "instead of", "unlike", "apart from", "aside from")
NEGATION_WINDOW = 40      # characters before the title to inspect


def films_mentioned(answer, titles=None, exclude=()):
    """Which films does this answer actually RECOMMEND, in the order named?

    Only exact catalogue titles count, and only mentions that are not rejections.
    The panel can never show a film the agent did not name — the same grounding
    rule the agent itself works under — nor one it named in order to dismiss.
    """
    dismissed = {name.strip().lower() for name in exclude if name}
    # `titles` is injected by the tests so this can be checked without a database.
    # A function that reaches out and fetches its own input cannot be tested cheaply.
    if titles is None:
        titles = graph_film_titles()
    titles = sorted(titles, key=len, reverse=True)

    lowered = answer.lower()
    found, claimed = [], []
    for title in titles:                       # longest first
        if title.lower() in dismissed:         # the agent asked us to leave it out
            continue
        at = lowered.find(title.lower())
        if at == -1:
            continue
        # A negation governs its own SENTENCE and nothing after it. Two false
        # positives had to be fixed before this behaved:
        #   a plain 40-character window reached across the line break from
        #     `...but are not "Jurassic Park":` into `1. **Predator (1987)**`
        #   stopping at the line break alone still let "Predator is not for
        #     everyone. Alien is safer." swallow Alien
        # So: look back only as far as the previous sentence end or line break.
        start = max((lowered.rfind(mark, 0, at) for mark in "\n.!?:"), default=-1) + 1
        before = lowered[max(start, at - NEGATION_WINDOW):at]
        if any(word in before for word in NEGATIONS):
            continue                           # named only to be ruled out
        # A longer title already covering this span wins; "Terminator 2" must not
        # match again inside "Terminator 2: Judgment Day".
        if any(start <= at < end for start, end in claimed):
            continue
        claimed.append((at, at + len(title)))
        found.append((at, title))
    return [title for _, title in sorted(found)]


def reasons_for(answer, title, others=()):
    """The agent's OWN sentences about this film — never our paraphrase.

    Answers are written as blocks separated by blank lines:

        1. **Predator (1987)**
           - A team of elite commandos ... hunted by an extraterrestrial warrior.

    The description sits on the line AFTER the title, so matching line-by-line
    found the heading and threw the actual reason away. Match the block instead,
    then drop the heading, because the title is already displayed beside it.
    """
    rest = [name for name in others if name.lower() != title.lower()]
    marker = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s")

    for block in re.split(r"\n\s*\n", answer):
        if title.lower() not in block.lower():
            continue
        lines, started, indent = [], False, 0
        for line in block.splitlines():
            text = re.sub(r"\*\*", "", line).strip().lstrip("-*0123456789. ").strip()
            if not text:
                continue
            if not started:
                if title.lower() not in text.lower():
                    continue                     # still in the preamble
                started, indent = True, len(line) - len(line.lstrip())
            else:
                # WHERE DOES THIS FILM'S ENTRY END?
                # A compact answer has no blank line between films, so "the block"
                # is the whole list and Alien's sentence would be served up as one
                # of Predator's reasons.
                #
                # Two independent stops, because either can be absent:
                #   1. the line names another film we already matched
                #   2. the line starts a NEW list item at the same indentation as
                #      the title. A description is indented deeper than its own
                #      heading; the next film's entry is not. This one needs no
                #      knowledge of the catalogue, so the function is correct even
                #      when a caller passes no `others` — which is exactly how the
                #      test called it, and how it broke.
                if any(text.lower().startswith(name.lower()) for name in rest):
                    break
                if marker.match(line) and (len(line) - len(line.lstrip())) <= indent:
                    break
            bare = re.sub(r"\s*\(\d{4}\)\s*$", "", text)
            if bare.lower() == title.lower():        # the heading; the panel shows it
                continue
            lines.append(text)
        if lines:
            return lines[:3]
    return []


class Panel(BaseModel):
    answer: str
    exclude: list[str] = []      # titles the agent explicitly asked to leave out


@app.post("/api/panel")
def panel(request: Panel):
    """Everything the results panel draws: poster, the agent's reasons, banded offers."""
    rows = []
    titles = films_mentioned(request.answer, exclude=request.exclude)
    for title in titles:
        found = availability(title)
        if not found["found"]:
            continue
        offers = [{
            "display": o["display"],
            "band": o["band"],
            "band_label": providers.BAND_LABEL[o["band"]],
            "price_text": o["price_text"],
            "verified": o["verified"],
            "note": o["note"],
            "resold_from": o["resold_from"],
            "logo_url": f"{TMDB_IMAGE}/{LOGO_SIZE}{o['logo_path']}" if o.get("logo_path") else None,
        } for o in found["offers"]]
        rows.append({
            "title": found["title"],
            "year": (found["release_date"] or "----")[:4],
            "runtime_minutes": found["runtime_minutes"],
            "poster_url": (f"{TMDB_IMAGE}/{POSTER_SIZE}{found['poster_path']}"
                           if found["poster_path"] else None),
            "reasons": reasons_for(request.answer, found["title"], titles),
            "has_listing": found["has_listing"],
            "region": found["region"],
            "checked_on": found["checked_on"],
            "stale_days": found["stale_days"],
            "link": found["link"],
            "offers": offers,
        })
    return {"films": rows}


@app.get("/api/availability/{title}")
def one_film(title: str):
    """Single film, for poking at by hand."""
    return availability(title)


@app.get("/app")
def app_page():
    """The React build. The original page stays at / until this one replaces it."""
    return FileResponse(os.path.join(HERE, "static", "app", "index.html"))

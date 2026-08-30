"""The agent — a LOOP, not a pipeline. This is what makes MovieMotions agentic.

WHAT CHANGED FROM PASS 0
    Pass 0's Reason step was one LLM call on a fixed path:
        query -> always retrieve -> always one call -> answer
    You could draw that before the query arrived. That is RAG: a pipeline, a noun.

    Here the model decides what it needs, gets it, LOOKS AT WHAT CAME BACK, and
    decides again:
                    ┌─────────────────────────────┐
                    ▼                             │
        START -> [ think ] --needs a tool?--> [ act ]
                    │  no
                    ▼
                   END
    The shape depends on the query, so it cannot be drawn in advance. That is
    agentic: a loop, a verb.

THE THREE PARTS OF ANY LANGGRAPH
    STATE   what travels round the loop. Here: the running list of messages.
            Every pass appends to it, so the model always sees the full history —
            including what its own tool calls returned.
    NODES   the things that do work. `think` asks the model; `act` runs tools.
    EDGES   the wiring. One edge is CONDITIONAL — that single branch is the
            entire difference between a pipeline and an agent.

HOW IT STOPS — two mechanisms, and only one of them is the safety net
    Natural termination: the model replies with an ANSWER instead of a tool call,
        `should_continue` returns END, the loop exits. This is the normal path and
        the one to design for.
    Recursion limit: a hard cap on passes. A BACKSTOP for when natural termination
        fails — a vague tool description makes a model retry forever, and two tools
        with a blurry boundary make it oscillate. Regularly hitting this cap is a
        bug signal, not a working design.
"""

import os
import re
import textwrap
import uuid

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from tools import TOOLS

load_dotenv()

# The agent's model is its own setting, not the same knob as the embedding model.
# One line to swap vendor or tier — the LLMProvider seam from Pass 0, honoured.
AGENT_MODEL = os.environ.get("BEDROCK_MODEL_AGENT", os.environ["BEDROCK_MODEL_TEXT"])
REGION = os.environ["AWS_REGION"]
MAX_PASSES = 6          # backstop only; natural termination should fire long before

SYSTEM_PROMPT = """You are MovieMotions, a film recommendation assistant.

You can only recommend films a tool has returned to you in this conversation. You have
no other catalogue and no reliable memory of what films exist — never recommend a film
from your own knowledge, however sure you feel.

Pick the tool by what the user's sentence CONTAINS, not by what it is about:
- a description of a mood, feeling or plot      -> search_films
- one film named, and they want its details     -> lookup_film
- a person's name, a genre, or "like <a film>"  -> find_films_by_fact
When a sentence contains BOTH a named film and a description — "like Jurassic Park but
more intense" — use search_films, put what that film FEELS LIKE into the query rather than
what it is about, and set exclude_title to the named film. Searching for a film's subject
matter can only find that film again.

A fact is never a matter of degree. If the user names a director, an actor or a genre,
find_films_by_fact is the only correct tool — search_films cannot answer those and will
return confident nonsense if you ask it to.

How to work:
- When the user describes what they want to watch, call search_films with a clear
  description of the film, not a copy of their words.
- Read the scores on search_films results. If the top score is weak, say plainly that
  nothing in the catalogue fits rather than offering the least-bad option.
- find_films_by_fact returns NO scores, because there is nothing to be unsure about.
  State its results plainly, never hedge them, and never re-check them with search_films. An honest "I don't have
  anything like that" is a better answer than a confident wrong one.
- If results look off-target, you may search once more with different wording.
- DO NOT describe every result. The tool always returns five; most are filler. Name only
  the films you would genuinely recommend — often one, sometimes two, sometimes NONE.
  If you catch yourself writing "X is not really a match, but…", do not name X at all.
  Saying "I don't have anything like that" is a complete and correct answer.
- Never quote or repeat the extract. It is there so you know what is TRUE about a film,
  not so you can paste it. Write your own short sentence in your own words.
- EVERY result from EVERY tool carries a QUOTED EXTRACT from the film. Anything you say
  about a film — what it is about, who is in it, how it feels — must come from ITS OWN
  quote. Never describe a film from your own knowledge, even one you are certain about. If the quote does not support the
  reason you want to give, give a reason it does support, or do not recommend the film.
  Naming the right film for an invented reason is still wrong.
- When you have what you need, name each film on ITS OWN LINE, one or two sentences each,
  saying why it fits. No preamble, no bullet characters, no numbering.
- Every sentence must be finished. Never write a placeholder, a trailing "because...",
  an ellipsis standing in for content, or a template of what you were going to say.
  If you cannot complete a sentence, delete it.

Never mention scores, tools, searches, or how you found anything. The user is asking
for a film, not for a description of your machinery. Say "Predator is a good fit
because..." — never "the top result has a relevance score of 0.317".
"""

llm = ChatBedrockConverse(model_id=AGENT_MODEL, region_name=REGION, temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)


def split_content(message):
    """Return (visible_text, reasoning) from a reply.

    Nova answers with a LIST of typed blocks, not a plain string — a `text` block for
    the user plus a `reasoning_content` block showing its own thinking. That reasoning
    is gold for a trace and must never reach the user, so the two are split here rather
    than at the print site. Other models return a plain string; both shapes are handled.
    """
    content = message.content
    if isinstance(content, str):
        return content, ""
    visible, reasoning = [], []
    for block in content:
        if not isinstance(block, dict):
            visible.append(str(block))
        elif block.get("type") == "reasoning_content":
            inner = block.get("reasoning_content") or {}
            reasoning.append(inner.get("text", "") if isinstance(inner, dict) else str(inner))
        elif block.get("type") == "text":
            visible.append(block.get("text", ""))
    return "".join(visible).strip(), " ".join(reasoning).strip()


def think(state: MessagesState) -> dict:
    """NODE — ask the model what to do next, given everything that has happened."""
    reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [reply]}


CRITIC_PROMPT = """You are a fact-checker. You are NOT writing an answer.

Below is EVIDENCE gathered from a film database, then a DRAFT reply written by someone
else. Your only job is to say which lines of the draft are not supported by the evidence.

A line is UNSUPPORTED if it states anything the evidence does not say. Judge only what
is written, not what you happen to know about these films from anywhere else. Treat
these as unsupported:
  - a claim about content the evidence never mentions (violence, gore, romance, humour)
  - a description of a film that does not appear in the evidence for THAT film
  - an assertion about how a film feels, unless the evidence uses such words itself

A line is SUPPORTED if the evidence says it, or says something that plainly means it.
Do not mark a line unsupported merely because it is short, vague, or a greeting.

EVIDENCE
{evidence}

DRAFT
{draft}

Reply with ONLY the numbers of the unsupported lines, separated by commas.
If every line is supported, reply with exactly: NONE
Reply with nothing else — no explanation, no punctuation beyond the commas."""


def critic(state: MessagesState) -> Command:
    """NODE — strike any line the retrieved text does not support.

    THE SPLIT THAT MAKES THIS SAFE
        The model JUDGES and the code EDITS. It is asked for line numbers, never for a
        rewrite. A critic allowed to rewrite can remove one invented claim and introduce
        another in the same breath, and nothing downstream would be able to tell.

    WHY LINE BY LINE
        The system prompt already puts one film per line, so a line is the natural unit.
        It also isolates the opening flourish — "here are films that are more intense and
        have gore" — as its own line, strikeable without touching the recommendations.

    THE GUARD
        If every line is struck, the draft is kept unchanged. An empty answer is a worse
        failure than an unsupported one, and a critic that deletes everything is broken
        rather than strict.
    """
    message = state["messages"][-1]
    draft = split_content(message)[0]
    lines = [ln for ln in draft.splitlines() if ln.strip()]
    if not lines:
        return Command(goto="review")

    # Everything the tools actually returned in this conversation. This is the only
    # thing the draft is allowed to be true about.
    evidence = "\n\n".join(
        str(m.content) for m in state["messages"]
        if m.__class__.__name__ == "ToolMessage"
    )
    if not evidence.strip():
        return Command(goto="review")     # nothing was retrieved; nothing to check against

    numbered = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, start=1))
    try:
        verdict = split_content(llm.invoke([HumanMessage(
            CRITIC_PROMPT.format(evidence=evidence, draft=numbered))]))[0]
    except Exception as error:
        # A critic that cannot run must not block an answer. Degrade, do not fail.
        print(f"  [critic unavailable: {type(error).__name__} — draft passed through]")
        return Command(goto="review")

    struck = {int(n) for n in re.findall(r"\d+", verdict) if 1 <= int(n) <= len(lines)}
    kept = [ln for i, ln in enumerate(lines, start=1) if i not in struck]

    if not kept:                          # THE GUARD
        print("  [critic struck every line — keeping the draft]")
        return Command(goto="review")

    if struck:
        print(f"  [critic struck {len(struck)} of {len(lines)} lines]")
        # DIAGNOSTIC ONLY — changes no behaviour, just makes the deletion visible.
        # "2 of 4 lines struck" tells us nothing about WHICH claim was thrown away.
        for i in sorted(struck):
            print(f"      - struck: {lines[i - 1].strip()[:88]}")

    return Command(goto="review",
                   update={"messages": [AIMessage(content="\n".join(kept),
                                                  id=message.id)]})


# ─────────────────────────────────────────────────────────────────────────────
# THE CRITIC IS OFF. 29 Aug 2026.
#
# It was added to strip unsupported claims out of a draft, and it never earned its
# place: the faithfulness gain it was meant to produce (0.75) sat inside the noise
# floor, so it was shipped UNPROVEN. It is now the prime suspect in a regression
# that reproduces — case 8 of the agent eval lost a correct film two runs running,
# and the critic struck 2 of that answer's 4 lines both times.
#
# We are not diagnosing it right now, because the metric that would judge the fix
# cannot resolve anything smaller than about 0.1 and the critic's effect is smaller
# than that. Measuring with an instrument that cannot see the effect is how you
# argue for a week about nothing.
#
# So: switch it off, return the agent to a state whose behaviour we understand, and
# leave it off until the redesign in docs/verification.md is actually built —
# judge by CLAIM TYPE (the agent may interpret what it was given; it may not add
# facts it was not given) and send a line back for ONE rewrite instead of deleting it.
#
# Flip to True to run it again. The node, its prompt and its guards are untouched.
# ─────────────────────────────────────────────────────────────────────────────
CRITIC_ENABLED = False


def should_continue(state: MessagesState) -> str:
    """THE CONDITIONAL EDGE — the one branch that makes this an agent.

    A reply carrying tool_calls is the model ASKING for something; loop round and
    run it. A reply without tool_calls is the model ANSWERING — and the answer now
    goes past a human before it goes out.
    """
    if getattr(state["messages"][-1], "tool_calls", None):
        return "act"
    return "critic" if CRITIC_ENABLED else "review"


def review(state: MessagesState) -> Command:
    """NODE — HUMAN IN THE LOOP. The graph STOPS here and waits for a person.

    `interrupt()` is not a callback and not a blocking prompt. It throws a special
    exception that LangGraph catches: the graph's state is written to the checkpointer,
    execution ends, and invoke() returns with an `__interrupt__` key holding whatever
    was passed in. The process can exit. Hours can pass. Resuming later with
    Command(resume=<value>) reloads the checkpoint, re-enters THIS node, and this time
    `interrupt()` RETURNS that value instead of throwing.

    That is why a checkpointer is mandatory. Without somewhere to write the state, a
    pause is just a crash — there would be nothing to come back to.

    Three outcomes, and they are deliberately different in kind:
        approve  ship the draft unchanged
        edit     replace the text with the human's wording — the machine defers
        revise   send it back round the loop with a note — the human steers, the
                 machine still does the work

    `revise` is the one that matters: approval alone is a gate, but sending work back
    makes the human part of the loop rather than a rubber stamp at the end of it.
    """
    draft = split_content(state["messages"][-1])[0]
    decision = interrupt({"draft": draft}) or {}          # <-- execution stops HERE
    action = decision.get("action", "approve")

    if action == "revise":
        note = decision.get("note") or "That is not right. Try a different search."
        return Command(goto="think", update={"messages": [HumanMessage(note)]})

    if action == "edit":
        # add_messages REPLACES a message when the id matches and appends when it does
        # not — so reusing the id rewrites history instead of tacking an extra reply on.
        # `or draft`, not `.get(..., draft)`: the key EXISTS and holds "" when the human
        # pressed enter on an empty edit. A blank edit means "leave it", never "erase it".
        edited = AIMessage(content=decision.get("text") or draft,
                           id=state["messages"][-1].id)
        return Command(goto=END, update={"messages": [edited]})

    return Command(goto=END)


builder = StateGraph(MessagesState)
builder.add_node("think", think)
builder.add_node("act", ToolNode(TOOLS))     # runs whatever the model asked for
builder.add_node("critic", critic, destinations=("review",))
builder.add_node("review", review, destinations=("think", END))
builder.add_edge(START, "think")
# Every value should_continue can return must appear here. It can return "review"
# whenever CRITIC_ENABLED is False, and LangGraph rejects a destination that is
# not declared — so switching the critic off broke the graph until this line grew.
builder.add_conditional_edges("think", should_continue,
                              {"act": "act", "critic": "critic", "review": "review"})
builder.add_edge("act", "think")             # <-- the backward edge IS the loop
# A checkpointer is what makes a pause resumable. InMemorySaver keeps it in this
# process; swapping in a Postgres saver is the only change needed to survive a restart.
graph = builder.compile(checkpointer=InMemorySaver())


def _panel_lines(draft: str, width: int = 62) -> list:
    """Render a draft for review WITHOUT destroying its line structure.

    textwrap.wrap() re-flows a whole string as one paragraph, so every newline in it
    silently disappears. In a review panel that is not cosmetic — it makes the human
    reject work the model actually did correctly, and there is no way to tell from the
    panel that anything is wrong.

    A review surface that misrepresents the draft is worse than no review at all.
    Wrap each line separately, and mark real newlines with ↵ so a soft wrap can never
    be mistaken for a hard break.
    """
    out = []
    for raw in draft.split("\n"):
        if not raw.strip():
            out.append("↵")                       # a genuinely blank line
            continue
        wrapped = textwrap.wrap(raw, width) or [""]
        wrapped[-1] = wrapped[-1] + "  ↵"
        out.extend(wrapped)
    if out and out[-1].endswith("  ↵"):
        out[-1] = out[-1][:-3]                    # no marker after the final line
    return out


def ask_human(payload: dict) -> dict:
    """The console front-end for the interrupt — swappable for any UI.

    Worth noticing: NOTHING about pausing lives in here. The graph has already stopped
    and its state is already saved by the time this runs. This function only decides
    what value to hand back. A Slack message, a web form or an approval queue would
    drop straight into this slot; that separation is the whole point.
    """
    print("\n  ┌─ REVIEW ──────────────────────────────────────────────────────────")
    print("  │ about to reply with:   (↵ marks a real line break, not a wrap)")
    print("  │")
    for line in _panel_lines(payload.get("draft", "")) or ["(nothing)"]:
        print(f"  │   {line}")
    print("  │")
    print("  │ [enter] approve    e = edit    r = revise (send back with a note)")
    choice = input("  └─ > ").strip().lower()
    if choice.startswith("e"):
        return {"action": "edit", "text": input("     new wording: ").strip()}
    if choice.startswith("r"):
        return {"action": "revise", "note": input("     what is wrong: ").strip()}
    return {"action": "approve"}


def auto_approve(payload: dict) -> dict:
    """Non-interactive reviewer — for scripted runs and evals."""
    return {"action": "approve"}


def converse(question: str, show_trace: bool = True, decide=ask_human) -> list:
    """Run one turn and return the FULL message list — every think, tool call and result.

    run() returns only the final sentence, which is all a person needs. An EVAL needs the
    whole transcript: which tool was chosen, what arguments it was given, what came back.
    You cannot score a decision you never recorded.
    """
    config = {
        # thread_id names the conversation the checkpointer saves under. Resuming an
        # interrupt means "reload THIS thread", so it must be the same on both calls.
        "configurable": {"thread_id": str(uuid.uuid4())},
        # the review node costs a step per pass, so the backstop allows for it
        # x4, not x3: the critic adds a node to every lap, and the limit counts NODE
        # EXECUTIONS rather than laps. Left at x3 a long conversation would hit the
        # backstop and look like a runaway loop when nothing is wrong.
        "recursion_limit": MAX_PASSES * 4,
        # names the trace in LangSmith; metadata makes runs filterable by model
        "run_name": "moviemotions-agent",
        "metadata": {"agent_model": AGENT_MODEL},
    }
    result = graph.invoke({"messages": [HumanMessage(question)]}, config)

    # A pause is a RETURN, not a block. invoke() came back with __interrupt__ set and
    # the state safely checkpointed; resuming re-enters the review node with the value.
    while result.get("__interrupt__"):
        answer = decide(result["__interrupt__"][0].value)
        result = graph.invoke(Command(resume=answer), config)
    if show_trace:
        print("─" * 74)
        for message in result["messages"]:
            kind = message.__class__.__name__.replace("Message", "").upper()
            calls = getattr(message, "tool_calls", None)
            if calls:
                for call in calls:
                    print(f"  [{kind:9}] wants tool → {call['name']}({call['args']})")
            elif kind == "TOOL":
                # print the WHOLE result, not the first line: the model judged all of it,
                # so a trace that hides the rest cannot explain the answer
                lines = str(message.content).rstrip().splitlines() or [""]
                print(f"  [{kind:9}] returned  → {lines[0]}")
                for extra in lines[1:]:
                    print(f"{'':26}{extra}")
            else:
                visible, reasoning = split_content(message)
                if reasoning:
                    print(f"  [{kind:9}] (thinking) {reasoning[:180]}")
                if visible:
                    for n, line in enumerate(_panel_lines(visible, 70)):
                        print(f"  [{kind:9}] {line}" if n == 0 else f"{'':14}{line}")
        print("─" * 74)
    return result["messages"]


def run(question: str, show_trace: bool = True, decide=ask_human) -> str:
    """Run one turn and return the final answer."""
    return split_content(converse(question, show_trace, decide)[-1])[0]


if __name__ == "__main__":
    questions = [
        "I want something where creatures are hunting people, really tense",
        "do you have a documentary about climate change?",
    ]
    for question in questions:
        print(f"\n\n=== {question}\n")
        answer = run(question)
        print(f"\nANSWER: {answer}")

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
    NODES   the things that do work. `think` decides; `act` runs tools;
            `write` turns the decision into sentences.
    EDGES   the wiring. One edge is CONDITIONAL — that single branch is the
            entire difference between a pipeline and an agent.

TWO MODELS, TWO JOBS — the reasoner and the writer
    `think` is the REASONER and it is the agentic part: it picks a tool, reads what
    came back, and decides whether it has enough. It writes no prose at all. It ends
    by emitting a small JSON verdict — which films, and the phrases that justify them.

    `write` is the WRITER. It receives that verdict and the person's question, and
    NOTHING else: no premise, no scores, no tool output, no history. It chooses
    nothing, calls no tool, runs once.

        query ─▶ think ⇄ act ─(verdict as JSON)─▶ write ─▶ review ─▶ answer
                   the loop                      one pass

    Deciding which film fits and saying it well are different skills, and one model
    asked for both in a single breath does neither: it copies the tool output back.
    Splitting them also makes the grounding check MECHANICAL — `backend/handoff.py`
    drops any film the tools never returned, before the writer ever sees it. A rule
    in a prompt is a request; a record the writer cannot see past is a guarantee.

    Today both slots hold the same model. That is not a mistake — the split is a
    SEAM, and the seam is worth having before there is a second model to put in it.

HOW IT STOPS — two mechanisms, and only one of them is the safety net
    Natural termination: the model replies with an ANSWER instead of a tool call,
        `should_continue` returns END, the loop exits. This is the normal path and
        the one to design for.
    Recursion limit: a hard cap on passes. A BACKSTOP for when natural termination
        fails — a vague tool description makes a model retry forever, and two tools
        with a blurry boundary make it oscillate. Regularly hitting this cap is a
        bug signal, not a working design.
"""

import re
import textwrap
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from backend import handoff
from backend.models import chat_model, writer_model
from backend.tools import TOOLS

load_dotenv()

# Re-exported so backend/api.py and the traces keep importing it from here. WHICH
# model, and whose, is decided in backend/config.py; this file must not know.
from backend.config import AGENT_MODEL, WRITER_MODEL   # noqa: E402,F401
MAX_PASSES = 6          # backstop only; natural termination should fire long before

# THE LIMIT COUNTS NODE EXECUTIONS, NOT LAPS. A lap can run think, act, write, critic
# and review, so the backstop is multiplied to match. Set too low, an ordinary
# conversation trips it and looks like a runaway loop when nothing is wrong.
#
# Defined ONCE. It was computed here and again in api.py, so the two front ends
# disagreed the moment either changed — and the browser is the one place nobody
# thinks to check after editing the graph.
RECURSION_LIMIT = MAX_PASSES * 5

REASONER_PROMPT = """You are the reasoner behind MovieMotions. You DECIDE which films
answer a person's question. Someone else writes the reply, so you write no prose at all.

You can only reach the catalogue through tools.

NEVER name a film unless a tool returned it in this conversation. You have no other
catalogue and no reliable memory of what films exist.

WHICH TOOL
- They want to be GIVEN a film            -> search_films
- They asked ABOUT a film they named      -> lookup_film
A sentence with a film AND a description is ONE search_films call with similar_to and
mood set together. Never two calls, never half of it.

ARGUMENTS
- mood: feeling words, slightly expanded. Not a sentence about films, just feelings.
- max_runtime, min_year, max_year: set these ONLY if an actual NUMBER appears in the
  user's message. Never otherwise. They delete films permanently, so a number you chose
  yourself builds a wall the user never asked for and then hides the catalogue behind it.
- If they give a vague limit and no number, DO NOT CALL ANY TOOL. Reply with one short
  question asking for the number, and nothing else.
- Set no argument they did not ask for.

READING THE RESULT
- Each film carries a score and how far it sits above this query's floor. The floor is
  what UNRELATED looks like for this query. A film sitting near it is not a weak match,
  it is not a match.
- Name only films clearly above the floor. Three if three are clear, two if two are, one
  if one is, none if none are. "I do not have anything like that" is a complete answer.
- Everything you say about a film must come from that film's own MATCHED line or its
  PREMISE. Some matched text is withheld; you may rank on it and must not invent it.
- If the results look off-target you may search once more with different wording. Twice
  is enough.

WHEN YOU ARE DONE, REPLY WITH JSON AND NOTHING ELSE
No sentence before it, none after it, no code fence. The reply IS the object:

{"films": [{"title": "<copied exactly from a tool result>",
            "year": <copied exactly from a tool result>,
            "why": ["<short phrase>", "<short phrase>"]}],
 "nothing_found": false}

- Copy title and year from the tool output character for character. Never from memory.
  A film the tools did not return is thrown away before anyone sees it.
- "why" is a PHRASE, not a sentence: what about this film answers what they asked.
  "creatures hunting people in a jungle", not "This film is a great match because it
  features creatures". One or two phrases. No sentence, no preamble, no full stop.
- Take each phrase from that film's own MATCHED line, FEELS line or PREMISE. Withheld
  text may be ranked on and must never be described.
- Nothing clearly above the floor: {"films": [], "nothing_found": true}. That is a
  complete and correct answer, and a better one than a film that does not fit.
- Best first. At most three.

THE ONE TIME YOU DO NOT REPLY WITH JSON
If a limit is vague and carries no number, call no tool and reply with the one short
question you need answered, as plain text. Having called no tool, there is nothing to
decide yet — so there is no verdict to hand over.
"""

WRITER_PROMPT = """You write the final reply to someone asking for a film.

You are given their question and a RECORD of what was decided. You decide nothing: not
which films, not their order, not whether any of them fit. That is settled.

THE RECORD IS EVERYTHING YOU KNOW ABOUT THESE FILMS. Not a summary of what you know —
the whole of it. You have no other knowledge of them and no way to check. If a thing is
not written in the record it did not happen, and writing it anyway is the one failure
that matters here. No plot, no cast, no comparisons to other films, no atmosphere you
imagined from the title.

FORMAT — mechanical, follow exactly
- The first characters you write are a film's title. There is no opening sentence.
- One film per line, in the order given. No blank lines between them. No bold, no
  asterisks, no bullets, no numbering, no headings.
- After the title, one or two sentences saying WHY IT ANSWERS THEIR QUESTION — built
  from that film's reasons and aimed at the words they actually used.
- After the last film, stop. No closing sentence, no summary, no offer to search again.
- Finish every sentence. No ellipsis, no trailing "because", no placeholder.
- Never mention tools, scores, searches, records, or how the film was found.

THEIR QUESTION
{question}

THE RECORD
{record}
"""

# NOTHING FOUND IS WRITTEN BY CODE, NOT BY A MODEL.
#
# There is nothing to phrase. A refusal has no facts in it, so a model call here buys
# no quality and costs the one thing a model can always do — add something.
NOTHING_FOUND = ("I do not have anything in the catalogue that fits that. Try a "
                 "different feeling, or name a film you liked and I will look around it.")

llm = chat_model()
llm_with_tools = llm.bind_tools(TOOLS)
writer = writer_model()


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
            reasoning.append(inner.get("text", "") if isinstance(inner, dict)
                             else str(inner))
        elif block.get("type") == "text":
            visible.append(block.get("text", ""))
    return "".join(visible).strip(), " ".join(reasoning).strip()


NO_ANSWER = ("I could not put an answer together. Tell me a mood, a genre, or a film "
             "you liked, and I will find something.")


def empty_reply(message):
    """True when this reply would be stored with nothing in it.

    Nova answers in typed blocks. It can return a `reasoning_content` block and no
    `text` block — most often to a greeting, which the system prompt gives it nothing
    to say to. `split_content` then yields "".

    A reply carrying TOOL CALLS is never empty, however blank its text: the tool call
    is the content, and Bedrock encodes it as a toolUse block.
    """
    # .strip(): split_content strips the block-list shape but returns a plain string
    # untouched, so "   \n " arrives here looking like content. It is not.
    return (not split_content(message)[0].strip()
            and not getattr(message, "tool_calls", None))


def think(state: MessagesState) -> dict:
    """NODE — ask the model what to do next, given everything that has happened.

    THE EMPTY-REPLY GUARD
        An empty reply is harmless to SHOW — the user sees a blank — and poisonous to
        KEEP. The checkpointer saves it, the next question in the same thread replays
        the whole conversation, and Bedrock rejects a message whose content is empty:

            ValidationException: The content field in the Message object at
            messages.1 is empty.

        So the turn that breaks is never the turn that caused it. Someone says "hi",
        gets a blank, asks a real question a minute later, and THAT one fails. The two
        halves were each correct; nothing tested the join.

        The fix is to refuse the empty reply at the door rather than to sanitise the
        history later. One place, one rule: nothing empty enters the state.
    """
    reply = llm_with_tools.invoke([SystemMessage(REASONER_PROMPT)] + state["messages"])
    if empty_reply(reply):
        reply = AIMessage(content=NO_ANSWER, id=reply.id)
        return {"messages": [reply]}

    # The scaffolding stripper used to run here, on this node's prose. This node no
    # longer writes prose — it writes a verdict — so the stripper moved to `write`,
    # which is where prose is now made.
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


LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*\u2022])\s+")


def tool_evidence(messages):
    """Everything the tools returned this turn.

    This is the only thing any answer is allowed to be true about, so it is worth
    having ONE definition of it rather than one per reader.
    """
    return "\n\n".join(str(m.content) for m in messages
                        if m.__class__.__name__ == "ToolMessage")


def first_question(messages):
    """What the person originally asked, in their own words.

    The FIRST human message, not the last: a `revise` note from the review panel is
    also a HumanMessage, and steering a draft is not the question being answered.
    """
    for message in messages:
        if message.__class__.__name__ == "HumanMessage":
            return str(message.content)
    return ""


def strip_scaffolding(lines, evidence):
    """Drop opening and closing lines that name no film.

    WHY THIS IS CODE AND NOT A PROMPT RULE
        "Do not write an opening sentence" was stated three ways in three rewrites and
        ignored every time. A small model reaches for a framing sentence the way anyone
        does. Asking a fourth time was not going to work, and each attempt made the
        prompt longer and displaced a rule that WAS holding.

        So it is enforced the same way the critic is: the model writes, the code edits.
        A line naming no film is scaffolding — "here are some films for you" at the top,
        "both of these have a comedic twist" at the bottom. The second is the more
        dangerous one: a summary is a claim no tool made, and it is the line most likely
        to be the only untrue thing in the answer.

    THE GUARD
        A refusal names no film at all and is a correct, complete answer. If stripping
        would empty the reply, nothing is stripped.
    """
    titles = set(handoff.films_in_evidence(evidence).values())
    if not titles:
        return lines

    def names_a_film(line):
        return any(title.lower() in line.lower() for title in titles)

    kept = [i for i, line in enumerate(lines) if names_a_film(line)]
    if not kept:
        return lines                       # a refusal, or nothing recognisable. Leave it.

    # LEADING scaffolding only. Trimming the tail as well looked symmetrical and was
    # wrong: the answer format is a title line followed by the sentence saying why it
    # fits, so the LAST line naming a film is a title and everything after it — the
    # reason for the final recommendation — was deleted as a closing summary. Seen in
    # the browser on 7 Sep: every answer came back as a bare "Predator (1987)".
    #
    # A stray closing sentence is a much smaller problem than a deleted reason, and
    # the critic exists to catch an unsupported summary if it ever runs.
    trimmed = lines[kept[0]:]

    # And drop list markers. The tool's own output is a numbered list, and the model
    # copies whatever shape it is shown — the same reason a worked example in a prompt
    # gets filled in rather than imitated.
    return [LIST_MARKER.sub("", line) for line in trimmed]


def plain_answer(verdict):
    """The record as sentences, written by CODE. Flat, and true.

    The writer is the only part of this application that can go missing without the
    answer becoming WRONG — by the time it runs, every decision has been made and
    checked. So a writer outage costs style, not availability, and this is what that
    costs: the right films, the right reasons, no craft.
    """
    return "\n".join(f"{film['title']} ({film['year']}) — {'; '.join(film['why'])}."
                      for film in verdict["films"])


def write(state: MessagesState) -> dict:
    """NODE — turn the reasoner's verdict into the reply. Decides nothing.

    THE HANDOFF IS A RECORD, NOT A PARAGRAPH
        The reasoner ends its work with a small JSON object naming films and the
        phrases that justify them. This node validates that object against what the
        tools actually returned, then hands the WRITER only the surviving record and
        the person's question — no premise, no scores, no history, no tool output.

        That is the whole point. The writer cannot smuggle in a fact it was never
        given, because it was never given any. Two models passing prose to each other
        is how an invented detail gets in: the second reads a sentence, likes the
        shape of it, and finishes the thought. Structure closes that gap in a way no
        instruction can, because an instruction is a request.

    THREE WAYS OUT, AND ONLY ONE OF THEM CALLS A MODEL
        no evidence   nobody searched — a greeting, or the reasoner asking for the
                      number it was told to ask for. Its own words stand.
        no films      a refusal, written by code. There is nothing to phrase.
        films         the writer runs.
    """
    message = state["messages"][-1]
    draft = split_content(message)[0]
    evidence = tool_evidence(state["messages"])

    if not evidence.strip():
        return {}                       # nothing retrieved; nothing to hand over

    payload = handoff.extract(draft)
    if payload is None:
        # DEGRADE, DO NOT FAIL. A malformed verdict is the reasoner's mistake and the
        # person waiting did not make it. Its own words are a worse answer than the
        # writer would have produced, and an answer.
        print("  [no verdict in the reasoner's reply — passing its own words through]")
        return {}

    verdict, problems = handoff.validate(payload, evidence)
    for problem in problems:
        print(f"  [verdict] {problem}")

    if verdict["nothing_found"]:
        return {"messages": [AIMessage(content=NOTHING_FOUND, id=message.id)]}

    print(f"  [verdict] {handoff.summarise(verdict)}")
    try:
        reply = writer.invoke([HumanMessage(WRITER_PROMPT.format(
            question=first_question(state["messages"]),
            record=handoff.for_writer(verdict)))])
        prose = split_content(reply)[0].strip()
    except Exception as error:
        print(f"  [writer unavailable: {type(error).__name__}]")
        prose = ""

    if not prose:
        prose = plain_answer(verdict)

    # The stripper still runs. It lived on the reasoner's prose until the reasoner
    # stopped writing prose; the habit it catches — an opening line, a list marker
    # copied from whatever the model was shown — belongs to whoever is writing.
    lines = [line for line in prose.splitlines() if line.strip()]
    cleaned = strip_scaffolding(lines, evidence) if lines else lines
    if cleaned != lines:
        print(f"  [tidied {len(lines) - len(cleaned)} scaffolding line(s); "
              f"list markers removed]")
    if not cleaned:
        cleaned = plain_answer(verdict).splitlines()

    return {"messages": [AIMessage(content="\n".join(cleaned), id=message.id)]}


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

    evidence = tool_evidence(state["messages"])
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

    A reply carrying tool_calls is the reasoner ASKING for something; loop round and
    run it. A reply without tool_calls is the reasoner having DECIDED — and a decision
    is not yet an answer, so it goes to the writer next.
    """
    if getattr(state["messages"][-1], "tool_calls", None):
        return "act"
    return "write"


def after_write(state: MessagesState) -> str:
    """Where a finished reply goes. The critic if it is switched on, the human if not."""
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
builder.add_node("act", ToolNode(TOOLS))     # runs whatever the reasoner asked for
builder.add_node("write", write)
builder.add_node("critic", critic, destinations=("review",))
builder.add_node("review", review, destinations=("think", END))
builder.add_edge(START, "think")
# Every value a router can return must appear in its own map. LangGraph rejects a
# destination that was never declared — which is how switching the critic off once
# broke the graph.
builder.add_conditional_edges("think", should_continue,
                              {"act": "act", "write": "write"})
builder.add_conditional_edges("write", after_write,
                              {"critic": "critic", "review": "review"})
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
        "recursion_limit": RECURSION_LIMIT,
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
    # A prompt you can TYPE AT, not a fixed list. The questions worth asking are the
    # ones that occur to you while reading the last answer, and a hardcoded list can
    # never contain those. Pass questions as arguments to run a fixed set instead.
    import sys

    print(f"MovieMotions · {AGENT_MODEL}")

    if len(sys.argv) > 1:
        for question in sys.argv[1:]:
            print(f"\n\n=== {question}\n")
            print(f"\nANSWER: {run(question)}")
    else:
        print("Ask for a film. Empty line to stop.\n")
        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            print()
            print(f"\nANSWER: {run(question)}\n")

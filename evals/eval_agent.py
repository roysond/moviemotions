"""Score the AGENT, not the retriever. Three metrics, and only one needs a judge.

WHY THIS EXISTS ALONGSIDE eval_variants.py
    eval_variants.py measures RETRIEVAL: given a query, do the right films come back?
    It cannot see the agent at all. Every failure it reports is a failure of search.

    But the agent adds three new ways to be wrong, and none of them are search:
        it can pick the WRONG TOOL
        it can recommend a film the tool never returned  (the hallucination we designed against)
        it can say things the retrieved text does not support

    A system is only measured where it is measured. Until now the loop was unmeasured.

THE THREE METRICS, AND WHY ONLY ONE USES AN LLM
    1. TOOL ACCURACY   exact. Did it call the tool we expected? A tool name is a string;
                       comparing strings needs no judge.
    2. GROUNDING       exact. Does the answer name any catalogue film the tools did NOT
                       return? That is a set-difference, not an opinion.
    3. FAITHFULNESS    judged. "Is every claim supported by the retrieved text?" has no
                       exact test — claims are prose. This one, and only this one, is
                       handed to an LLM.

    Same rule as the search tool: never ask the fuzzy machine a question the exact machine
    can answer. Two of these three metrics are deterministic, free, and cannot drift.

THE JUDGE
    RAGAS drives the faithfulness metric over OpenRouter, reusing the key the reranker
    already uses — no new vendor, no new account. The judge is deliberately NOT the model
    under test: Nova Micro grading Nova Micro measures self-consistency, not correctness.

    ragas 0.4.x hard-imports a module that langchain-community 0.4 removed. Pin it:
        pip install "ragas" "langchain-community<0.4"
    That pin leaves langchain-core 1.x and langgraph 1.x untouched — the agent is unaffected.
    If ragas is missing or broken this script still runs and reports the two exact metrics.

USAGE
    python -m evals.eval_agent
"""

import asyncio
import json
import os
import sys

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.agent import auto_approve, converse, split_content  # noqa: E402
from backend.config import DATABASE_URL  # noqa: E402

load_dotenv()

# The judge must NEVER be the model under test — a model asked to grade its own
# output grades generously. Set RAGAS_JUDGE_MODEL in .env to change it.
#
# CHANGING THE JUDGE VOIDS THE SCORE. Faithfulness is one model's opinion of
# another's answer. A new judge gives a different number on identical answers,
# and that difference measures the judge, not the system. Re-baseline, never
# compare across judges.
JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "openai/gpt-5.6-luna")

# IncompleteOutputException means the judge's reply was CUT OFF before its structured
# output closed — a token ceiling, not a quality problem. It hit case 4 (the longest
# answer) on two different judges, which is what ruled out "that model is bad".
JUDGE_MAX_TOKENS = 4096

# A judge at default temperature gives a different verdict on identical text. Measured:
# two runs of the same code scored 0.88 and 0.85. Pinning it to 0 removes the judge's
# own variance, so any movement left is the AGENT changing — which is the thing we are
# actually trying to measure.
JUDGE_TEMPERATURE = 0

# ...except it does not, and this is the finding that matters most about this metric.
# Measured: the SAME frozen answers, judged three times at temperature 0, scored
# 0.87 / 0.75 / 0.79, with one case landing on 0.67, 0.33 and 0.50 — exactly 2/3, 1/3
# and 1/2. RAGAS faithfulness first DECOMPOSES an answer into atomic claims, then checks
# each against the context, and scores supported/total. The decomposition is itself an
# LLM call, so the DENOMINATOR moves before any judging happens. Temperature cannot fix
# a different question being asked each time.
#
# A single draw from that distribution is not a measurement. So judge every case several
# times and report the spread alongside the mean. Slower and a few tenths of a cent more
# expensive; the alternative is a number that invites conclusions it cannot support.
JUDGE_REPEATS = int(os.environ.get("RAGAS_JUDGE_REPEATS", "3"))
OPENROUTER = "https://openrouter.ai/api/v1"

# `tool` is the tool we expect to be chosen. `expect` lists films that SHOULD be named in
# the answer; an empty list means the honest answer is "I don't have that" and naming any
# film is a failure.
CASES = [
    {"id": 1, "query": "tell me about Predator",
     "tool": "lookup_film",  "expect": ["Predator"]},
    {"id": 2, "query": "I want something where creatures are hunting people, really tense",
     "tool": "search_films", "expect": ["Predator"]},
    {"id": 3, "query": "do you have The Godfather?",
     "tool": "lookup_film",  "expect": []},
    {"id": 4, "query": "a man wrongly imprisoned who never gives up",
     "tool": "search_films", "expect": ["The Shawshank Redemption"]},
    # "documentary" is BOTH a description and a genre word, so two tools are defensible
    # and both reach the same correct refusal. This case was written when only two tools
    # existed; nova-pro chose find_films_by_fact, followed the documented rule, and was
    # marked wrong by a stale expectation. A test that punishes the right answer is a
    # broken test. `tool` accepts a set where the ambiguity is genuine.
    {"id": 5, "query": "do you have a documentary about climate change?",
     "tool": {"search_films", "find_films_by_fact"}, "expect": []},
    {"id": 6, "query": "how long is Titanic?",
     "tool": "lookup_film",  "expect": ["Titanic"]},
    # Nothing above exercises the graph. Added after find_films_by_fact shipped with
    # ZERO test coverage — the tool that answers factual questions had never once been
    # required by an eval. Deterministic: an edge either exists or it does not.
    {"id": 7, "query": "anything by Christopher Nolan?",
     "tool": "find_films_by_fact",
     "expect": ["Inception", "The Dark Knight"]},
    # The hybrid shape: a named film AND a description. This is the case whose fix
    # shipped untested, and whose exclude_title argument was silently dropped for
    # an hour without any eval noticing.
    {"id": 8, "query": "something like Jurassic Park but more intense",
     "tool": "search_films", "expect": ["Predator"]},
]


def catalogue_titles():
    with psycopg.connect(DATABASE_URL) as conn:
        return [r[0] for r in conn.execute("SELECT title FROM movies").fetchall()]


def walk(messages):
    """Pull (tool names, tool outputs, final answer) out of a transcript."""
    tools, contexts = [], []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tools.append(call["name"])
        if message.__class__.__name__ == "ToolMessage":
            contexts.append(str(message.content))
    return tools, contexts, split_content(messages[-1])[0]


def mentioned(answer, titles):
    """Catalogue titles named in the answer. Longest first so 'Alien' cannot shadow a
    longer title that contains it."""
    found, remaining = [], answer
    for title in sorted(titles, key=len, reverse=True):
        if title.lower() in remaining.lower():
            found.append(title)
            remaining = remaining.replace(title, " ")
    return found


async def faithfulness_scores(rows):
    """RAGAS faithfulness over OpenRouter.

    Returns (scores, failed) where scores is {case_id: score} and failed is a list of
    (case_id, reason). A judge that errors must NEVER just vanish: dropping it shrinks
    the denominator and quietly RAISES the mean, so a broken judge looks like a better
    system. Failures are counted and reported.
    """
    try:
        from openai import AsyncOpenAI
        from ragas.llms.base import llm_factory
        from ragas.metrics.collections import Faithfulness
    except Exception as error:
        print(f"\n  [faithfulness skipped — {type(error).__name__}: {error}]")
        print('  [fix: pip install "ragas" "langchain-community<0.4"]')
        return {}, []

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("\n  [faithfulness skipped — OPENROUTER_API_KEY not set]")
        return {}, []

    client = AsyncOpenAI(base_url=OPENROUTER, api_key=key)
    metric = Faithfulness(llm=llm_factory(
        JUDGE_MODEL, provider="openai", client=client,
        temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS))
    print(f"\njudging faithfulness with {JUDGE_MODEL} ...")

    scores, failed = {}, []
    for row in rows:
        if not row["contexts"] or not row["answer"]:
            continue
        # Faithfulness asks "is every claim supported by the retrieved text?" A REFUSAL
        # claims an absence — "I don't have that" — and an absence cannot be supported by
        # a list of what IS there. The metric is structurally unable to score it, so a
        # correct refusal reads as 0.00. Excluding these is not hiding a failure; it is
        # refusing to apply a metric outside its domain.
        if row["is_refusal"]:
            continue
        draws = []
        for attempt in range(JUDGE_REPEATS):
            try:
                result = await metric.ascore(user_input=row["query"],
                                             response=row["answer"],
                                             retrieved_contexts=row["contexts"])
                draws.append(float(getattr(result, "value", result)))
            except Exception as error:
                reason = f"{type(error).__name__}: {error}".strip().rstrip(":").strip()
                print(f"  case {row['id']} draw {attempt + 1}: judge failed — {reason}")
        if draws:
            scores[row["id"]] = draws
        else:
            failed.append((row["id"], "every draw failed"))
    return scores, failed


TRANSCRIPT = "data/agent_transcript.json"


def main():
    # --rejudge: judge the SAVED answers again instead of asking the agent for new ones.
    #
    # Two things move between ordinary runs — the agent writes a different answer, and
    # the judge forms a different opinion. A score that changes tells you nothing while
    # both are free to move. Freezing the answers holds one still, so whatever variance
    # is left belongs to the judge alone.
    if "--rejudge" in sys.argv:
        if not os.path.exists(TRANSCRIPT):
            print(f"no {TRANSCRIPT} — run the eval once normally first.")
            return
        with open(TRANSCRIPT) as handle:
            rows = json.load(handle)
        print(f"RE-JUDGING {len(rows)} frozen answers from {TRANSCRIPT}")
        print("the agent was NOT run; the text being judged is identical to last time.\n")
        report(rows)
        return

    titles = catalogue_titles()
    print(f"catalogue: {len(titles)} films · {len(CASES)} agent cases\n")

    rows = []
    for case in CASES:
        # auto_approve: the human-in-the-loop is bypassed on purpose. An eval measures the
        # MACHINE's output; a human editing answers mid-run would score the human.
        messages = converse(case["query"], show_trace=False, decide=auto_approve)
        tools, contexts, answer = walk(messages)
        named = mentioned(answer, titles)
        retrieved = mentioned(" ".join(contexts), titles)
        rows.append({
            "id": case["id"], "query": case["query"], "answer": answer,
            "contexts": contexts,
            # `tool` is a string, or a set when more than one route is genuinely right.
            "tool_ok": bool(tools) and tools[0] in (
                case["tool"] if isinstance(case["tool"], set) else {case["tool"]}),
            "tool_called": tools[0] if tools else "(none)",
            "tool_expected": (" or ".join(sorted(case["tool"]))
                              if isinstance(case["tool"], set) else case["tool"]),
            # every film named must have been returned by a tool in THIS conversation
            "ungrounded": [t for t in named if t not in retrieved],
            "missing": [t for t in case["expect"] if t not in named],
            "named": named,
            # a no-answer case: the right reply names no film at all
            "is_refusal": not case["expect"],
        })
        mark = "ok " if rows[-1]["tool_ok"] and not rows[-1]["ungrounded"] else "FAIL"
        print(f"  {mark} [{case['id']}] {case['query'][:52]:52} -> {rows[-1]['tool_called']}")

    # Freeze the exact text that was judged, contexts included, so --rejudge can
    # replay it. Written BEFORE judging so a judge crash cannot lose the transcript.
    with open(TRANSCRIPT, "w") as handle:
        json.dump(rows, handle, indent=2)

    report(rows)


def report(rows):
    judged, judge_failures = asyncio.run(faithfulness_scores(rows))

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    tool_ok = sum(r["tool_ok"] for r in rows)
    grounded = sum(not r["ungrounded"] for r in rows)
    complete = sum(not r["missing"] for r in rows)
    print(f"  tool accuracy   {tool_ok}/{len(rows)}   exact — right tool chosen")
    print(f"  grounding       {grounded}/{len(rows)}   exact — named no film the tools did not return")
    print(f"  expected films  {complete}/{len(rows)}   exact — named the film we wanted")
    # judged maps case_id -> LIST of draws. Average per case, then across cases.
    per_case = {cid: sum(d) / len(d) for cid, d in judged.items()}
    if judged:
        mean = sum(per_case.values()) / len(per_case)
        spread = max(max(d) - min(d) for d in judged.values())
        skipped = sum(r["is_refusal"] for r in rows)
        flag = "  ** INCOMPLETE **" if judge_failures else ""
        print(f"  faithfulness    {mean:.2f}     judged by {JUDGE_MODEL} over "
              f"{len(judged)} cases ({skipped} refusals excluded — see docstring){flag}")
        print(f"                  {JUDGE_REPEATS} draws per case · widest disagreement "
              f"on one case: {spread:.2f}")
        if spread >= 0.10:
            print("                  a change smaller than that is NOISE, not a result.")
        for case_id, reason in judge_failures:
            print(f"                  !! case {case_id} NOT judged — {reason}")
        if judge_failures:
            print("                  !! the mean above is over FEWER cases and is NOT")
            print("                     comparable to a run where every case was judged.")

    print("\n" + "=" * 78)
    print("WHERE IT WENT WRONG")
    print("=" * 78)
    clean = True
    for row in rows:
        problems = []
        if not row["tool_ok"]:
            problems.append(f"called {row['tool_called']}, expected {row['tool_expected']}")
        if row["ungrounded"]:
            problems.append(f"UNGROUNDED: named {row['ungrounded']} — not returned by any tool")
        if row["missing"]:
            problems.append(f"did not name {row['missing']}")
        if row["id"] in per_case and per_case[row["id"]] < 0.8:
            draws = judged[row["id"]]
            spread_note = (f" (draws: {', '.join(f'{d:.2f}' for d in draws)})"
                           if len(draws) > 1 and max(draws) - min(draws) >= 0.10 else "")
            problems.append(f"faithfulness {per_case[row['id']]:.2f}{spread_note}")
        if problems:
            clean = False
            print(f"\n  [{row['id']}] {row['query']}")
            for problem in problems:
                print(f"       - {problem}")
            print(f"       answer: {row['answer'][:150]}")
    if clean:
        print("\n  nothing — every case passed every metric.")

    with open("data/agent_eval.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "contexts"} for r in rows],
                  f, indent=2)
    print("\nsaved data/agent_eval.json")


if __name__ == "__main__":
    main()

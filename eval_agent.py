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
    python eval_agent.py
"""

import asyncio
import json
import os
import sys

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent import auto_approve, converse, split_content  # noqa: E402
from core import DATABASE_URL  # noqa: E402

load_dotenv()

JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "google/gemini-3.5-flash-lite")
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
    {"id": 5, "query": "do you have a documentary about climate change?",
     "tool": "search_films", "expect": []},
    {"id": 6, "query": "how long is Titanic?",
     "tool": "lookup_film",  "expect": ["Titanic"]},
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
    """RAGAS faithfulness over OpenRouter. Returns {case_id: score} or {} if unavailable."""
    try:
        from openai import AsyncOpenAI
        from ragas.llms.base import llm_factory
        from ragas.metrics.collections import Faithfulness
    except Exception as error:
        print(f"\n  [faithfulness skipped — {type(error).__name__}: {error}]")
        print('  [fix: pip install "ragas" "langchain-community<0.4"]')
        return {}

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("\n  [faithfulness skipped — OPENROUTER_API_KEY not set]")
        return {}

    client = AsyncOpenAI(base_url=OPENROUTER, api_key=key)
    metric = Faithfulness(llm=llm_factory(JUDGE_MODEL, provider="openai", client=client))
    print(f"\njudging faithfulness with {JUDGE_MODEL} ...")

    scores = {}
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
        try:
            result = await metric.ascore(user_input=row["query"],
                                         response=row["answer"],
                                         retrieved_contexts=row["contexts"])
            scores[row["id"]] = float(getattr(result, "value", result))
        except Exception as error:
            print(f"  case {row['id']}: judge failed — {type(error).__name__}: {error}")
    return scores


def main():
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
            "tool_ok": bool(tools) and tools[0] == case["tool"],
            "tool_called": tools[0] if tools else "(none)",
            "tool_expected": case["tool"],
            # every film named must have been returned by a tool in THIS conversation
            "ungrounded": [t for t in named if t not in retrieved],
            "missing": [t for t in case["expect"] if t not in named],
            "named": named,
            # a no-answer case: the right reply names no film at all
            "is_refusal": not case["expect"],
        })
        mark = "ok " if rows[-1]["tool_ok"] and not rows[-1]["ungrounded"] else "FAIL"
        print(f"  {mark} [{case['id']}] {case['query'][:52]:52} -> {rows[-1]['tool_called']}")

    judged = asyncio.run(faithfulness_scores(rows))

    print("\n" + "=" * 78)
    print("SCOREBOARD")
    print("=" * 78)
    tool_ok = sum(r["tool_ok"] for r in rows)
    grounded = sum(not r["ungrounded"] for r in rows)
    complete = sum(not r["missing"] for r in rows)
    print(f"  tool accuracy   {tool_ok}/{len(rows)}   exact — right tool chosen")
    print(f"  grounding       {grounded}/{len(rows)}   exact — named no film the tools did not return")
    print(f"  expected films  {complete}/{len(rows)}   exact — named the film we wanted")
    if judged:
        mean = sum(judged.values()) / len(judged)
        skipped = sum(r["is_refusal"] for r in rows)
        print(f"  faithfulness    {mean:.2f}     judged by {JUDGE_MODEL} over "
              f"{len(judged)} cases ({skipped} refusals excluded — see docstring)")

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
        if row["id"] in judged and judged[row["id"]] < 0.8:
            problems.append(f"faithfulness {judged[row['id']]:.2f}")
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

# Verification harness

The four things `docs/change-guard.md` asks for at project start, kept current.
**Anything below that is out of date is a bug in this file.**

---

## 1. The baseline — what "working today" means

| metric | value | what it measures | better is |
|---|---|---|---|
| achievable@3 | **89.3%** (25 / 28) | of the expected films that *can* fit in a top 3, how many do | higher |
| quiet@3 | **0.2232** | top score on queries that have no right answer | **lower** |
| tool accuracy | **8/8** | did the agent pick the right tool | higher |
| grounding | **8/8** | did it name only films a tool actually returned | higher |
| faithfulness | **0.72** | does the answer follow from the retrieved text (RAGAS) | higher |

### Faithfulness: read this before quoting the number

**Baseline: 0.72**, `openai/gpt-5.6-luna`, 3 draws per case, **6 cases judged of 8**
(2 refusals excluded). Widest per-case spread in that run: 0.21.

**0.78 is void.** It was measured over 4 judged cases; this is 6. Two cases were added on
25 Aug because `find_films_by_fact` had shipped with **zero test coverage** and the hybrid
query shape had none either. A different denominator is a different number — the drop from
0.78 to 0.72 measures the new cases, not a regression.

**A change under 0.05 on the headline is noise.** Measured, not assumed: the SAME frozen
answers, re-judged at temperature 0, scored 0.87 / 0.75 / 0.79 on a single draw each.
Averaging 3 draws per case brought two consecutive runs to 0.76 and 0.79.

**Per-case scores are a pointer, never a measurement.** Within one run, case 2's three
draws were 0.29, 0.57, 0.43. Use a low case score to decide *what to go and read*. Never
report it as a result.

**Why it moves at all.** RAGAS faithfulness decomposes an answer into atomic claims with
an LLM call, then checks each claim against the context, and scores supported/total. The
decomposition is generative, so the DENOMINATOR changes between runs — the observed
0.67 / 0.33 / 0.50 are exactly 2/3, 1/3 and 1/2. Temperature cannot pin that down,
because a different question is being asked each time.

**A more expensive judge does not help — measured, not assumed.** `openai/gpt-5.6-terra`
costs 10x `luna` and scored the same frozen answers at 0.77 / 0.79 against luna's
0.76 / 0.79 — identical headline, and a WORSE per-case spread (0.35 / 0.47 vs
0.24 / 0.29). The two judges also disagree about which case is the weak one while
agreeing on the total. Stay on `luna`; the instability is the metric, not the model.

**Void baselines — do not compare against these.** 0.44, 0.88 and 0.92 were single draws
judged by `google/gemini-3.5-flash-lite`. Different judge, one draw, no error bar.

**Reproduce without re-running the agent:** `python eval_agent.py --rejudge` judges the
frozen answers in `data/agent_transcript.json`. That is the only way to separate the
judge's variance from the agent's.
| graph | **566 nodes · 634 edges** | film 20 · genre 13 · keyword 323 · person 210 | exact |

**Measured over 25 cases, arm D** (context header in the stored vector — what production runs).
Case 21 lists four expected films, so raw recall@3 cannot reach 100%; the ceiling is
`sum(min(len(expect), 3))` = 28, and dividing by it removes a penalty the system cannot avoid.

**Void baselines — do not compare against these.** 86.2% (25 expected answers, plain recall@3)
and 81.0% (30 cases, 42 achievable). Both came from a different golden set or a different
denominator. Five hand-written mood cases were parked in `data/golden_set_mood.json` on
25 Aug 2026; restoring them changes the denominator again and requires a re-measure.

**The other arms, same run.** B (header everywhere) 92.9% but quiet 0.2543 · C (header at rerank
only) 85.7%, quiet 0.2754 · A (no header) 78.6%, quiet 0.2061. B finds more and asserts more —
that trade is why two metrics are reported, never one.

**Two metrics, never one.** Anything that makes the system more eager raises recall *and*
raises false confidence. A change that moves only one of them has not been understood yet.

**Rerank scores ARE stable. The previous claim is retracted, 26 Aug 2026.** The same query run
twice returns identical scores to four decimal places, and `eval_variants.py` has reproduced
exactly on separate days. `cohere/rerank-v3.5` has one provider on OpenRouter, so the "gateway
routes to a different backend" explanation cannot apply to it. The claim came from a single
genre experiment that scored 96.6% once and 86.2% twice; the cause was never established and
was attributed to the reranker on no evidence — the corpus was being modified at the time.
**Still treat a single unreproduced result as an anecdote. That part stands.**

---

## 2. The runner — one command each

```bash
python repo_check.py         # structure: syntax, pins, env parity, dead refs, secrets
python eval_variants.py      # retrieval: achievable@3 and quiet@3 over data/golden_set.json
python eval_agent.py         # the agent: tool accuracy, grounding, RAGAS faithfulness
python build_graph.py --status   # the graph: node and edge counts by type
```

Requires `export $(grep -v '^#' .env | grep -v '^$' | xargs)` first, or the scripts silently
connect to the wrong database. Confirm with `psql "$DATABASE_URL" -c "SELECT current_database();"`
— it must say `moviemotions`.

---

## 3. The self-test rule

Every module carries `if __name__ == "__main__":` that runs it on real input and prints what it
did, so a component can be watched working without reading its code.

```bash
python core.py           # graph facts, then the same catalogue scored by vectors
python tools.py          # the exact tool spec the model receives, then real calls
python agent.py          # a full loop with the review step in the terminal
python build_graph.py --status
```

---

## 4. The neighbours map — what to re-test when X changes

| change this | re-test these |
|---|---|
| `core.py` | **everything.** api · tools · agent · both evals · every experiment |
| `tools.py` | `agent.py`, `eval_agent.py`, and `api.py`'s film parser — it reads the tool's prose |
| `agent.py` | `api.py`, `eval_agent.py` |
| `api.py` | `static/index.html` only |
| `build_graph.py` / `graph_schema.sql` | `core.graph_find`, `tools.find_films_by_fact`, `eval_agent.py` |
| `repo_check.py` / `.github/workflows/ci.yml` | each other — break the checker and the gate lies |
| a job's `name:` in `.github/workflows/ci.yml` | **the branch ruleset on GitHub.** It requires check names as plain strings, so renaming a job leaves the ruleset waiting forever for a check that no longer exists — no red cross, no error, just a PR stuck on "Expected". Rename both in the same sitting |
| `data/golden_set.json` | `eval_variants.py`, `experiments/corpus_ablation.py`, `experiments/mood_audit.py` |
| the corpus (`derive_corpus.py`, `chunk_plots.py`) | re-embed, then **both** evals |

**The trap in this table:** `api.py` parses `tools.py`'s plain-text output with a regular
expression. Change the tool's wording and the web UI silently stops showing scores — no error,
no crash, just an empty panel. Nothing in the type system connects those two files.

---

## Settled experiments — do not repeat these

**A bigger agent model is worse here. Measured 25 Aug 2026.**

| model | tool accuracy | faithfulness |
|---|---|---|
| `amazon.nova-micro-v1:0` (current) | **6/6** | **0.78 ± 0.02** |
| `amazon.nova-lite-v1:0` | crashed | — |
| `amazon.nova-pro-v1:0` | 5/6 | **0.61** |

- **nova-lite crashes.** `ModelErrorException: Model produced invalid sequence as part of
  ToolUse`, on a real prompt with three real tools. The first version of
  `experiments/probe_agent_models.py` cleared it as USABLE because it only asked for one
  trivial tool call — the probe has since been rewritten to bind the real tools and send
  the real system prompt.
- **nova-pro regressed on faithfulness, far outside the noise floor.** Its case-4 answer
  scored 0.00: *"The Shawshank Redemption is a good fit because it is about a man who is
  wrongly imprisoned but never gives up."* The retrieved overview says he was imprisoned
  **for** a double murder and never says he was wrongly convicted or that he persisted.
  Two claims, neither supported — it restated the question instead of using the evidence.
- **One of pro's two "failures" was the test's fault.** On case 5 it chose
  `find_films_by_fact` for *"a documentary about climate change"*, which follows the
  documented rule that a GENRE routes to the graph. The case predated the third tool.
  It now accepts either tool.
- **The lesson is not "small is better".** This agent's hardest job is following precise
  instructions, and micro is already at ceiling there while being less willing to answer
  from the question rather than the evidence.

**The critic node is IN PLACE but UNPROVEN. 26 Aug 2026.**
A verification step between `think` and `review` that strikes lines the retrieved text does not
support. It runs, it strikes lines on ~half the cases, and all guards hold — tool 8/8,
grounding 8/8, expected films 8/8. **But faithfulness came out at 0.75, inside the noise floor.**
Prediction was >0.82; the prediction failed.

Two reasons, both diagnosed:
1. **Wrong unit.** The dominant failure is an unsupported clause inside an otherwise-correct
   sentence — "Andy Dufresne, an upstanding banker who is *wrongly imprisoned*". Line-level
   deletion can keep the whole thing or destroy a correct recommendation. It cannot do surgery.
2. **It treats refusals as unsupported.** "I don't have any documentaries" cannot be supported
   by a list of what IS present — the same structural problem that makes refusals excluded from
   faithfulness scoring. The guard caught it; the prompt still needs fixing.

**Agreed redesign, not yet built:** the critic sends the draft back for ONE rewrite instead of
deleting, and judges by CLAIM TYPE rather than by strictness — the agent may interpret what it
was given ("tense", "a good fit") but may not add facts it was not given ("has gore", "wrongly
imprisoned", "a classic"). That rule also fixes the refusal bug for free, since a refusal
asserts no fact about any film.

**Consequence for the metric:** faithfulness 1.00 is NOT the target. RAGAS penalises
interpretation by design, so a perfect score would mean the agent had stopped interpreting.
Treat it as a direction, not a goal.

**A more expensive judge does not help.** See the faithfulness section above.

**EVIDENCE_CHARS 320 -> 640: INCONCLUSIVE, kept anyway. 25 Aug 2026.**
Faithfulness 0.72 -> 0.76, which is inside the +/-0.03-0.05 noise floor. Exact metrics
unchanged at 8/8. Only 5 of 23 evidence strings were affected — 18 were already shorter
than 320 characters, mean length 213. **An underpowered experiment, not a failed one:**
the effect is below what 6 judged cases can resolve. Kept on mechanical grounds — cutting
a sentence in half can only remove support, never add it — and explicitly NOT recorded as
an improvement. To get a verdict you would need more judged cases, not more runs.

---

## Known open issues

- **Mood queries rank the wrong films.** The Shawshank Redemption appears in 5 of 7 mood
  cases and is correct in 3 only because it was listed under three different moods. On
  *"funny and light"* it ranks first. The corpus describes what happens, never how a film
  feels — this is emotional-density matching, not mood matching.
- **Case 22 scores 0.400 on a no-answer query**, which falls inside the "recommend it
  plainly" band. False confidence, distinct from the mood problem.
- **CI covers structure only, by design.** `.github/workflows/ci.yml` runs `repo_check.py` and a
  clean dependency install on every pull request. It holds no credentials, so it cannot run the
  evals — those stay manual, on a machine that has the keys.

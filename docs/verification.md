# Verification harness

The four things `docs/change-guard.md` asks for at project start, kept current.
**Anything below that is out of date is a bug in this file.**

---

## 1. The baseline — what "working today" means

| metric | value | what it measures | better is |
|---|---|---|---|
| achievable@3 | **81.0%** (34 / 42) | of the expected films that *can* fit in a top 3, how many do | higher |
| quiet@3 | **0.2232** | top score on queries that have no right answer | **lower** |
| tool accuracy | **6/6** | did the agent pick the right tool | higher |
| grounding | **6/6** | did it name only films a tool actually returned | higher |
| faithfulness | judged | does the answer follow from the retrieved text (RAGAS) | higher |
| graph | **566 nodes · 634 edges** | film 20 · genre 13 · keyword 323 · person 210 | exact |

**Why `achievable@3` and not `recall@3`.** Four golden-set cases list more expected films
than fit in a top 3, so raw recall@3 can never exceed 79.2% no matter how good retrieval is.
The ceiling is `sum(min(len(expect), 3))`, and dividing by it removes a penalty the system
cannot avoid. Recall@3 was 86.2% under the old 25-case set; **that number is void** — it came
from a different denominator and a different golden set.

**Two metrics, never one.** Anything that makes the system more eager raises recall *and*
raises false confidence. A change that moves only one of them has not been understood yet.

**Rerank scores are not stable run to run.** OpenRouter is a gateway and the same model id can
land on a different backend. Trust the *gap* between scores, not the absolute value, and treat
any single unreproduced result as an anecdote.

---

## 2. The runner — one command each

```bash
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
python core.py           # retrieval on a sample query
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
| `build_graph.py` / `graph_schema.sql` | nothing yet — the graph is not wired into retrieval |
| `data/golden_set.json` | `eval_variants.py`, `experiments/corpus_ablation.py`, `experiments/mood_audit.py` |
| the corpus (`derive_corpus.py`, `chunk_plots.py`) | re-embed, then **both** evals |

**The trap in this table:** `api.py` parses `tools.py`'s plain-text output with a regular
expression. Change the tool's wording and the web UI silently stops showing scores — no error,
no crash, just an empty panel. Nothing in the type system connects those two files.

---

## Known open issues

- **Mood queries rank the wrong films.** The Shawshank Redemption appears in 5 of 7 mood
  cases and is correct in 3 only because it was listed under three different moods. On
  *"funny and light"* it ranks first. The corpus describes what happens, never how a film
  feels — this is emotional-density matching, not mood matching.
- **Case 22 scores 0.400 on a no-answer query**, which falls inside the "recommend it
  plainly" band. False confidence, distinct from the mood problem.
- **No CI gate.** There is no `.github/workflows/`, so nothing runs automatically on a PR.

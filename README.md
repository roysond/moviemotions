# MovieMotions

A mood- and situation-driven film recommendation **agent**. You describe how you want to feel or
what you want to watch — *"something tense where creatures hunt people, under two hours"* — and it
decides what to look up, searches a scene-level corpus, judges what came back, and either
recommends something or tells you plainly that nothing fits.

Built from scratch in passes as a working study of retrieval and agent design. It is not a wrapper
around a hosted RAG service: the chunking, retrieval, reranking, agent loop and evaluation harness
are all in this repo.

It then shows **where each film can actually be watched, and what each way costs.**

```
you ──▶ agent (LangGraph) ──▶ picks a tool ──▶ search_films        ──▶ graph + column filters
           ▲                                                         THEN pgvector + rerank
           │                                   lookup_film         ──▶ exact title
           │                                   find_films_by_fact  ──▶ knowledge graph
           │                                   check_availability  ──▶ graph + backend/providers.py
           └────── reads the results, decides again ◀────────────────────┘
                               │
                        human review  ⏸  approve / edit / send back
                               │
                               ▼
                  answer  +  a panel of posters, reasons and prices
```

Three of those four tools never touch a model.

`search_films` filters **before** it ranks. Runtime and year are columns on `movies`; genre,
actor and director are edges in the knowledge graph, checked with an `EXISTS` in the same
query. Whatever survives is what the vector search sees. When nothing survives, the tool works
out which constraint is to blame — length is a convenience and may be given up, a genre is the
request and never is — and offers the trade instead of returning nothing.

---

## What it does that a plain RAG pipeline doesn't

| | |
|---|---|
| **It loops** | The model chooses a tool, reads the result, and can search again with different wording. The shape of a run depends on the query, so it cannot be drawn in advance |
| **Hard constraints are enforced in SQL** | "under two hours" becomes `runtime_minutes <= 120`, not an embedding. Vectors capture topic, not truth value |
| **Facts go to a graph, not a reranker** | "anything by Nolan" is a yes/no question about an edge, so it never touches a vector. Measured: the vector path returns Terminator 2 at 0.465 for that query, and scores Schwarzenegger films highly even though his name appears nowhere in the corpus — the reranker was answering from its own training |
| **It refuses** | A weak top score produces *"I don't have anything like that"* rather than the least-bad option |
| **A human can intervene** | The graph pauses before answering; you can approve, reword, or send it back round the loop with a note |
| **It is measured** | Retrieval and agent behaviour each have an eval harness — exact metrics where an exact test exists, an LLM judge only where none does |

---

## Where the code lives

Every folder holds files with the **same lifecycle**. That is the whole organising rule —
if two files are run at different times, by different people, for different reasons, they
do not belong together.

```
backend/     the application. This is what runs in production.
pipeline/    builds the corpus. Run by hand, never serves a request.
evals/       measures behaviour. Needs credentials, so never runs in CI.
scripts/     dev tooling. Standard library only, no credentials, no database.
frontend/    the React app.
tests/       deterministic, no credentials, runs on every pull request.
experiments/ DISPOSABLE. Delete any of these without a second thought.
```

`experiments/` is the only folder meant to be thrown away. Everything else is permanent;
the folders separate them by *job*, not by importance.

### `backend/` — the application

| file | what it is |
|---|---|
| `backend/config.py` | settings, read once at import so a misconfigured machine fails immediately |
| `backend/models.py` | **the only file that names a vendor.** Every call that leaves this machine to reach a model: Bedrock for embeddings, Cohere for reranking. Swapping either means editing this file and nothing else |
| `backend/retrieval.py` | the **vector** half — the search SQL, reranking, collapsing chunks to films, exact title lookup, and what a hard filter removed |
| `backend/graph.py` | the **exact** half — facts, relationships, and availability. No model reaches this file |
| `backend/providers.py` | the semantic layer: 34 US services with dated prices, sources and a `verified` flag. Owns `REGION` |
| `backend/tools.py` | what the agent is allowed to do. The docstrings *are* the interface — only they travel to the model |
| `backend/agent.py` | the LangGraph loop: think → act → think → review. The conditional edge is the whole difference between a pipeline and an agent |
| `backend/api.py` | HTTP over the *same compiled graph*. No prompts, no tools, no logic. If it and `backend/agent.py` ever disagree, one is a bug |
| `backend/tracing.py` | one optional dependency, isolated, so neither half has to depend on the other to get it |

**The dependency direction never reverses:**

```
config ──▶ models ──▶ retrieval ──┐
   └─────▶ graph ─────────────────┼──▶ tools ──▶ agent ──▶ api
           providers ─────────────┘
```

`backend/retrieval.py` and `backend/graph.py` share a database URL and nothing else — which is exactly
why they are two files and not one. They were a single module — **core**, 786 lines doing five jobs — and every change to
it put the whole repository in the blast radius. Retired on 30 Aug 2026.

### The other folders

| file | what it is |
|---|---|
| `pipeline/fetch_titles.py` · `pipeline/fetch_plots.py` | pull raw data from external sources to disk |
| `pipeline/chunk_plots.py` · `pipeline/derive_corpus.py` | turn raw text into chunks and derived descriptions |
| `pipeline/load_corpus.py` · `pipeline/load_derived.py` | embed and write to Postgres |
| `pipeline/build_graph.py` | derives the knowledge graph from `movies.raw_payload`. Idempotent; `--status` and `--remove` |
| `evals/eval_variants.py` · `evals/eval_agent.py` | the two harnesses. See `docs/verification.md` |
| `scripts/repo_check.py` | the structural checks CI runs. Standard library only |
| `scripts/build_docs.py` | renders the markdown docs to HTML. **The HTML is derived — never edit it** |
| `search.py` | retrieval from the command line, no agent. The one entry point left at the root |
| `experiments/graph_vs_vector.py` | the same factual question sent to both machines, side by side |

**Run everything from the repository root, as a module:**

```bash
python -m backend.tools           # the spec the model receives, then real calls
python -m backend.graph           # facts and availability, no model involved
python -m backend.retrieval       # the vector path, scored
python -m pipeline.build_graph --status
python -m evals.eval_agent
python -m scripts.repo_check
uvicorn backend.api:app --reload --port 8000
```

**One caution.** `backend/api.py` parses `backend/tools.py`'s plain-text output with a
regular expression, so changing the tool's wording can silently break the web display
with no error anywhere.

---

## Requirements

- **Python 3.11+**
- **PostgreSQL 15+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension**
- **An AWS account with Amazon Bedrock access** — Nova for embeddings and text
- **An [OpenRouter](https://openrouter.ai) key** — reranker and eval judge
- *(optional)* A **[LangSmith](https://smith.langchain.com)** key for tracing
- *(optional)* A **[TMDB](https://www.themoviedb.org/settings/api)** key — only to rebuild the catalogue from scratch

---

## Setup

### 1. Clone and create the environment

```bash
git clone <your-fork-url> moviemotions
cd moviemotions
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a full freeze — exact versions including transitive dependencies — so the
environment you get is the one the project was measured on.

> **One pin matters:** `langchain-community` is held below 0.4. RAGAS imports a module that version
> removed. Do not upgrade it casually.

### 2. Configure credentials

```bash
cp .env.example .env
```

Then fill in `.env`. It is gitignored and must stay that way.

| variable | what it is |
|---|---|
| `DATABASE_URL` | `postgresql://user:pass@localhost:5432/moviemotions` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | an IAM user with permission to invoke Bedrock |
| `BEDROCK_MODEL_EMBED_NOVA` | the embedding model id |
| `BEDROCK_MODEL_TEXT` | the text model id — also the agent's model unless `BEDROCK_MODEL_AGENT` is set |
| `OPENROUTER_API_KEY` | reranker and eval judge |
| `RERANK_MODEL` | defaults to `cohere/rerank-v3.5` |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | optional tracing |
| `TMDB_READ_TOKEN` | only needed to rebuild the catalogue |

Verify without printing anything secret:

```bash
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
for k in ['DATABASE_URL','AWS_REGION','BEDROCK_MODEL_EMBED_NOVA','OPENROUTER_API_KEY']:
    v = os.environ.get(k)
    print(f'{k:28}', f'present, {len(v)} chars' if v else 'MISSING')
"
```

### 3. Create the database

```bash
createdb moviemotions
psql moviemotions -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql moviemotions -f schema.sql
```

`schema.sql` is structure only, no data. Five tables — three for retrieval, two for the graph:

```
movies             one row per film     movie_id · title · release_date · runtime_minutes · context_header
chunks             many rows per film   chunk_id · movie_id · source_field · chunk_index · content
chunk_embeddings   one row per vector   embedding_id · chunk_id · model_id · embed_variant · embedding

graph_nodes        one row per thing    node_key · node_type · name · properties
graph_edges        one row per fact     from_key · to_key · edge_type · source · confidence
```

The graph tables live in their own file, applied separately:

```bash
psql moviemotions -f graph_schema.sql
```

A key that joins two tables carries the **same column name in both** (`movie_id`, `chunk_id`), and
each table's own key is named for its grain. The column name then tells you what a row *is*.

### 4. Build the corpus

No film data is committed. TMDB's payloads are licensed to them, Wikipedia's text is CC BY-SA, and
embeddings are derived data that go stale the moment a model changes. **Keep the raw thing, derive
everything else from it.** Run these in order:

```bash
python -m pipeline.fetch_titles      # TMDB → data/raw/tmdb_*.json            (needs TMDB_READ_TOKEN)
python -m pipeline.fetch_plots       # IMDb id → Wikidata → Wikipedia plots   → data/plots.json
python -m pipeline.load_corpus       # films + overview chunks                → Postgres
python -m pipeline.derive_corpus     # a model writes mood/theme text         → data/derived.json
python -m pipeline.load_derived      # derived text → chunks + embeddings
python -m pipeline.chunk_plots       # semantic → recursive → overlap chunking, embeds each chunk
python -m pipeline.build_graph       # films · people · genres · keywords → nodes + edges
```

`pipeline/chunk_plots.py` is resumable: it commits per film and caches vectors by content hash, so a rate
limit costs time, never finished work. Re-run it and it continues.

> **Wikimedia may refuse an automated fetch** under its robot policy. If `pipeline/fetch_plots.py` returns
> 403s the plots have to be gathered another way; everything downstream is unaffected.

---

## Running it

**Web UI** — shows the tool chosen, its exact arguments, every result with its score and quoted
evidence, and the human-review step with buttons:

```bash
uvicorn backend.api:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** for the original page, or **/app** for the two-panel
React interface.

`static/app/` is build output — derived, gitignored, and absent from a fresh clone — so
**/app answers 503 with instructions until you build it**:

```bash
cd frontend && npm install && npm run build     # writes static/app/, then restart uvicorn
```

Restarting matters: the server mounts `/app/assets/` only if that folder exists when it
starts, so a build made while the server is running is not picked up until it reloads.

**The React front end**, in development, with hot reload:

```bash
cd frontend && npm install && npm run dev     # then open http://localhost:5173
```

Vite serves the app on :5173 and forwards `/api` to FastAPI on :8000. `npm run build`
typechecks and writes the bundle into `static/app`; Vite then exits — nothing of it runs
in production.

**Command line**, with the review step in the terminal:

```bash
python -c "from agent import run; print(run('something tense with creatures, under two hours'))"
```

**Retrieval only**, no agent:

```bash
python search.py "a father and son separated and trying to find each other"
```

---

## Evaluation

Two harnesses, measuring two different things.

```bash
python -m evals.eval_variants     # RETRIEVAL: achievable@3 and quiet@3 over a 25-case golden set
python -m evals.eval_agent        # THE AGENT: tool accuracy, grounding, RAGAS faithfulness
python -m pytest tests -q   # THE MATHS: no model, no database, 0.6 seconds
```

**Faithfulness is reported and never gated on.** Two identical runs with no code change
scored 0.78 and 0.72; within one run a single case scored 0.86 / 0.33 / 0.86 on three
draws of the same answer. Treat anything under 0.10 as unresolvable. Gate on the
deterministic; report the rest.

**Reproduce CI's conditions before pushing:**

```bash
env -i HOME="$HOME" PATH="$PATH" .venv/bin/python -m pytest tests -q
```

`env -i` wipes the environment, which is the blank slate a CI runner gets. A normal run
inherits everything you exported from `.env` and will lie to you — that is exactly how a
wrong variable name passed locally and failed on the first clean machine.

`evals/eval_variants.py` cannot see the agent at all. `evals/eval_agent.py` measures the three ways a loop can
be wrong that a retrieval eval structurally cannot detect: the wrong tool, a film no tool returned,
and claims the retrieved text does not support.

**Only one metric uses an LLM.** Tool choice is a string comparison; grounding is a set difference.
Both are deterministic, free, and cannot drift. Faithfulness has no exact test, so — and only it —
goes to a judge, and the judge is deliberately not the model under test.

### Current numbers

| metric | value | meaning |
|---|---|---|
| achievable@3 | **89.3%** (25/28) | of the expected films that *can* fit in a top 3, how many do |
| quiet@3 | **0.2232** | top score on queries with no right answer — **lower is better** |
| tool accuracy | **8/8** | exact |
| grounding | **8/8** | exact — named no film a tool did not return |
| faithfulness | **0.72** | RAGAS via OpenRouter, 3 draws per case, 6 of 8 cases judged. **A change under 0.05 is noise** — see `docs/verification.md` |

Two metrics, never one: anything that makes the system eager raises recall **and** false confidence.

**Why `achievable@3` rather than plain recall@3.** One case in the golden set names four films,
so raw recall@3 cannot reach 100% however good retrieval gets. Dividing by
`sum(min(len(expect), 3))` removes a penalty the system cannot avoid.

**The numbers above are arm D** — the context header stored in the vector, which is what runs in
production. Arm B (header everywhere) scores higher on achievable@3, **92.9%**, and worse on
quiet@3, **0.2543** vs 0.2232. That is the trade in one line: the arm that finds more also
asserts more on questions with no answer. Run `python -m evals.eval_variants` to see all four.

### Experiments

`experiments/` holds the diagnostics that produced those numbers — not dead code:

| file | the question it answers |
|---|---|
| `why_chunk.py` | which chunk won, and did the quota even admit it? |
| `corpus_ablation.py` | what is each corpus worth? (leave-one-out) |
| `db_audit.py` | read-only schema, row counts, integrity checks |
| `genre_corpus.py` | the genre-as-corpus experiment — add, measure, remove |
| `mood_audit.py` | which films dominate mood queries, and are they ever right? |

---

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request. Four jobs, four different questions:

```bash
python -m scripts.repo_check        # the same structural checks CI runs — run it before you push
```

| job | what it asks |
|---|---|
| **Structure** | Does every module parse? Is every third-party import pinned? Does `.env.example` match what the code reads? Does any doc point at a file that doesn't exist? Has anything secret-shaped been committed? Does `.gitignore` protect `.env` without swallowing a schema file? Does every import of our own code point at a module that still exists there? |
| **Dependencies** | Does `requirements.txt` actually install on a clean machine, and does the third-party stack import? |
| **Unit tests** | Is the maths right? The damped sum against hand-computed numbers, price banding, and whether `pipeline/build_graph.py` and `graph_schema.sql` still agree about edge types |
| **Front end builds** | Do `backend/api.py` and the React app still agree about their data? `npm run build` runs `tsc --noEmit` first, so a mismatch fails the pull request rather than the browser |

A fifth workflow, `.github/workflows/staleness.yml`, runs **on a schedule** rather than on a change: it reads
the date stamped in `backend/providers.py` and opens an issue when prices pass 30 days old. Nothing
in a repository changes when Apple raises a price — only the calendar knows.

**CI holds no credentials and never will.** A workflow with your AWS keys is a workflow
that can leak them, and a pull request from a fork could read them.

The old line here said *CI checks structure, the evals check behaviour*. That stopped being
true when `tests/` arrived. The honest version: **CI checks everything that gives the same
answer every time; the evals check the rest**, on a machine that already has the keys.

**A green build is not a working system.** Switching the critic off left a routing function
returning a value the graph had not been told to expect, so every request returned 500 —
and all four checks passed, because the file still parsed and no unit test touches the
graph. Re-run the thing you changed.

Every check exists because it caught something real — the `.gitignore` rule that silently
excluded `graph_schema.sql`, three environment variables the code required and the
template omitted, a doc pointing at a file that no longer existed, and three
`import build_graph` lines left behind by the folder reorganisation. Found by hand once,
then written down so they cannot recur.

`scripts/repo_check.py` uses the standard library only. A gate that needs a dependency install
can be broken *by* a dependency.

---

## Documentation

| file | read it when |
|---|---|
| `docs/PASS-0-DIAGRAM.html` · `PASS-1` · `PASS-2` | **open all three side by side** — same rows, same order; only the boxes change |
| `docs/decisions.md` | before starting any change — one line of reasoning per decision |
| `docs/ARCHITECTURE.html` | when you lose the shape of the system |
| `docs/retrieval-pipeline.md` | the query path, end to end |
| `docs/verifying-code.md` | how to verify a change you cannot read |
| `docs/verification.md` | **start here before changing anything** — the baseline numbers, the one command that reproduces each, and what to re-test when a file changes |
| `docs/change-guard.md` | the change protocol: contract → one variable → prove three ways |
| `docs/third-party.md` | licence and terms for every external source |

---

## Design notes worth knowing before changing anything

**Hard constraints are filtered in SQL, above `ROW_NUMBER()`.** Filtering *after* the per-source
quota shrinks the result set once the budget is already spent; filtering before it means the quota
refills with eligible films.

**Films are collapsed after reranking, never in SQL.** Collapsing early let a 120-character mood
blurb beat its own film's 600-character scene, so the plot corpus never reached the reranker.

**A film's score is its top 3 chunks, damped (1, ½, ⅓).** Max-pooling rewards one lucky scene.

**The context header lives on the film, not in the chunk.** Measured: putting it in the stored
vector is worth +7–10 recall points; adding it at rerank time adds less and costs the most false
confidence.

**Tool descriptions are the interface.** Only the docstring travels to the model — the code never
leaves the machine. Score thresholds, when-not-to-use rules and the query-expansion rule all live
in prose, and the model obeys them.

---

## Known weaknesses

- **Mood queries rank the wrong films.** *"warm, comforting, rainy evening"* returns prison dramas.
  The corpus describes what *happens*, never how a film *feels* — a corpus problem, not a query or
  threshold problem.
- ~~**Rerank scores are not stable run to run.**~~ **Retracted 26 Aug 2026 — this was wrong.**
  Measured: the same query run twice returns identical scores to four decimal places, and
  `evals/eval_variants.py` has reproduced exactly on separate days. `cohere/rerank-v3.5` also has
  exactly one provider on OpenRouter, so there is no backend to route between. The claim came
  from one genre experiment that scored 96.6% once and 86.2% twice; the corpus was being
  changed at the time, which explains it far better than the model did.
- **The graph is thin on people.** Only 3 directors and 7 actors appear in more than one film, so
  "another film by this director" works for three directors. A graph's value scales with shared
  nodes, and 20 films is a small world.
- **Answer quality is at the small-model floor.** The agent model is strong at structured decisions
  and weak at prose. `BEDROCK_MODEL_AGENT` is the seam for swapping it.

---

## Third-party data and licences

Film metadata from **TMDB** (this product is not endorsed or certified by TMDB). Plot summaries
from **Wikipedia**, CC BY-SA 4.0. Embeddings and text generation via **Amazon Bedrock**; reranking
and eval judging via **OpenRouter**. See `docs/third-party.md` before redistributing anything.

# MovieMotions — Decisions and Why

> Not a reference doc. A **memory aid.** One line of reasoning per decision, so that
> looking at the architecture reminds you why it's shaped that way.
>
> Read this before starting a pass. It should take four minutes.

---

## How we build

| Decision | Why |
|---|---|
| **Walking skeleton, then depth** — everything crude and connected before anything is good | You cannot tell whether a layer is right until something downstream consumes it |
| **Passes, not phases** | Revisiting a component three times, spaced apart, beats one long stretch on it |
| **Iterative ≠ incremental** — improving what exists vs adding what doesn't | Two different activities. Naming them separately keeps a pass honest |
| **Every pass gets a written exit condition, before it starts** | Otherwise "improve retrieval" never ends |
| **Only 7 things must be right early** | Everything else is allowed to be embarrassing on the first attempt |

---

## The domain

| Decision | Why |
|---|---|
| **Mood-driven movie recommendation** | Genuinely multi-tool, so agentic behaviour is inherent rather than decorative |
| **Not a trivia corpus** | Streaming and theatre data go stale weekly — real drift, not simulated |
| **Location is part of it** | IP and coordinates are personal data, so privacy becomes real, not theoretical |
| **Scene-level retrieval is a requirement** | "They confront an animal in the woods" must find films where that's *not* the plot |

**Proven, not assumed:** TMDB's synopsis for Jurassic Park is **348 characters**. It mentions no
character, no dinosaur species, no scene. A semantic search over it cannot answer a scene question,
because the information isn't there.

---

## Sources

| Decision | Why |
|---|---|
| **TMDB is primary** | Facts, cast/crew, keywords, availability — rich in structure |
| **Wikipedia plot sections for narrative** (Pass 1) | The only free, legally reusable, long-form, scene-by-scene source at scale (CC BY-SA 4.0) |
| **Join them via Wikidata ID, not title** | Titles collide; identifiers don't. Same problem as matching a patient across two systems |
| **Pass 0 uses TMDB only** | You need to *feel* shallow retrieval fail before enrichment means anything |
| **Keep the top ~10 cast, filter crew to 4 roles** | One film returned 206 people. Noise in a graph costs you at every traversal, forever |

> **TMDB is rich in facts and poor in narrative.** That one sentence explains the whole source strategy.

---

## Storage

| Decision | Why |
|---|---|
| **One Postgres holds vectors, keyword index, and graph** | One database, one backup, one connection. pgvector + a BM25 extension cover it |
| **Schema built from requirements, not from the source's shape** | Sources change; requirements don't. A TMDB-shaped table would break when Wikipedia arrives |
| **Keep `raw_payload` untouched** | Every future extraction — the graph, new fields, a different curation — comes from it. No re-fetching |
| **Our own `id`, plus `source` + `source_id`** | The identifier can't belong to any one provider |
| **`UNIQUE (source, source_id)`** | The database prevents duplicates. Not your code remembering to check |
| **No cast/crew/keyword tables yet** | They're already inside `raw_payload`. That rule paid off on day one |

> **A field earns a column when something needs to filter, sort, join, or look it up.**
> Otherwise it stays in the raw payload. Test in reverse: *if I deleted this column, what breaks?*

---

## Models

| Decision | Why |
|---|---|
| **AWS Bedrock as provider** | The target job description names it first. Free plan allows Amazon's own models |
| **`us-east-1`, not Ohio** | 89 models vs 75, and every tutorial assumes N. Virginia |
| **US inference profile, not Global** | Global may process outside the US. That dropdown *is* data residency |
| **Nova Micro for query parsing** | Measured: identical output to Claude Haiku, **337 ms faster** |
| **Nova 2 Multimodal Embeddings @ 1024 dims** | Benchmarked 3 models on 5 real queries — it ranked best. 1024 sits inside pgvector's 2,000-dim index cap; more dims isn't "safer" (embeddings have no overflow) |
| **Reranker host: OpenRouter (`cohere/rerank-v3.5`)** | Every in-house AWS path was walled — Cohere is a paid Marketplace product the free plan can't subscribe to; Amazon Rerank 1.0 isn't offered in us-east-1 and the account was denied it cross-region. OpenRouter is a provider-agnostic gateway: one HTTPS call, no Marketplace subscription, no per-region model access, no BAA to the vendor. Same Cohere model, ~$0.001/search |
| **The reranker walls were about the *vendor*, not the model** | Every Cohere version (3.5 / v4 Fast / v4 Pro) hits the identical Marketplace wall; first-party vs third-party is a *who-is-allowed-to-see-the-data* line, not a quality one. A rerank call is stateless (query + docs in, order out), so hosting is free to move — but a gateway that hides the vendor is a liability for PHI, not a convenience (bridges 1, 2, 8) |
| **`httpx`, not `urllib`, for the rerank call** | macOS python.org Python ships no trusted CA bundle, so `urllib` failed every HTTPS call with `CERTIFICATE_VERIFY_FAILED`. `httpx` verifies against `certifi`'s bundle automatically → portable, no system-wide cert install to depend on |
| **`converse` API, not `invoke_model`** | Same request shape for every model, so swapping models is a config change |
| **`temperature: 0` for parsing** | Translation, not creativity. Same input should give the same output |
| **Explicit prompt over a stronger model's inference** | Nova Pro *guessed* "Denzel Washington" correctly. Nova Micro was *told*. **Specified beats emergent** |
| **An `unresolved` field in the output** | Give the model a way to say "I don't know", or it will guess silently |

**Measured, not assumed:** the same prompt costs 184 / 192 / 222 input tokens on Nova / DeepSeek /
Claude — different tokenizers. **You can't compare model cost by token count. Only by cost per task.**

---

## Retrieval — the designed target, and what is actually built

**Target (full design):**
```
parse query  →  hard filter (metadata + graph)  →  vector + BM25 in parallel
             →  reciprocal rank fusion  →  rerank ~50  →  top 3
```

**Built as of 20 Aug — recall@3 7/7.** Still semantic-only; no parse, no filter, no BM25/RRF yet:
```
embed  →  stratified candidates (plot 30 · derived 10 · overview 10)
       →  rerank (cross-encoder, reads query+chunk together)
       →  aggregate per film (best + 2nd/2 + 3rd/3)
       →  collapse to films  →  top 3
```

| Decision | Why |
|---|---|
| **Parse the query into structured filters first** | A database can't act on a sentence. It needs names and numbers |
| **Only the mood gets embedded** | "Nothing over two hours" in a vector adds noise. Parsing *improves* search, not just filtering |
| **The graph is a filter, not a ranker** | Graph facts are binary. There's nothing to rank |
| **Fuse on rank, never on score thresholds** | A cosine of 0.78 means different things per query, and BM25 scores aren't even on the same scale |
| **Small-to-big: chunk at scene level, return the parent film** | Small chunks match precisely; users want the film, not the paragraph |
| **Retrieve wide (10), rerank to narrow (3)** | The right answer may sit at rank 4–8 by vector; a top-3 retrieval never sees it. The wide net gives the reranker something to rescue |
| **External model calls degrade, never crash** | `search()` wraps the rerank call in try/except and falls back to vector order. A vendor outage costs quality, not availability |
| **Filters output sets; retrievers output lists; RRF fuses lists** | Which is *why* a graph (a filter → a set) never enters RRF. Rank-fusion needs two lists to exist |
| **Hard constraints deterministic, soft preferences semantic** | The most important architectural line in the build. The model must never invent a showtime |
| **Collapse to one row per film AFTER reranking, never in SQL** | `DISTINCT ON (m.id)` let a 120-char mood blurb beat its own film's 600-char scene on cosine, so the plot corpus never reached the reranker. An optimisation is only safe if it discards what the downstream component wouldn't have wanted |
| **Stratified candidates — a quota per source type** | Chunk length is a confound: short abstract chunks win on *concentration*, not relevance, and a global top-N hands the whole pool to the shortest type. Rank each type against its own kind |
| **Score a film by its top 3 chunks, damped (1, ½, ⅓)** | Max-pooling rewards one lucky scene — Finding Nemo's single barracuda outranked Predator. Breadth of evidence, with the best chunk still dominating |
| **Header lives on the film, not in the chunk — arm D** | Measured on a 25-query golden set: header in the STORED VECTOR is worth +7–10 recall points; adding it at RERANK time adds less (+3.5) and raises false confidence most. So it is stored ONCE in `movies.context_header` (was 145 copies, 31% of the corpus) and baked into the index only. B looked better by 3.5 points, but 1 answer = 3.4 points and the whole gap was one lucky hit on a query both arms failed |
| **`embed_variant` on every vector** | An embedding is identified by *chunk × model × what text was embedded*. That third axis is what let the header question be settled by measurement, and it is the same slot a challenger model will use |
| **Two metrics, never one — recall@3 AND quiet@3** | They pull opposite ways: anything that makes the system eager raises recall *and* false confidence. `quiet@3` = top score on queries with no right answer, lower is better. A single number lets you buy recall by shouting about everything |
| **Convention B key naming** | A key that joins two tables has the SAME name in both (`movie_id`, `chunk_id`), and each table's own key is named for its grain (`movies.movie_id`, not `movies.id`). The column name then carries the grain — the thing the 40-row bug hid |
| **(superseded) Every chunk carries a film-level context header** | Chunking preserves events and destroys relationships: *"Nemo swims to a speedboat and is captured"* has no father, no son, no search. Built from title + overview, so zero extra model calls. Took recall 4/7 → 7/7 |
| **Chunking = semantic → recursive → overlap, in that order** | Semantic cuts where meaning changes; recursion enforces the context cap by cutting at each segment's *own* biggest shift; overlap stitches **only** size-forced seams — at a semantic boundary the meaning genuinely changed, so overlap there would blur both sides |
| **Percentile break threshold, not a fixed one** | Nova's space is narrow (unrelated text ≈ 0.64), so "split below 0.8" means something different per document and dies on a model swap. Score each plot against itself |
| **Hard constraints go in the SQL `WHERE`, above `ROW_NUMBER()`** | Vectors capture topic, not truth value — "under 2 hours" embeds as mood and "not a cartoon" embeds *next to* "a cartoon". Measured: 3 of 5 results broke a limit the user stated out loud. And the filter must run BEFORE the per-type quota, or the quota is spent on films about to be thrown away |
| **The LLM extracts, the database enforces** | Extraction is fuzzy and unenumerable ("I've only got 90 minutes", "nothing epic") so it needs a model. Checking is exact, so it needs a column. The model never gets to *decide* whether a film is under two hours — only that one was asked for |
| **Genre is a filter, not a corpus** | Royson's call, and he was right. A label from a closed list has a yes/no test, so it belongs in a `WHERE` clause. Built as a corpus and measured: worth **±0 answers** on leave-one-out. Deleted |
| **The matched text travels back with the result** | It used to be discarded — the tool said *which* film matched and *where*, never the words. So every reason the model gave came out of its training, not the database. RAGAS faithfulness caught it at 0.44 while every exact metric said 6/6. **Retrieval you don't pass on is retrieval you didn't do** |
| **`sources=[...]` is an instrument, not a feature** | Leave-one-out ablation. "Adding X helped" says X was useful once; **"removing X hurts" says X is earning its place today.** Only the second answers "why is this still in the pipeline?" |
| **Throttling is a pause, not a crash** | Self-pacing delay that grows on push-back and decays on success, plus per-film commits and a content-keyed vector cache. Rate limits cost time, never finished work |

---

## The agent loop — Pass 2

**Built as of 24 Aug.** The Reason step stopped being one LLM call:
```
START → think ──has tool_calls?──→ act ──┐
          ↑                              │
          └──────────────────────────────┘
          │ no
          ▼
        review  ⏸ human: approve / edit / revise ──revise──→ think
          │
          ▼
         END
```

| Decision | Why |
|---|---|
| **Hand-wire the graph, don't call `create_react_agent`** | The prebuilt builds exactly this graph. Wiring `StateGraph` by hand is the difference between "I've used ReAct" and "I can draw one and tell you which edge makes it an agent" |
| **Natural termination is the design; the recursion limit is a backstop** | The loop is supposed to end because the model answers instead of calling a tool. Regularly hitting the step ceiling is a **bug signal** — a vague tool description, or two tools with a blurry boundary — not a config to raise |
| **A second tool must have a DISJOINT purpose** | `parse_query` was rejected for sharing the user's sentence with `search_films`: no crisp rule for when to call which, so the model oscillates. `lookup_film` (exact title) versus `search_films` (mood) has one rule and no overlap. Nova Micro then scored 6/6 on selection |
| **The docstring IS the interface** | Only the description travels to the API; the code never leaves the machine. Score bands, when-NOT-to-use, and the expansion rule all live in prose — and the model obeys them: it declined Toy Story at 0.104 with no `if score < 0.25` anywhere in the codebase |
| **Examples in a docstring override the prose** | My own example *demonstrated compressing* a query to keywords while the prose asked for expansion. The model copied the example. `"cosy"` → top score 0.08; `"a warm gentle feel-good film for a rainy evening"` → 0.37. Same corpus, 4× the signal |
| **Query expansion comes free in an agentic design** | Pipeline RAG bolts on an extra LLM call (HyDE, multi-query) to rewrite thin queries. Here, choosing the `query` argument *is* the rewrite — a decision the model was already making |
| **A pause is a return, not a block** | `interrupt()` throws, LangGraph writes state to the checkpointer, and `invoke()` returns. The process can exit. `Command(resume=x)` reloads and re-enters the node. This is why a checkpointer is mandatory: **without one, a pause is a crash with nothing to come back to** — and it is what lets the review survive an HTTP request ending |
| **HITL offers three outcomes, not one** | Approve is a gate. `revise` — routing back to `think` with a note — makes the human *part of the loop*. It caught Terminator 2 recommended as "creatures hunting people": it scored **0.536**, above every threshold, and no metric could have caught it |
| **The review surface must not misrepresent the draft** | `textwrap.wrap()` silently ate the newlines the model had correctly inserted, costing six revise cycles arguing with a display bug. **A review panel that lies is worse than no review**, because it manufactures disagreement neither party can see |
| **`backend/api.py` holds no agent logic** | It imports the same compiled graph the CLI runs. If the two ever disagree, one of them is a bug |

---

## Evaluation — three layers, and only one gets an LLM

| Decision | Why |
|---|---|
| **Tool accuracy and grounding are exact; only faithfulness is judged** | A tool name is a string; "did it name a film no tool returned?" is a set difference. Neither needs an opinion. Same rule as the search tool, one level up: **never ask the fuzzy machine a question the exact machine can answer** |
| **The judge is never the model under test** | Nova grading Nova measures self-consistency, not correctness. RAGAS runs on OpenRouter through the reranker's existing key |
| **Never score a refusal for faithfulness** | A refusal claims an *absence*, and a list of what IS there cannot support a statement about what isn't. The metric is structurally unable to score it — a correct refusal reads 0.00. Excluding it is refusing to apply a metric outside its domain |
| **Each layer catches the layer below's blind spot** | Exact metrics said 6/6 and missed invented reasons. The judge caught those and could never have caught Terminator 2. A human did. **Keep all three** |
| **A score is a proxy for relevance, and a proxy can be confidently wrong** | The two worst answers of the pass both scored *above* every threshold — T2 at 0.536, Shawshank as a "detective mystery" at 0.469. Thresholds say how sure the machine is, never whether it is right |
| **Reproduce before you interpret** | One run showed recall jumping 86.2 → 96.6%. It never came back. Write the prediction down *before* the run, and never author the explanation for the number you preferred |
| **Trace your own Python, not just the framework's** | LangChain traces itself for free; `core.search()` is plain Python and would appear as one opaque box. `@traceable` on embed / rerank / search opens it. Vectors are redacted — a trace holds evidence, not payload |

---

## Security and governance

| Decision | Why |
|---|---|
| **One `.env`, gitignored, loaded in exactly one place** | The single `load_dotenv()` line is the seam. Swapping to a secret manager deletes one line |
| **IAM user with no console access** | Least privilege by *kind*, not just amount. A leaked key isn't a browser session |
| **Two identities: app invokes, assistant reads** | Neither can do the other's job. Separation of duties |
| **`deny` rules in `.claude/settings.json`** | Enforced by the program, outside the model. **A prompt is a request; code is a guarantee** |
| **Never echo a secret — verify by property or by effect** | A key's *length* proves it's well-formed. A successful call proves it works. Neither exposes it |
| **A third-party register** | "Can we ship this?" should be answerable in one sentence, not by re-reading five sets of terms |
| **Source precedence declared per field, never globally** | An API may be authoritative on runtime while a press kit is authoritative on billing order |
| **Never rewrite source text** | `chunks` = what a source *said*. `claims` = what the system *believes*. Delete the loser and you destroy the evidence |

---

## Build versus buy

Every item on this roadmap has a managed AWS equivalent — Knowledge Bases, Evaluations, Guardrails,
Prompt Management, Prompt Router, Flows, AgentCore.

| Decision | Why |
|---|---|
| **Build them by hand first** | Clicking "Create evaluation" teaches nothing about what makes a good test case |
| **Compare against the managed version in Pass 3** | *"I built it, then evaluated theirs"* is an architect's answer. *"I used theirs"* isn't |

> **Buy the generic layer. Build the layer that encodes your business.**
> **Rent what isn't your differentiator. Own what is.**

The strongest version of that argument, in your own system: *the metric that matters most —
"is this film actually available to this user?" — doesn't exist in any managed eval service, because
no vendor knows your business rules.*

---

## Pass 3 — the graph, and learning to trust the instruments

| Decision | Why |
|---|---|
| **Genre becomes a NODE, not a column** | A column answers "is this Horror?". An edge also answers "which films share the most genres with Alien?" and "what connects these two films?". The moment you want to ask *what else is connected*, you needed an edge |
| **The graph derives from `movies.raw_payload`, not from `data/raw/`** | Reading the files would let the graph describe films that are not in `movies`. Reading the stored payload makes drift impossible |
| **The graph returns no scores** | A person either directed a film or did not. Attaching a confidence to a fact invites the caller to doubt it |
| **Three tools with disjoint triggers** — a description, one title, or a name/category | Two tools with a blurry boundary make a model oscillate. Each rule keys off a different thing in the sentence |
| **Every tool returns the text it matched on** | A tool that returns no evidence leaves a gap, and a model fills gaps from training. Learned twice: once for `search_films`, then repeated in `find_films_by_fact` a month later |
| **CI holds no credentials, ever** | A workflow with AWS keys is a workflow that can leak them, and a fork's pull request could read them. CI checks structure; the evals check behaviour on a machine that already has the keys |
| **The gate was proven to FAIL before it was trusted to pass** | Five deliberate breaks. That test found a bug in the checker itself — a syntax error crashed the run, so it exited 1 for the wrong reason, which reads exactly like working |
| **Faithfulness is reported with error bars, or not at all** | The same frozen answers scored 0.87 / 0.75 / 0.79. RAGAS decomposes an answer into claims with an LLM call, so the denominator itself moves. One draw is not a measurement |
| **A metric that fails must get LOUDER, not quieter** | A judge error printed one line and shrank the denominator, which *raised* the mean. A broken judge made the system look better |
| **Settled experiments get written down** | A bigger agent model is worse here; a 10x more expensive judge is not better. Both measured, both recorded, so neither is retried on a hunch |
| **An underpowered experiment is not a failed one** | Raising `EVIDENCE_CHARS` moved faithfulness less than the noise floor. Kept on mechanical grounds, explicitly not recorded as an improvement |
| **Ground truth about meaning comes from the human** | Which films *feel* warm, and whether a T-800 counts as a creature, are not facts a model can look up. Royson supplied both, and both changed the diagnosis |

---

## Pass 3 — availability, a semantic layer, and a screen

| Decision | Why |
|---|---|
| **The offer type lives in the EDGE TYPE, not in `properties`** | Amazon both rents and sells Alien: two different facts about the same pair. The UNIQUE constraint keys on (from, to, type, source), so as a property the database would silently keep one and discard the other. **A difference that matters must sit where uniqueness is enforced** |
| **The country lives in the edge's `source` (`tmdb:US`)** | "TMDB's US listing" and "TMDB's UK listing" are different claims. Adding a second country later cannot overwrite the first |
| **The database keeps TMDB's mess; `backend/providers.py` tidies at display time** | TMDB reports four separate Paramount+ entries for one thing a person calls Paramount+, and "Apple TV" beside "Apple TV Store" for a subscription and a shop. Storage stays faithful, presentation gets to be sensible. Keep the raw thing |
| **Every price carries a date and a source, and unverified ones say so** | Apple TV moved $12.99 → $14.99 on the day the file was written. 12 of 34 could not be confirmed from an official page and are marked, never guessed |
| **A semantic layer is born from a screen, not from an architecture diagram** | It was an abstract roadmap item for weeks. It became necessary the moment a panel had to show "Paramount+" four times |
| **Bands, never one sorted list** | $3.99 once and $8.99 a month are not the same kind of cost. A numeric sort puts the rental first and misleads. Free → Subscription → Rent → Buy → Needs a TV provider, cheapest within each |
| **"price unknown" is printed, never left blank** | An empty cell in a price column reads as *free* |
| **Rent and buy are shown as "from $x"** | TMDB publishes no per-film price. The mockup said "$3.99" flat, which invented a precision we do not have |
| **The panel fills from the DRAFT, not after approval** | The human-in-the-loop pause is for reviewing the WORDING. Hiding the evidence until after approval gets it backwards |
| **The panel may only show films the agent named, and not ones it named to reject** | Same grounding rule the agent works under. It once showed Jurassic Park as pick #1 of an answer that said "…but are not Jurassic Park" |
| **`REGION` is defined once, in `backend/providers.py`** | A constant written down twice is a constant that will eventually disagree with itself |
| **A function that fetches its own input cannot be tested cheaply** | `films_mentioned` took a database call; it now takes a list. Same behaviour, injectable |
| **Fixed pixels for the poster and its column; the offer list absorbs the resize** | A poster that scales with the window makes the row feel unstable, and the title and poster have a *correct* size. Only the offer list genuinely reads fine narrower |
| **No structural breakpoints in the UI** | A narrow window gets a smaller version of the same layout, never a different one |

### Hard constraints, and making them speak

| Decision | Why |
|---|---|
| **A length or year describes the REQUEST, not the person** | A mood can carry across a conversation; "under two hours" cannot. The agent carried `max_runtime=120` into "something fictional with magic" and deleted the only film in the catalogue about magic |
| **A hard filter must report what it removed** | Enforced exactly is the value AND the danger: a spurious constraint does not degrade the answer, it deletes the right one and leaves no trace. `max_runtime=120` removes **10 of 20 films** here |
| **An alarm that fires on the normal case is not an alarm** | The first wording told the model to "drop it and search again" on *every* filtered search — including correct ones, and contradicting an instruction four paragraphs above it. Now it reports facts and leaves the judgement to the caller |
| **Report facts, not directives, in a tool result** | Facts survive being read a hundred times. Directives get ignored, or obeyed in the wrong case |
| **The trace shows what each tool TOUCHES** | Node, arguments, services, tables. Three of the four tools never reach a model, and the screen now says so as it runs — the architecture's central claim, checkable at a glance instead of remembered |
| **Only this turn's tool calls are printed** | `/api/ask` returns the whole thread's trace every time. Reprinting all of it turned the log into a wall of repeats by the third question |

---

### Testing and CI

| Decision | Why |
|---|---|
| **Gate on deterministic things; report non-deterministic things** | Faithfulness swung 0.78 → 0.72 across two runs with no code change. Gating on a metric that cannot resolve your change teaches the team to ignore red builds |
| **The test environment is DERIVED from the code, not typed out** | The hand-written list said `AWS_DEFAULT_REGION` where the code says `AWS_REGION`. It passed on the laptop, because a developer shell has already exported the real values, and failed on the first clean machine |
| **A green build is not a working system** | Four CI jobs passed on an application where every single request returned 500 |
| **Tests guard the code you are NOT currently looking at** | A test written the previous day caught a bug in that day's fix — the new `reasons_for` returned another film's sentence as this film's reason |
| **Verify by running the real function the way the caller calls it** | A fix was "verified" in a scratch copy called with three arguments; the test calls it with two, and the fix depended entirely on the third |
| **A count says something happened; only the content says what** | "critic struck 2 of 4 lines" told us nothing for three runs. Printing the struck TEXT solved it in one |
| **Drift alarms run on a SCHEDULE, not on a change** | Nothing in the repository changes when Apple raises a price. Only the calendar knows |

---

## 30 Aug 2026 — SOLID audit, and the folder restructure it caused

An honest read of the five principles against the code as it stood, and what each verdict
changed. Two were already good, one does not apply, two were violated.

| principle | verdict | what was done |
|---|---|---|
| **S** single responsibility | **violated** — the single **core** module was 786 lines and 14 functions doing embedding, retrieval, reranking, the knowledge graph, availability and filter reporting. The neighbours map said "change this → re-test everything", which is the cost written down | split into `backend/config.py`, `backend/models.py`, `backend/retrieval.py`, `backend/graph.py` and `backend/tracing.py`. The vector half and the exact half shared a database URL and nothing else |
| **O** open/closed | **already good, and deliberate** — a new tool is a row in `TOOLS`, a new price a row in `SERVICES`, a new offer type a row in `OFFER_EDGE`, a new doc a row in `PAGES` | left alone. Extending by adding data rather than editing logic is the principle working |
| **L** Liskov substitution | **not applicable** — five classes in the repo, zero inheritance. Claiming a pass would be theatre | nothing. Recorded so the gap is a decision, not an oversight |
| **I** interface segregation | **the strongest part** — each tool takes exactly the arguments its job needs. Not merely clean: narrow, non-overlapping interfaces are what make the model route correctly. A single `do_movie_stuff(**kwargs)` would work in Python and fail as an agent | left alone |
| **D** dependency inversion | **violated, and it was a promise** — Pass 0 named five seams as "must be right even here". Four were never built: `MovieDataSource`, `EmbeddingProvider`, `LLMProvider`, `Retriever`. No `Protocol` or `ABC` anywhere | `backend/models.py` now holds every call that leaves the machine to reach a model. It is a module boundary rather than an abstract class, because there is exactly one implementation and inventing an interface for one implementation is its own smell |

| Decision | Why |
|---|---|
| **One folder per LIFECYCLE, not per topic** | 17 files in the root had six different lifecycles mixed together. `backend/` runs in production, `pipeline/` builds the corpus by hand, `evals/` needs credentials, `scripts/` is dev tooling, `tests/` is permanent, **`experiments/` is the only folder meant to be deleted** |
| **`backend/models.py` is the only file that names a vendor** | Swapping Bedrock or Cohere means editing one file. `backend/retrieval.py` asks for `embed(text)` and does not know who answers. The roadmap called this `EmbeddingProvider`; this is the same idea with less ceremony |
| **A module boundary instead of an abstract class** | One implementation does not justify an interface. The dependency still points inward, and nothing above `backend/models.py` names a vendor — which was the actual goal |
| **Three of the five promised seams are dropped on purpose** | `MovieDataSource`, `LLMProvider` and `Retriever` are YAGNI until a second implementation exists. Written down so the gap reads as judgement rather than oversight. `EmbeddingProvider` was the one with teeth — changing the embedding model means re-embedding the corpus — and it is now `backend/models.py` |
| **Everything runs from the root as a module** — `python -m backend.tools` | One rule for every entry point, no `sys.path` juggling for the reader, and the same commands work in CI |
| **A dated log may name files that no longer exist** | The dead-reference check now exempts `docs/session-notes.md`. Rewriting old entries to match a new layout would falsify the record. Documents describing the CURRENT system are still checked strictly |
| **A file .gitignore excludes is never a dead reference** | `docs/session-notes.md`, `docs/roadmap.md` and `docs/groundwork.md` are private and never committed, so no checkout can contain them. `repo_check` reads that exemption out of `.gitignore` itself rather than keeping a second list, and applies it BEFORE asking whether the file exists — so the gate gives the same answer on a laptop, where the file is present, as in CI, where it never is |
| **The restructure was one pull request, pure moves, no logic changes** | One variable. And it was only safe because the harness existed first: 37 tests, four CI jobs, a docs gate and a structure gate all go red the moment a move breaks something |

---

## Deployment — 1–2 September 2026

| Decision | Why |
|---|---|
| **ECS Fargate, not EC2** | There is a computer underneath and we never see it: no OS to patch, no SSH, no restart script. EC2 would earn its keep only for a specific machine shape — a GPU, a pinned kernel — and this needs neither |
| **Not Lambda** | The agent loop takes 20+ seconds and holds a Postgres connection. Lambda suits short, spiky, event-driven work; a cold container reconnecting to the database on every request is slow and unkind to it. Lambda *would* fit the nightly availability refresh |
| **ECS Express Mode** | App Runner stopped accepting new customers on 30 April 2026. Express Mode creates the load balancer, certificate, listener, target group, log group and scaling policy in one form — and deletes them together |
| **A two-stage Docker build** | Node compiles the front end in stage one and is discarded; the runtime image carries Python only. The build tool is not a runtime dependency |
| **`requirements-runtime.txt`, separate from `requirements.txt`** | The image should not ship RAGAS, pytest or the eval judge. Three tests keep the two files honest: the runtime file must cover every `backend/` import, versions must match, and nothing extra may creep in |
| **`--platform linux/amd64` and `--provenance=false`, always** | Apple Silicon builds do not run on AWS, and Docker's attestation manifest is rejected by ECS. Both are silent failures — the first push shipped the wrong image entirely and only the digest revealed it |
| **The image tag is the deploy unit** (`:v1`, `:v2`, `:v3`) | A moving `:latest` makes "which code is running?" unanswerable. A tag per deploy makes rollback a dropdown |
| **`/` serves the React build; `/classic` keeps the original page** | The public URL must show the finished thing. The original page renders the raw agent trace — every tool call and score — which the React panels summarise away, so it is kept as a debugging surface rather than deleted |
| **Secrets in Secrets Manager, not environment variables** | An environment variable is readable by anyone with console access and gets printed in error text. The three questions are answered by three services on purpose: **where is it stored** (Secrets Manager), **who may read it** (IAM), **who needs it** (ECS) |
| **An inline, prefix-scoped IAM policy — not `SecretsManagerReadWrite`** | `GetSecretValue` on `moviemotions/*` only. The AWS-managed policy grants every secret in the account plus write access. The prefix is what let the second secret need no IAM edit at all |
| **`ecsTaskExecutionRole` and `moviemotions-task` stay separate** | The execution role is what *ECS* uses to pull the image, write logs and fetch secrets. The task role is what *the application* uses to call Bedrock. Merging them gives the app permissions it never needs |
| **An empty model reply is refused at the door, not cleaned up later** | `think()` is the only place a reply enters the state. Sanitising history at send time would fix one caller and leave the bad value stored — and the bad value is what breaks the *next* request |
| **One change per deployment** | Both failed deployments this session changed the image and something else at once, so each needed a round of guessing. The rule was already in force for code; it applies to configuration identically |
| **Test on `localhost` before building an image** | A Docker build takes minutes and hides its own output. Every deployment this session that started with `uvicorn --port 8010` succeeded first time |

---

## Four things worth remembering above all

1. **RAG is a noun, agentic is a verb.** A pipeline versus a loop. Retrieval is one tool the loop can use.
2. **The flowchart test.** If you can draw it before the query arrives, it's a pipeline. If the shape depends on the query, it's agentic.
3. **Structure is created at write time, not read time.** Someone always pays the extraction cost.
4. **Keep the raw thing. Derive everything else from it.**

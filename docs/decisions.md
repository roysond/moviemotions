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
| **The database keeps TMDB's mess; the pricing layer tidies at display time** | TMDB reports four separate Paramount+ entries for one thing a person calls Paramount+, and "Apple TV" beside "Apple TV Store" for a subscription and a shop. Storage stays faithful, presentation gets to be sensible. Keep the raw thing |
| **Every price carries a date and a source, and unverified ones say so** | Apple TV moved $12.99 → $14.99 on the day the file was written. 12 of 34 could not be confirmed from an official page and are marked, never guessed |
| **A semantic layer is born from a screen, not from an architecture diagram** | It was an abstract roadmap item for weeks. It became necessary the moment a panel had to show "Paramount+" four times |
| **Bands, never one sorted list** | $3.99 once and $8.99 a month are not the same kind of cost. A numeric sort puts the rental first and misleads. Free → Subscription → Rent → Buy → Needs a TV provider, cheapest within each |
| **"price unknown" is printed, never left blank** | An empty cell in a price column reads as *free* |
| **Rent and buy are shown as "from $x"** | TMDB publishes no per-film price. The mockup said "$3.99" flat, which invented a precision we do not have |
| **The panel fills from the DRAFT, not after approval** | The human-in-the-loop pause is for reviewing the WORDING. Hiding the evidence until after approval gets it backwards |
| **The panel may only show films the agent named, and not ones it named to reject** | Same grounding rule the agent works under. It once showed Jurassic Park as pick #1 of an answer that said "…but are not Jurassic Park" |
| **`REGION` is defined once, in the pricing layer** | A constant written down twice is a constant that will eventually disagree with itself |
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
| **S** single responsibility | **violated** — the single **core** module was 786 lines and 14 functions doing embedding, retrieval, reranking, the knowledge graph, availability and filter reporting. The neighbours map said "change this → re-test everything", which is the cost written down | split into `backend/config.py`, `backend/models.py`, `backend/tracing.py` and separate retrieval and knowledge-graph modules. The vector half and the exact half shared a database URL and nothing else |
| **O** open/closed | **already good, and deliberate** — a new tool is a row in `TOOLS`, a new price a row in `SERVICES`, a new offer type a row in `OFFER_EDGE`, a new doc a row in `PAGES` | left alone. Extending by adding data rather than editing logic is the principle working |
| **L** Liskov substitution | **not applicable** — five classes in the repo, zero inheritance. Claiming a pass would be theatre | nothing. Recorded so the gap is a decision, not an oversight |
| **I** interface segregation | **the strongest part** — each tool takes exactly the arguments its job needs. Not merely clean: narrow, non-overlapping interfaces are what make the model route correctly. A single `do_movie_stuff(**kwargs)` would work in Python and fail as an agent | left alone |
| **D** dependency inversion | **violated, and it was a promise** — Pass 0 named five seams as "must be right even here". Four were never built: `MovieDataSource`, `EmbeddingProvider`, `LLMProvider`, `Retriever`. No `Protocol` or `ABC` anywhere | `backend/models.py` now holds every call that leaves the machine to reach a model. It is a module boundary rather than an abstract class, because there is exactly one implementation and inventing an interface for one implementation is its own smell |

| Decision | Why |
|---|---|
| **One folder per LIFECYCLE, not per topic** | 17 files in the root had six different lifecycles mixed together. `backend/` runs in production, `pipeline/` builds the corpus by hand, `evals/` needs credentials, `scripts/` is dev tooling, `tests/` is permanent, **`experiments/` is the only folder meant to be deleted** |
| **`backend/models.py` is the only file that names a vendor** | Swapping Bedrock or Cohere means editing one file. The retrieval module asks for `embed(text)` and does not know who answers. The roadmap called this `EmbeddingProvider`; this is the same idea with less ceremony |
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

## Corpus derivation — 3 September 2026

| Decision | Why |
|---|---|
| **Vertex is used at BUILD time, not run time** | Gemini as the live agent: 57.6s and 48.3K tokens against Nova's ~2s and ~10K, for fewer films. Writing the mood corpus runs offline with nobody waiting, so the one thing Gemini is bad at costs nothing there |
| **Bedrock stays the default agent; the seam stays** | The loss was a *prompt* loss, not a model loss. The system prompt was tuned for Nova over weeks. **Prompts do not transfer between models** |
| **The mood judge is Royson's hand-written ground truth, not an LLM** | Which films feel warm is taste. A vendor eval service can score how well the text is written; it cannot score whether the text makes retrieval find the right films, because it does not know the database exists |
| **A generic quality metric is not a task metric** | The distinction that decides whether a vendor eval product is the right tool. Usually it is not |
| **The bake-off writes nothing to the database** | `chunks` allows one row per (film, index), so four models' mood text cannot coexist. Running it in memory also means a bad challenger can never damage the live corpus |
| **The bake-off parses the prompt out of `pipeline/derive_corpus.py`, never copies it** | A copied prompt drifts. Then the arms differ by model *and* instructions, and no number can be attributed to either |
| **The noise floor is printed above the result** | 5 queries, ceiling 15 — one film moving is 6.7 points. Stating it after seeing the number is how a non-result gets believed |
| **`gemini-2.5-pro` over `gemini-3.1-pro-preview`** | Preview can change without warning. A model that silently rewrites your corpus later is not one you can promote, however good it is today |
| **Model first, input second — never both** | Deriving mood from the plot as well as the overview is a second variable. Two clean measurements, or an unattributable one |

### Storage — the fact that sets the ceiling  ·  SUPERSEDED 5 Sep 2026

> **This block describes the design that the September rebuild replaced.** It is kept because the
> ceiling it identifies is *why* the rebuild happened, and because the open proposal at the end of
> it is still the open question. See "RAG rebuild" below for what replaced it.

The old loader glued `feel`, `moods` and `themes` into **one paragraph and one vector**. The
individual moods cease to exist as data at that point. "Which films have the mood *panic*" is
not a question the database can answer.

Moods and themes are **not** in the knowledge graph — it allows film, person, genre, keyword and
provider only, and is built entirely from `movies.raw_payload`.

**The open proposal:** two new node types (`mood`, `theme`) rather than new tables, `feel` as a
column on `movies` rather than a table, and one extra argument on `find_films_by_fact` rather
than three new tools. Blocked on a decision only Royson can make — **writing the fixed mood
vocabulary**, the way TMDB fixed the 19 genres. Free-text moods cannot be looked up; nobody will
ever type "sea peril".

---

## RAG rebuild — corpus and retrieval design, 5–7 September 2026

The corpus was demolished and rebuilt. `feel`, `moods` and `themes` are gone, and so is the
loader that glued them into one paragraph. Three fields replace them, written by one model call
per film. **This section supersedes "Storage — the fact that sets the ceiling" above.**

### The three derived fields

| Decision | Why |
|---|---|
| **`mood_feel`, `theme`, `premise` — three fields, ONE call** | They change for the same reason: Royson decided the text should say something different. Three calls would cost 3x per run and buy independent tuning nobody wants yet. Split `premise` out the day it needs its own model, not before |
| **One file, five NAMED prompt blocks — not one file per field** | A file per field is SRP applied at the wrong grain: all three change together because the taste behind them is one taste. Named blocks give the same "edit one concept in one clearly labelled place" without three imports |
| **A partial run never writes `data/derived.json`** | Half a corpus in the real file looks exactly like a whole one, and nothing downstream can tell the difference. `--limit` and `--titles` write `data/derived.sample.json` instead |
| **`premise` is the DISPLAY surface; `theme` and the plot scenes are MATCHING surfaces** | You never have to classify spoilers if spoilers never reach the screen |
| **`theme` is matching-only and is NEVER shown to a user** | The prompt was pushed twice to make it spoiler-free and moved the wording without moving the fact. For Get Out the underlying idea *is* the twist, so the only safe theme is a vague one — and a vague theme vector matches everything, which is the same as matching nothing. **Placement solved what prompting could not** |
| **The spoiler rule had to ban the DESTINATION, not just events** | "Builds toward a feeling of earned liberation" and "learns to care for its human charge" name no event and give the ending away anyway. Forbidding events left the arrival point wide open |
| **`mood_feel`: one feeling, plain word first, no plot clause** | A mood that continues "…as an unseen threat stalks its prey" puts plot words inside the one field that exists to hold none. And a sentence leaning two ways leans strongly at neither, so a one-word query for either feeling misses both |
| **A prompt cannot fix a CORPUS-LEVEL property** | Each film is derived in its own call, so the model cannot know it already opened four films with "Frantic". Measured: 13 distinct opening words across 20 films, half the corpus sharing one. Repetition is a property of the FILE; the prompt only ever sees ONE FILM. Any fix is a second pass over the finished file, or acceptance |
| **Judge a corpus by counting, not by reading down the terminal** | Reading the 20-film output missed two entries that named a character and one that named two feelings. A ten-line script found all three, and also proved that two of my own flags were the checker being wrong |
| **`text_of()` — a model reply is not always a string** | Gemini returns a list of content blocks. `str(list)` yields Python's repr, `json.loads` then fails at character 2, and it reads exactly as though the MODEL had misbehaved. It had not |

### Two model roles, two switches

| Decision | Why |
|---|---|
| **`AGENT_PROVIDER` and `DERIVE_PROVIDER`, never one `LLM_PROVIDER`** | With a single switch, reaching for a better *writer* also moved the live agent. The two jobs have opposite constraints: the agent has a person waiting, the deriver has nobody and only quality counts |
| **`backend/models.py` holds ONE private builder and two named wrappers** | `chat_model()` and `derive_model()` differ only in which config they hand over. The role name travels into the error text, so a failure says WHICH job could not start |
| **A prompt file never names a vendor** | `pipeline/derive_corpus.py` asks for a model and does not know who answers. `tests/test_seam.py` is what keeps that true |

### The `movies` table

| Decision | Why |
|---|---|
| **`movie_id` is the primary key; `tmdb_id` is merely UNIQUE** | The surrogate key protects the ROW, the natural key protects the FILM. `source` and `source_id` were deleted along with the idea that an identifier could belong to a provider |
| **Columns sourced from TMDB carry a `tmdb_` prefix** | `tmdb_overview`, `tmdb_raw_payload`. The column name states its provenance, so nobody has to remember it |
| **Postgres cannot reorder columns — recreate and copy, inside `BEGIN … COMMIT`** | Dragging columns in a client changes the client's view and not the table. And a create/copy/drop that half-completes leaves a worse state than one that fails outright |
| **Every loader is safe to re-run — `ON CONFLICT (tmdb_id) DO NOTHING`** | Proven live: a second run of `pipeline/load_corpus.py` printed `inserted 0, skipped 20` |
| **Plots are matched on `tmdb_id`, never on title** | Titles collide, and remakes share them exactly — the corpus holds *The Karate Kid (2010)*, not the 1984 film. Proven live: `loaded 20 · no plot 0 · unmatched 0` |
| **`pipeline/load_plots.py` reports three outcomes separately** | "Loaded", "no plot available" and "no matching film" fail for different reasons and need different fixes. One combined number hides which one happened |

### "Similar to X"

| Decision | Why |
|---|---|
| **Reuse X's STORED vectors; do not rebuild a text query out of X's fields** | X's mood vector already *is* the point to search around. Re-describing X and re-embedding costs a call and lands somewhere slightly different, for nothing |
| **Never inherit X's hard facts** — year, runtime, cast, crew, language, keywords | "Similar to Alien" carrying 1979 asks for 1979 films and deletes the answer. Only the FEELING is inherited |
| **`premise` sits out of "similar to", but does the work when a user DESCRIBES a film** | A premise is a setup, so it finds the same *situation* rather than the same *feeling*. Same field, two jobs — and the job is chosen by the shape of the question |
| **Whatever the USER says wins; whatever they don't say is copied from the film** | One rule covers both mixed cases: "similar to Alien but funnier" takes mood from the user and theme from Alien; "similar to Alien from the last 10 years" takes both from Alien and adds a year wall. No special case needed |
| **Sequels are excluded, and the user is TOLD they were** | Toy Story 2 is the most similar film to Toy Story and the least useful answer, because they have seen it. Verified in the stored payload: 15 of 20 films carry `belongs_to_collection.name`, and the 5 blanks are genuinely standalone films |
| **Genre gets NO hard filter — and the decision was made by DEFERRING it** | Either genre mirrors mood, in which case a wall adds a noisy label and no information; or it does not, in which case the wall deletes correct answers. A submarine film that feels like Alien is exactly what this app exists to find and exactly what a genre wall removes |
| **Only agonise over decisions that are EXPENSIVE to reverse** | Genre-as-wall versus genre-as-bonus is one line in scoring, changeable forever. Table shape is not. And at 20 films a wall and a bonus return identical results, so deciding today would be guessing where evidence arrives later |

---

## Retrieval rebuilt, and the law it kept breaking — 7 September 2026

The corpus grew a fourth surface, the search was written from scratch, and the same
mistake was found three times in one day wearing three different disguises.

### THE LAW: a cosine is comparable within ONE kind of text and meaningless across two

| Decision | Why |
|---|---|
| **Scores are compared by RANK WITHIN A KIND, never as raw numbers across kinds** | Measured on one query: scenes scored 0.338-0.508, premises 0.457-0.569, themes 0.513-0.561. Not overlapping ranges — different SCALES. Any operation putting two kinds side by side inherits the offset |
| **Caught three times in one day, and it looked like three different bugs** | `min(mood, theme)` was ALWAYS the theme score, so "similar to X" was decided entirely by theme and never by mood. `max(premise, scene)` was ALWAYS the premise, so **1 film in 20** had its best scene beat its own premise and the 148-row scene corpus was never in the competition. Both were one bug |
| **Theme is a weak discriminator, and the reason generalises** | Toy Story's nearest theme in the catalogue is TITANIC at 0.827 — both are built as "not status, but X". Abstract moral statements resemble each other; sensory concrete sentences do not. **Things match when they are at the same level of abstraction**, which is also why an abstract query finds a summary rather than the scene it describes |
| **"similar to X" seeds from MOOD only** | Mood alone ranks Finding Nemo and Home Alone top for Toy Story — the two answers a person gives. Adding theme replaced them with Mortal Kombat |
| **RANK says which film is best. It cannot say whether ANY of them is good** | With twenty films the winner is rank 1.00 whether it fits perfectly or not at all. So each film also carries its raw score and its margin over the bottom of its own band. A number that always looks like confidence is not a measurement |
| **There is no usable score THRESHOLD in this space** | Unrelated text scores about 0.64 for a mood query and nothing ever scores below 0.6. The previous build's "refuse under 0.25" could never fire. Calibration has to come from the field this query produced, never from a number chosen once |
| **A floor needs a field deep enough to have a bottom** | After a hard filter leaves three films, the worst of three is not "unrelated", it is just third. Below eight candidates no margin is reported and the caller is told why, rather than handed a number that means nothing |

### Plot scenes

| Decision | Why |
|---|---|
| **A whole new corpus needed NO schema change** | 148 scenes went into `movie_data` as `data_kind = 'plot_scene'`, `seq = 1..n`, and their vectors into `movie_vectors` like every other vector. That is the tall-table decision earning its keep: had the three derived fields been columns on `movies`, scenes would have needed a second table, a second loader and a second embedder |
| **Semantic cut, then recursive, then overlap — in that order** | Cut where the meaning changes; split anything still over the cap at its own biggest internal drop; overlap ONLY the seams the cap forced. At a real meaning boundary, repeating a sentence blurs two distinct scenes — overlap repairs damage, it is not a default |
| **The break threshold is a PERCENTILE of each plot's own joins** | "Split below 0.8" dies on a model swap and would cut every sentence in this space, where unrelated text sits at 0.64. Each plot is scored against itself |
| **When the rules conflict, the SIZE CAP wins** | No scene over 900 characters, none under 200, never cut where the meaning held — a short segment between two full ones breaks one of them whichever way it moves. An oversized chunk averages several events and matches none of them sharply, which is the failure chunking exists to prevent; a short one is merely weak |
| **A merge looks backwards AND forwards** | Merging only into the previous scene produced a 53-character orphan whose neighbour was already full. Both directions, and if neither has room the short scene stands |
| **A re-cut deletes the scenes it no longer believes in** | Nine scenes replacing twelve leaves three embedded, findable rows describing a chunking that no longer exists |
| **`plot_scene` is a MATCHING surface, never a display one** | A scene from the third act is the sharpest thing to search and the least safe to print. Same rule as theme, enforced in the same `DISPLAYABLE` set rather than in an instruction someone has to remember |
| **A generic chunk is a universal weak match** | Get Out's cold open — "a man walks alone at night, talking on the phone" — was the top-scoring scene in the catalogue for four unrelated queries. Text with no distinctive vocabulary sits near the middle of the space, and the middle is close to everything. Not a scoring bug; a property of that row |

### The tool layer

| Decision | Why |
|---|---|
| **TWO tools, not three: "similar to" is an ARGUMENT** | "Like Alien, but funnier" needs a comparison film AND a mood in the SAME call. Two separate tools cannot express that sentence at all — the model would have to pick one half and discard the other. **Where two tools would always have to be called together, they were one tool** |
| **A typed constraint ADDS to a seeded one; it does not replace it** | "But funnier" is a modification, not a replacement. Replacing gave a plain list of funny films with nothing of Alien left in it |
| **Two constraints of the same kind may be combined; two of different kinds may not** | "Like Alien" and "funny" are both mood comparisons, so they share a scale and taking the minimum is honest. That is the same law as above, used the other way round |
| **Withheld evidence is DECLARED, never silently absent** | A film returned with a score and no reason is a gap, and a gap is filled from the model's training. When the matched text may not be shown, the tool says so explicitly and forbids inventing it |
| **Both display surfaces are returned every time, whichever kind matched** | A film returned with only its premise leaves nothing to say except the plot — and "say why it fits, do not retell the plot" is then an impossible instruction. The rule and the data contradicted each other, and the data won |

### The agent, and the limits of a prompt

| Decision | Why |
|---|---|
| **An EXAMPLE in a prompt gets FILLED IN, not imitated** | A worked "right answer" naming one film produced a factually false sentence about a different one — the model swapped the title and kept the description. Recorded once before about docstrings; it cost a whole answer to learn again. **No prompt in this repository names a film** |
| **Adding rules to a saturated prompt makes it worse** | Four rewrites, each longer, each fixing one behaviour and losing another. The prompt was cut roughly in half and the rules made mechanical — "the first characters you write are a film's title" rather than "do not write an opening sentence" |
| **Formatting is deterministic, so CODE owns it** | Three prompt attempts failed to remove an opening sentence. It is now stripped after the fact — the same split as the critic: the model writes, the code edits |
| **A guarantee placed on an optional code path is not a guarantee** | That stripper was first written inside the critic node. `CRITIC_ENABLED = False`. It had never run once, and was reported as tested |
| **The model chooses tools well and writes badly, and those are separable** | Correct tool every time, `similar_to` set correctly, two arguments combined in one call, a refusal on an unanswerable query — while copying its own tool output back as the answer. The split that already exists for AGENT and DERIVE applies one level down |

### Verification and the shape of the code

| Decision | Why |
|---|---|
| **`backend/vectors.py` holds the two facts both sides must agree on** | The variant string and the literal format. Written twice, they would eventually disagree, and the failure is silent: the writer stores vectors under one name and the reader looks for another, so search returns nothing at all |
| **`pipeline/embed_data.py` asks the DATABASE what is missing** | Rather than being told what to embed. A row with no vector IS the definition of work outstanding, which makes the script resumable by nature and meant 148 plot scenes were picked up with no change to it at all |
| **`search.py` exists to separate two suspects** | When an answer is wrong, either retrieval found the wrong films or the model wrote badly about the right ones. One command, no model, and you know which before anything is changed |
| **A test that fails for the WRONG reason trains you to ignore red** | Three failures after the rebuild: one was a real regression — `search()` had lost its `@traceable` and every search would have been an opaque box — and two named modules and variables deleted days earlier. The first was fixed in the code, the other two in the tests |
| **A long-running process holds the code it imported** | Two rounds of "the fix did not work" were a server and a prompt still running the module they loaded at startup. Restart before believing a result |
| **The regex in `backend/api.py` is the only thing joining the tool's prose to the screen** | Nothing type-checks it. Change a tool's wording and the panel silently empties — no error, no crash. It has broken twice. It is the first place to look when the display goes blank |
| **A trace that lies is worse than no trace** | The UI printed "pgvector cosine over chunks + chunk_embeddings → Cohere rerank → damped sum" for weeks after none of those existed. Same failure as the review panel that ate newlines: a surface that misreports manufactures disagreement nobody can see |

---

## The reranker, and the writer split — 8 September 2026

| Decision | Why |
|---|---|
| **Vector rank chooses WHICH text represents a film; the reranker chooses the ORDER** | A cross-encoder reads the query and the document together and judges relevance directly, so its scores are comparable between a premise and a plot scene — which cosine is not. It solves the band problem as a side effect of doing its actual job |
| **The reranker is the FINAL ranker, not a signal to blend** | Blending a calibrated relevance score into an uncalibrated distance would destroy the only property that makes it useful |
| **A rerank score is the first number in this project that means something ON ITS OWN** | Measured across eight queries: a query with a real answer tops 0.3 and often 0.5 — Predator 0.711, Home Alone 0.551, Finding Nemo 0.521. A query with no answer here leaves the whole field under 0.1. Every earlier attempt at a threshold failed because cosine has no absolute meaning and rank always makes the winner look perfect |
| **The rerank query must carry EVERY constraint, not just the typed ones** | Built from the typed text alone, "similar to Alien, but funny" asked the reranker only for "funny" and returned the film scoring WORST on the Alien half at number one — the vector stage had it 18th and was right. **A final ranker given a partial query does not refine the earlier work, it discards it** |
| **The reranker reads text the user may never see** | It ranks on the strongest matched row, third-act scene included, and that text is stripped before anything renders. Internal evidence and displayable evidence are different questions |
| **Vendor outage costs QUALITY, never AVAILABILITY** | A failed rerank falls back to vector order with a note saying so |
| **A first-party reranker is now possible and was not in August** | Vertex AI has a Ranking API and LangChain has an integration for it. Checked, not remembered. Not switching yet: measure the current one first, then swap behind `backend/models.py` and compare. Moving a reranker between two vendors by editing one file is what the seam was built for |

### Fine-tuning: form, not facts

The prose problem — four prompt rewrites, each longer and worse — is a FORM problem, and
form is what fine-tuning is reliably good at. Knowledge is what it is bad at, and here
knowledge comes from the corpus anyway.

| Decision | Why |
|---|---|
| **Split REASONING from COMMUNICATION** | A managed model reasons over the retrieved evidence; a fine-tuned model says it in the house voice. The writer never needs to know a single film exists, so nothing stale is ever baked into weights |
| **The handoff between them is STRUCTURED, never prose** | Reasoner emits films, evidence and reasons as JSON; the writer turns that into the sentence. Prose-to-prose makes it impossible to tell which stage invented something; structure makes "the writer named no film the reasoner did not" a mechanical check. Also finally adopts the Pass 0 habit that was never taken up, and is why `backend/api.py` reads the answer with a regular expression today |
| **This IS the multi-agent item** | Two agents with genuinely different jobs and genuinely different models, with a checkable contract between them — rather than two agents that could have been one |
| **The training data is the work, not the training** | A few hundred examples of evidence bundle to ideal answer, generated by a strong model and hand-corrected. The correcting is what encodes the taste, and the taste is the point |
| **Hallucination MOVES, it does not disappear** | A fine-tuned writer handed evidence can still add a flourish nothing supports. The critic and the grounding check stay |
| **Where to fine-tune is a COST decision, not a capability one** | A SageMaker endpoint bills by the hour whether or not anyone queries it; Bedrock bills per token. To be checked against current documentation before planning, not decided from memory |

---

## Four things worth remembering above all

1. **RAG is a noun, agentic is a verb.** A pipeline versus a loop. Retrieval is one tool the loop can use.
2. **The flowchart test.** If you can draw it before the query arrives, it's a pipeline. If the shape depends on the query, it's agentic.
3. **Structure is created at write time, not read time.** Someone always pays the extraction cost.
4. **Keep the raw thing. Derive everything else from it.**

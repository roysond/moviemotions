# The full retrieval pipeline

Five stages. Each receives something, does one job, passes something on.

```
                    USER QUERY
   "something with creatures chasing you, Denzel, under 2 hours"
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 0 · PARSE  (LLM, extraction not inference)           │
│ Split the sentence into two kinds of thing:                │
│   hard → { actor: "Denzel Washington", max_runtime: 120 }  │
│   soft → "creatures chasing you, intense"                  │
└───────────────────────────────────────────────────────────┘
              hard facts ↓            soft text ↓
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 1 · METADATA FILTER — THE GATE  (plain SQL WHERE)    │
│ Throw out everything that breaks a hard rule BEFORE search.│
│ No Denzel? Over 2 hours? Gone — never considered.          │
│   candidate set = films WHERE 'Denzel' = ANY(cast)         │
│                         AND runtime_minutes <= 120         │
│ Runs FIRST: a hard constraint is a guarantee, and a        │
│ smaller set is faster and more accurate to search.         │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 1b · KNOWLEDGE GRAPH — A SMARTER GATE                │
│ Filter by RELATIONSHIPS, not just attributes.             │
│ Metadata asks "has property X?"; the graph asks "do two   │
│ entities RELATE the way the query needs?" Typed edges:    │
│ a creature that HUNTS a human != one that is a PET.       │
│                                                            │
│   query needs:  creature --HUNTS--> human                 │
│   Predator      creature --HUNTS--> human    -> keep      │
│   Jurassic Park creature --HUNTS--> human    -> keep      │
│   Alien         creature --HUNTS--> human    -> keep      │
│   The Hangover  tiger --IS_PET--> (comedy)   -> DROP      │
│                                                            │
│ THE TIGER FIX: "creatures chasing you" wrongly returned   │
│ The Hangover ("a tiger in the bathroom") — the vector     │
│ knew a creature was mentioned but not its ROLE. The graph │
│ drops it HERE, at the gate, before any search runs.       │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
   +=======================================================+
   | THE CANDIDATE SET — "the arena"                       |
   |   { Predator, Jurassic Park, Alien }                  |
   | This narrowed set STAYS FIXED while the two searches  |
   | run. The graph drew these walls; both searches operate|
   | only inside them. The graph does not rank or fuse.    |
   +=======================================================+
                        │
          ┌─────────────┴─────────────┐   (same query, two methods)
          ▼                           ▼
┌────────────────────┐      ┌────────────────────────┐
│ STAGE 2a · VECTOR  │      │ STAGE 2b · BM25 / KEYWORD │
│ Match by MEANING.  │      │ Match by LITERAL WORDS.   │
│ Embed soft text,   │      │ Score query's actual      │
│ cosine vs candidate│      │ words vs the SAME content │
│ vectors. Links     │      │ text — no embedding.      │
│ "creatures"→       │      │ Catches names/terms a     │
│ "dinosaur".        │      │ vector blurs.             │
│                    │      │                           │
│ list A (ranked):   │      │ list B (ranked):          │
│  1 Predator        │      │  1 Alien                  │
│  2 Jurassic Park   │      │  2 Predator               │
│  3 Alien           │      │  3 Get Out                │
└────────────────────┘      └────────────────────────┘
          │                           │
          └─────────────┬─────────────┘
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 3 · RECIPROCAL RANK FUSION                           │
│ Merge the two lists into one — using POSITION, not score.  │
│ Throw the scores away (they're on different scales).       │
│ Each film scores 1/(60 + rank) for every list it's in;     │
│ add them; re-sort. Dedupe to one row per film here.        │
│   Predator: 1/61 + 1/62 = 0.0325 → #1                      │
│   Alien   : 1/63 + 1/61 = 0.0323 → #2                      │
│ → fused list, top 10 kept                                  │
└───────────────────────────────────────────────────────────┘
                        │  top 10 (small enough to afford the costly step)
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 4 · RERANKER  (cross-encoder)                        │
│ Re-read the query AND each film TOGETHER, then reorder.    │
│ Stages 2–3 compared things made separately; this reads     │
│ them at the same time and asks "is THIS actually about     │
│ creatures chasing you?" — the only trustworthy relevance   │
│ score in the pipeline. Costly, so top 10 only.             │
│ → Predator, Jurassic Park, Alien, …                        │
└───────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 5 · RETURN  →  top 3 to the user                     │
│   Predator · Jurassic Park · Alien                         │
└───────────────────────────────────────────────────────────┘
```

---

## The three corrections that untangle the confusion

1. **You don't "add ranking logic" to the scores.** Vector search already returns results
   sorted by score. The order IS the ranking.

2. **RRF throws the scores away.** It uses only the rank *position* — 1st, 2nd, 3rd. You knew
   this ("fuse on rank, not score") but re-introduced scores. Drop them at the fusion door.

3. **BM25 has no "keyword table."** It reads the *same* `content` text the vectors came from and
   scores how well the query's literal words match it. Same source, different matching method.

## Why each technique exists — they are NOT interchangeable

| Technique | What it takes in | What it does | Output |
|---|---|---|---|
| **Metadata filter** | the query's hard facts | removes ineligible films | a *set* |
| **Knowledge graph** | the query's relationships | removes films whose entities don't relate right | a *set* |
| **Vector search** | soft query text | ranks by meaning | a *list* |
| **BM25 / keyword** | raw query words | ranks by literal-word match | a *list* |
| **RRF** | two ranked lists | merges them by rank position | one *list* |
| **Reranker** | one list + the query | reorders by re-reading query+doc together | one *list* |

- **BM25 and RRF are not alternatives.** RRF is the *glue* that joins BM25's list to the vector
  list. That combination is what "hybrid search" means.
- **Reranking is not fusion.** It reorders a single list; it doesn't merge lists.

## How the three (graph, vector, BM25) relate — the part people get wrong

- **The graph works ALONE, and first.** It's a filter: it outputs a *set*, not a list. It narrows,
  then steps back. It does not rank and it does not fuse.
- **Vector and BM25 work as a PAIR**, both searching the graph's narrowed set, each producing a list.
- **RRF fuses that pair — two lists, not three.** The graph never made a scorecard, so it isn't in
  the fusion. The shape is: **1 filter → 2 searches → fuse the 2 → rerank → return.**
- **Filters output sets; retrievers output lists; RRF fuses lists.** That's the whole rule. A graph
  is a filter → it doesn't enter RRF, exactly as "a graph is a filter, not a ranker" predicts.

**Advanced footnote (GraphRAG):** you *can* also let the graph **retrieve** — walk from a node to
related nodes — which turns it into a *third* list, and then RRF fuses all three. That's the graph
doing a second, harder job. Its clean primary role is the gate.

## Where MovieMotions is right now

*Rewritten 30 Aug 2026. The list below described Pass 1 and had gone badly stale — five of
the seven stages had changed state and the page still said they had not. Prose drift is the
one kind the docs gate cannot catch, because nothing in it points at a file.*

- Stage 1 metadata gate — **built.** Runtime, year and a named-film exclusion, as `WHERE`
  clauses on the film table. Extracted by the agent as tool arguments; there is still no
  separate parser, and there does not need to be.
- Stage 1b knowledge graph gate — **built (30 Aug).** Genre, actor and director as `EXISTS`
  clauses against `graph_edges`, in the same inner query as the column filters, so the
  per-source chunk quota is spent only on eligible films. Filter, then budget, then rank.
- Stage 2a vector — **built.** pgvector, arm D (header in the stored vector only).
- Stage 2b BM25 — NOT built. Retrieval is vector-only; there is no lexical arm.
- Stage 3 RRF — NOT built. Nothing to fuse until 2b exists.
- Stage 4 reranker — **built.** Cohere rerank-v3.5 over the ~50 candidates.
- Stage 5 return — **built.** Chunks collapse into films by a damped sum of each film's best
  three chunks, so breadth of evidence beats one lucky scene.

Today the system runs: **Stage 1 + 1b → Stage 2a → Stage 4 → collapse → Stage 5.**

**The gate's effect has NOT been measured cleanly, and the note that first said otherwise
was wrong.** Get Out scored 0.089 on *"tense, creatures hunting people"* across all twenty
films, and 0.626 on *"a frightening film about being trapped somewhere"* with
`genre="Horror"`. Those are two different queries. The pair says nothing about what the gate
is worth, and "same query, same corpus" was written into two documents before anyone
checked which query produced which number. The controlled version is one line of work — run
the gated query with the gate off — and until it is run, the gate is justified by argument
and not by measurement.

**The caution that comes with it.** A gate deletes rows before anything can score them, so it
is only set from facts the USER named. A genre inferred from a mood and then enforced removes
the right answer before it has a chance, which is precisely how a carried-over `max_runtime`
once deleted the only film in the catalogue about magic.

**What is still missing at this stage.** Nothing refuses to answer. Stage 5 always returns
three films, however weak the match, so an unanswerable question is indistinguishable from an
answerable one in the shape of the output — though not in the numbers, where a hopeless query
reads flat and low (0.244, 0.143, 0.124…) against an answerable one reading steep and high
(0.716, 0.395, 0.358…). A relevance floor is the next measured decision, and the obvious
threshold is already known to be wrong: a flat cut at 0.25 deletes Alien from "creatures
hunting people".

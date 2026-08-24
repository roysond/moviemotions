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

- Stage 1 metadata gate — NOT built (no query parsing yet)
- Stage 1b knowledge graph — NOT built (overkill at 20 films; a real point at scale)
- Stage 2a vector — **built**
- Stage 2b BM25 — NOT built
- Stage 3 RRF — NOT built (nothing to fuse until 2b exists)
- Stage 4 reranker — NOT built ← **highest-value next step; needs no second list**
- Stage 5 return — **built** (with one-row-per-film dedupe)

Today the system runs: **Stage 2a → dedupe → Stage 5.** Everything else is the roadmap.

**The reranker is first** because it fixes the `1/3` regression (Jurassic Park and Alien buried by
mood chunks get re-read and pulled back up) and it needs no second search to exist.

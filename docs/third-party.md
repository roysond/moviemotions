# Third-Party Service Register

> Every external service MovieMotions depends on — what it's for, what tier we're on, what data
> leaves our system to reach it, and what we owe them.
>
> **Why this file exists.** In an organisation this is called a *vendor register* or, where personal
> data is involved, a *record of processing activity* — the thing a compliance or privacy team asks
> for when someone says "can we ship this?" Without it, that question means re-reading five sets of
> terms and hoping you remember which tier you picked.
>
> Last reviewed: 15 August 2026 · Owner: Royson D'Souza

---

## 1. Service register

| Service | Used for | Tier | Cost | Auth | Rate limit | Status | Pass |
|---|---|---|---|---|---|---|---|
| **TMDB** | Canonical film metadata, cast, crew, genres, keywords, watch providers, now-playing | Developer *(non-commercial)* | Free | API key | ~50 req/sec | ✅ Active | P0 |
| **LLM provider** | Query parsing, ranking, explanation generation | — | Pay per token | API key | Per provider | ⏳ Not selected | P0 |
| **Embedding provider** | Turning text into vectors | — | Pay per token | API key | Per provider | ⏳ Not selected | P0 |
| **Wikipedia** | Long-form scene-level plot text | Public | Free | None | Be polite; set a User-Agent | 📋 Planned | P1 |
| **OMDb** | Ratings as a second opinion; a deliberate source of disagreement | Free tier | Free | API key | 1,000/day | 📋 Planned | P1 |
| **Reranker** | Reordering retrieved candidates | — | Pay per call | API key | Per provider | 📋 Planned | P1 |
| **LangSmith** | Tracing, cost and latency, eval run history | Free tier | Free | API key | Generous | 📋 Planned | P1 |
| **Open-Meteo** | Current weather at the user's location | Free, non-commercial | Free | **None** | ~10k/day | 📋 Planned | P2 |
| **IP geolocation** | Pre-filling the user's approximate location | Free tier | Free | API key | Varies | 📋 Planned | P2 |
| **Showtimes provider** | Nearby cinema listings | Sandbox *(educational)* | Free in sandbox | API key | Sandbox-limited | ⏳ Access requested | P2 |
| **AWS Bedrock** | Managed model inference | Pay-as-you-go | Per token | IAM | Per model | 📋 Planned | P3 |

---

## 2. Data flow and compliance

The column that matters most is **what leaves our system**. Everything else is bookkeeping.

| Service | What we send them | Personal data? | Commercial use today | We must attribute |
|---|---|---|---|---|
| **TMDB** | Film IDs and search terms | No | ❌ Developer tier is non-commercial | ✅ **Yes** — logo + required notice |
| **LLM provider** | The user's query text, retrieved film text | ⚠️ Query text is user-authored — treat as untrusted and potentially personal | Depends on plan | No |
| **Embedding provider** | Film synopsis text only | No | Depends on plan | No |
| **Wikipedia** | Article titles | No | ✅ Yes | ✅ **Yes** — CC BY-SA 4.0 attribution |
| **OMDb** | Film titles / IDs | No | ❌ Free tier is non-commercial | Check terms |
| **Reranker** | Query text + candidate film text | ⚠️ Query text | Depends on plan | No |
| **LangSmith** | Full traces — **including user queries** | ⚠️ Yes, by design | Depends on plan | No |
| **Open-Meteo** | **Latitude and longitude** | ⚠️ **Yes — location is personal data** | ❌ Free tier is non-commercial | Courtesy link |
| **IP geolocation** | **The user's IP address** | ⚠️ **Yes — IP is personal data under GDPR** | Depends on plan | No |
| **Showtimes provider** | **Latitude and longitude** | ⚠️ **Yes** | ❌ Sandbox is educational only | Per terms |
| **AWS Bedrock** | Prompts and retrieved text | ⚠️ Query text | ✅ Yes | No |

---

## 3. What this table is already telling us

Four things fall straight out of the rows above, before anyone writes a line of code:

**Four services will receive personal data.** IP address goes to the geolocation provider; latitude and
longitude go to the weather and showtimes providers; raw user queries go to the LLM provider and to
LangSmith. That is a real data-processing footprint and it needs consent, minimisation, and a
retention decision — not an afterthought in Pass 2.

**LangSmith is the easiest one to get wrong.** Its entire purpose is capturing everything that
happened, which by default includes the user's words. Whatever scrubbing rule we choose has to be
set deliberately, not inherited.

**Commercialising today is blocked in five places.** TMDB Developer, OMDb free, Open-Meteo free, and
the showtimes sandbox are all non-commercial tiers. That's the correct choice for a learning build —
but it means "can we ship this?" has a clear, documented answer: *not without renegotiating four
agreements.* Being able to answer that in one sentence is the entire point of this file.

**Two attributions are already owed.** TMDB requires its logo plus a specific notice. Wikipedia text
is CC BY-SA 4.0 and requires attribution wherever it appears. Both belong in an About or Credits
section from the first release, not retrofitted.

---

## 4. Where the secrets live

Every key in this register lives in **one place**: a `.env` file at the repository root, gitignored
before the first commit.

- Never in code, never in a notebook, never in a config file that gets committed
- One name per service, e.g. `TMDB_API_KEY`, `LLM_API_KEY`
- `.env.example` **is** committed — same key names, no values — so anyone cloning the repo knows
  what they need without ever seeing a secret

---

## 5. Review triggers

Revisit this file when any of these happen:

- A new external service is added
- A tier changes, or a free tier's limits change
- The project's purpose shifts toward revenue *(then every ❌ in the commercial column becomes a task)*
- Any new field of user data starts being sent outward
- Every six months regardless — terms change quietly

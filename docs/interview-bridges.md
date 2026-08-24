# MovieMotions → Interview Bridges

> **The split:** MovieMotions is where I *genuinely* build the skill — the tool, the concept, the
> judgment call are all real and mine. The interview story is told through **ProspectGuide at Centene
> (health insurance)**, because that's the professional frame the resume lives in. Same knowledge,
> regulated-domain narration.
>
> **How to read each entry:** the *Lived* line is what I actually did in MovieMotions and can defend
> in detail. The *Healthcare translation* is how that same judgment reads for PHI / member data —
> framed generically, because **Claude does not know ProspectGuide's specifics and won't invent
> them.** I plug the ProspectGuide details in myself.
>
> **Living doc** — a new row every time a MovieMotions detail maps cleanly to the regulated world.

---

## 1 · Region / data residency is a compliance gate, not a convenience

**Lived (MovieMotions).** The only reranker that runs on my free plan (Amazon Rerank 1.0,
first-party) isn't offered in my region, us-east-1 — it lives in us-west-2. A rerank call is
*stateless*, so I route only that one call cross-region and leave the rest of the stack in place.
For public TMDB data the only cost is a little latency.

**Healthcare translation.** For PHI, that same cross-region hop stops being a footnote and becomes
a gate you clear *before* writing code. Three questions: is the service **HIPAA-eligible and in your
AWS BAA** for that region; does the data **leave the permitted boundary** (the `us.` vs `global.`
inference-profile decision, now with teeth); and is the model's **vendor a Business Associate** at
all. Data classification drives region / model / vendor — before performance ever enters it.

**One-liner.** *"In a regulated domain, which region and which model aren't performance decisions —
they're compliance gates you clear before any code, because the data's classification decides them."*

---

## 2 · First-party vs third-party model = who is even *allowed* to see the data

**Lived.** Cohere's reranker is a paid AWS Marketplace product; Amazon's is first-party. On a free
plan that difference surfaced as a billing wall — the plan couldn't complete the Marketplace
subscription.

**Healthcare translation.** The exact same first-party/third-party line is a *compliance firewall*.
Sending PHI to a third-party model makes that vendor a **subprocessor touching PHI**, which legally
requires a **Business Associate Agreement** — or it's a breach. "Which vendor" is a legal question
before it's ever a quality or cost question.

**One-liner.** *"Picking between a first-party and a third-party model in healthcare isn't about
quality or price first — it's whether that vendor is contractually allowed to see the data at all."*

---

## 3 · Least privilege + separation of duties

**Lived.** The app identity (`moviemotions_app`) had `bedrock:InvokeModel` only; enabling reranking
was a *deliberate* new grant (`bedrock:Rerank`). And the app has no power to *subscribe / purchase*
(`aws-marketplace:Subscribe`) — that's an account-owner action, not an app's.

**Healthcare translation.** An app identity touching member data should hold the **minimum actions
for its job and nothing that can spend money or change access**. Separation of duties — the service
*invokes*, a human/owner *purchases and grants* — is auditable and is exactly what a security review
looks for.

**One-liner.** *"Least privilege by kind, not just amount — the service can call the model, but it
can't buy one or grant itself access. Those are separate identities on purpose."*

---

## 4 · Semantic layer — one undefined term = quietly inconsistent answers

*(This one is already a healthcare example — the sandbox data was synthetic member enrollment.)*

**Lived.** Over an 81-row enrollment file, "how many active members?" returned **75 / 72 / 69 / 62 /
60** depending on who defined "active" — and a text-to-SQL agent silently picked one, in bold, with
no caveat. Fixed with a governed view whose definition (and the reasoning) lives in its SQL
`COMMENT`, that every consumer must use.

**Healthcare translation.** This *is* the healthcare problem. Enrollment, eligibility, active
membership are precisely the terms that drive **payments and regulatory filings**. An LLM
re-deriving "active member" from column names on every query is a source of confident numbers that
disagree with each other. The fix is a **semantic layer nobody can route around** — and the hard
part isn't the first definition, it's coverage and removing the bypass.

**One-liner.** *"A natural-language interface over member data is only as trustworthy as the
definitions beneath it. 'Active member' gets defined once, governed, and made impossible to bypass —
or you ship confident numbers that quietly disagree."*

---

## 5 · Graceful degradation — availability vs quality

**Lived.** When the reranker became unreachable, `search()` catches the failure, logs one line, and
falls back to vector order. The app keeps answering.

**Healthcare translation.** A member-facing system should **degrade in quality under a vendor outage,
not go down.** That availability/SLA posture — quality is best-effort, availability is protected — is
what ops and reviewers care about, and it's a deliberate design choice, not an accident.

**One-liner.** *"External model calls are wrapped so a vendor outage costs answer quality, never
system availability."*

---

## 6 · Metrics can hide regressions in regulated AI

**Lived.** A retrieval eval passed "4 of 5" before and after a change — but one query silently went
from 3 correct answers to 1, because the metric only asked "did *any* right answer appear?" I
changed it to report `3/3` vs `1/3` so decay is visible.

**Healthcare translation.** In a governed AI system, **a coarse quality metric is a compliance risk**
— it lets a model degrade while the dashboard stays green. Model governance means metrics sensitive
enough to *see* drift, plus a fixed golden set and a recorded baseline you can defend in an audit.

**One-liner.** *"A pass rate without a definition of 'pass' hides regressions. In a regulated system,
the metric has to be sensitive enough to catch decay, or 'still passing' means nothing."*

---

## 7 · A better model exposes thin content — it doesn't fix it

**Lived.** I turned on a reranker — a strictly "smarter" retrieval stage that reads the query and
each film's text *together*. On nuanced queries it did exactly what I wanted (case 1, "creatures
chasing you," went from 1 correct answer in the top 3 to 2; case 4 from 1 to 2). But on "a father
and son separated, trying to find each other" it *dropped* the right answer, Finding Nemo, out of
the top 3. Not a bug: Finding Nemo's indexed text is abstract — "family bond," "parental love" —
and a cross-encoder reads more literally than a blurry vector. The better model didn't add the
missing signal; it *exposed* that the content never had it. And the coarse pass-count (4→3) called
this a regression while honest recall (5/7 vs 4/7) showed a net gain.

**Healthcare translation.** You cannot out-model thin source data. A stronger LLM over
sparsely-documented member records surfaces the gaps rather than filling them — and a coarse metric
misreads that as "the new model is worse," or misses it entirely. Model governance means measuring
at a grain that sees per-query behaviour, and investing in the *content* the model reads
(documentation, enrichment, structured capture) — not just buying a bigger model.

**One-liner.** *"A better model doesn't invent information that isn't in the data — it exposes where
the data is thin. The fix is better source documentation, not a bigger model; and the metric has to
be fine-grained enough to tell those two apart."*

---

## 8 · A model gateway: fast for public data, a liability for PHI

**Lived.** Every in-house AWS path to a reranker was walled — a paid Marketplace subscription the
free plan couldn't complete, then a per-region model grant the account didn't have. I routed around
all of it with OpenRouter, a provider-agnostic gateway: one endpoint, one call, and it picks a
vendor underneath. For public TMDB data that's pure upside — the fastest way to a working reranker.

**Healthcare translation.** The very thing that makes a gateway convenient — it fronts many vendors
behind one endpoint, so you don't have to think about who serves you — is what makes it a liability
for PHI. The gateway *and* whatever model it routes to both become subprocessors touching PHI, each
needing a Business Associate Agreement, and you've *reduced* your visibility into where the data
actually went. The mature call inverts the hobby call: keep the model first-party or in-VPC, even at
a quality or effort cost, precisely because you can name and contract every party that sees the data.

**One-liner.** *"A model gateway is the right answer for public data and the wrong answer for PHI —
its whole convenience is that you don't know or control which vendor served you, which is exactly
what a BAA regime forbids."*

---

*Add the next bridge here as it comes up.*

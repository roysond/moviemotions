# Databricks + Genie — a 45-minute exercise that produces real talking points

> The dataset is synthetic health-plan enrollment data with **deliberate defects**.
> The point isn't to clean it. The point is to discover something specific about
> what AI-over-data actually requires.

**File:** `member_enrollment_sample.csv` — 81 rows, 8 columns.

---

## Setup — 10 minutes

1. Sign up at `databricks.com/signup/free-edition`
2. In the workspace, find **Data** or **Catalog** → **Create table** → upload the CSV
3. Give it a name like `member_enrollment`
4. Open a **notebook** or the **SQL editor** and look at the table

That's your first Databricks navigation, same as the Bedrock console tour.

---

## Part 1 — Profile the data (this is "curation")

Run these. Each one exposes a different defect.

```sql
-- How many ways is "active" spelled?
SELECT status, count(*) FROM member_enrollment GROUP BY status ORDER BY 2 DESC;

-- How many distinct states? (there are only 10 real ones)
SELECT count(DISTINCT state) FROM member_enrollment;

-- Premiums stored as text with currency symbols
SELECT member_id, monthly_premium FROM member_enrollment
WHERE monthly_premium LIKE '$%' LIMIT 10;

-- Members appearing more than once, with different data
SELECT member_id, count(*) FROM member_enrollment
GROUP BY member_id HAVING count(*) > 1;

-- THE CONTRADICTION: flagged active, but has a termination date
SELECT member_id, status, termination_date FROM member_enrollment
WHERE upper(trim(status)) LIKE 'A%' AND termination_date <> '';
```

**What you'll find:** inconsistent casing and whitespace in `status`, states recorded
both as codes and full names, premiums as text, negative and missing premiums,
duplicate member IDs with conflicting values, and rows that claim to be active while
carrying a termination date.

**That last one is not a formatting problem. It's a contradiction in the business record.**

---

## Part 2 — Ask Genie the question

Create a **Genie space** over the table, then ask it in plain English:

> *"How many active members do we have?"*

It will give you a number. **Write it down.**

Now ask the same question three different ways yourself:

```sql
-- A: trust the status column
SELECT count(*) FROM member_enrollment
WHERE upper(trim(status)) LIKE 'A%';

-- B: active means no termination date
SELECT count(*) FROM member_enrollment
WHERE termination_date = '';

-- C: active means status active, not terminated, AND paid recently
SELECT count(*) FROM member_enrollment
WHERE upper(trim(status)) LIKE 'A%'
  AND termination_date = ''
  AND last_payment_date >= date_sub(current_date(), 45);
```

**You will get three different numbers.** All three definitions are defensible.
Different departments would each argue for a different one.

---

## Part 3 — The point

> **Genie wasn't wrong. The business hadn't decided.**

The model produced a confident, plausible number by silently picking one interpretation
of a word nobody had defined. Nothing in the data told it which one was correct — because
nothing in the data *could*.

This is what a **semantic layer** is for, and it's why "data readiness" and "text-to-SQL"
turn out to be the same problem.

---

## Part 4 — Fix it

Define the term once, in one place, and point Genie at that instead:

```sql
CREATE OR REPLACE VIEW active_members AS
SELECT * FROM member_enrollment
WHERE upper(trim(status)) LIKE 'A%'
  AND termination_date = ''
  AND last_payment_date >= date_sub(current_date(), 45);
```

Ask Genie the same question again. **One definition, one answer, every time.**

That view is a semantic layer in its simplest possible form — an agreed definition,
written down, that every consumer uses.

---

## Three talking points you'll have earned

**On data curation:**

> *"I profiled an enrollment file and found the usual things — inconsistent casing,
> states recorded both as codes and names, currency stored as text, duplicate IDs.
> But the one that mattered was records flagged active that also carried a termination
> date. That's not a formatting defect, it's a contradiction in the business record,
> and no amount of cleaning resolves it — somebody has to decide which field wins."*

**On Genie and semantic layers:**

> *"I asked Genie how many active members we had and got a confident number. Then I
> wrote the query three ways — trusting the status flag, checking for a termination
> date, and adding payment recency — and got three different counts. Genie wasn't
> wrong. The business hadn't defined the term. That's when semantic layers stopped
> being abstract for me: a natural-language interface is only as good as the
> definitions underneath it."*

**On data readiness for AI:**

> *"Text-to-SQL makes data quality problems visible in a way dashboards never did.
> A dashboard has a definition baked into it by whoever built it. An LLM writing SQL
> re-derives that definition every time, from nothing — so any term the business hasn't
> agreed on becomes a source of quietly inconsistent answers."*

---

## Why this is worth 45 minutes

You get **Databricks navigation**, a **real data-profiling pass**, **hands-on Genie**,
and a **semantic layer built to solve a problem you actually hit** — rather than one
explained to you.

And the story is domain-neutral. It's about enrollment data, not films.

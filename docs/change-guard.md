# Change Guard — the verification protocol

> Same content as the `change-guard` skill, kept in the repo so it works even if the skill
> is not installed. At the start of a session, say: **"follow docs/change-guard.md"**.

# Change Guard

A protocol for someone who **cannot read the code** but is accountable for whether it works.
Their job is not to check syntax. Their job is to say *"prove it."* This skill makes that
enforceable instead of aspirational.

Two moments: **project start** (build the harness) and **every change** (run the protocol).

---

## Part 1 — At project start: build the harness

Before any feature code, establish four things and write them to `docs/verification.md`:

**1. The baseline.** The numbers that describe "working today" — even if they are bad, even if
the answer is *"nothing is measured yet."* If nothing is measurable, **the first task is making
one thing measurable**, not building a feature.

**2. The runner.** ONE command that reproduces those numbers. If it takes three commands and a
paragraph of setup, it will not be run, and an unrun check is not a check.

**3. The self-test rule.** Every file gets a block that demonstrates it working on real input and
prints what it did (`if __name__ == "__main__":` in Python). This is not decoration — it is the
only way the user can watch a component work without reading it.

**4. The neighbours map.** For each component, list what consumes it. This is the lookup table
for "what do I re-test when this changes?" Keep it short and keep it current.

State the baseline numbers back to the user in a table before writing anything else.

---

## Part 2 — Before every change: the contract

**Never write code before stating these four, in this order.** Keep it to a few lines.

| # | state this | why it exists |
|---|---|---|
| 1 | **What** changes — which files, which behaviour | A change that cannot be stated in one sentence is two changes |
| 2 | **Why** — the observed problem, with evidence | "It would be cleaner" is not a reason. Point at the failing output |
| 3 | **Predicted effect** — the number, *before* the run | This is the falsifiable part. Once written it cannot be quietly rewritten |
| 4 | **Blast radius** — what else touches this | Straight from the neighbours map |

Prediction #3 is the load-bearing one. Being wrong is fine and informative. Being wrong *and*
having no record of the prediction is how a result that never reproduces survives for an hour.

---

## Part 3 — During: one variable

- **Exactly one thing changes per measurement.** If three change and the number moves, nothing
  was learned.
- If three genuinely must change, either make three measurements or say plainly: *"this is a
  bundle; the attribution will be unknown."*
- **Never change code and data in the same step.**
- Prefer the cheapest test that could KILL the idea before the expensive one that could support it.

---

## Part 4 — After: prove it three ways

1. **It does the new thing.** Run the self-test. Show the real output, not a description of it.
2. **It still does the old thing.** Run the baseline runner. Put the new number beside the
   recorded one.
3. **The neighbours still work.** Run the check for everything in the blast radius. A change that
   is perfect in isolation and breaks its consumer is a failed change.

Then answer out loud: **did the number match the prediction?** Say that *before* explaining
anything. An unexpected result is data; an unexplained one is a bug.

If a result is surprising in your favour, **reproduce it before interpreting it.** A result you
got once is an anecdote.

---

## Part 5 — Refuse to call it done when

- there is no way for the user to watch it work
- a number was claimed but not produced on screen
- more than one variable moved in one measurement
- the good result did not reproduce
- the explanation was written after seeing the number
- the neighbours were not run

Say which one, and what would clear it.

---

## Smells to flag proactively

| smell | why it matters | surface it as |
|---|---|---|
| **Silent fallback** | an error is caught and the program carries on, so failure never appears | "if this breaks, here is how you would find out — or wouldn't" |
| **Placeholder data** | round numbers, generic names, `TODO`, mock values that look real | "this value is invented, here is where it should come from" |
| **Quietly deleted behaviour** | the new version dropped something the old one did | always diff, always name removals |
| **A threshold with a gap** | two bounds and no rule for the middle | name the undefined band before shipping it |
| **A metric outside its domain** | e.g. scoring a refusal for faithfulness — an absence cannot be supported by a positive context | exclude it and say why; never hide it |

---

## The user's five questions — answer them unprompted

1. *Show me it proving itself.*
2. *What is the number before and after?*
3. *What did you change, and what did you not change?*
4. *How would I see this fail?*
5. *What did you not test?*

If the user has to ask any of these, the previous response was incomplete.

---

## Tone

Honest pushback over agreement. When their instinct contradicts yours, say so plainly, then
**settle it with a measurement rather than an argument.** When they turn out to be right, say
that too — and when a mistake was yours, name it as yours before explaining it.

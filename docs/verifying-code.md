# Verifying code you cannot read

> For working with me, Cursor, or Claude Code. **You do not need to read Python.**
> You need a protocol, and you already used it four times on 24 August without noticing.

---

## The reframe

**Nobody verifies code by reading it.** Not seniors, not architects. Reading finds typos; it
does not find bugs. The whole industry runs on tests, types and measurement precisely *because*
reading fails — a twenty-year engineer staring at 300 lines misses the same off-by-one you would.

So the gap is smaller than it feels. What you lack is **syntax**. What matters is **method**.

---

## You already did this

| what you did | what it caught | needed Python? |
|---|---|---|
| "Terminator 2 has no creatures hunting people" | a wrong answer scoring **0.536** — above every threshold | no · domain knowledge |
| "let's test it instead of wasting time in opinion" | a confident, wrong prediction | no · method |
| pushed back six times on a missing line break | a real broken output — it was a **display bug** in the review panel | no · you trusted the output |
| "genre is a filter, not a corpus" | a corpus worth **±0 answers** | no · design reasoning |

> **When the output is wrong and the explanation says it is fine, the output is right.**

---

## The five questions

Ask these of any tool that hands you code. None require reading a line.

**1. "Show me it proving itself."**
Every file should arrive with a way to watch it work — a self-test that runs it on real input and
prints what happened. Code with no way to run it has not been verified by anyone, including the
thing that wrote it.

**2. "What is the number before, and after?"**
`86.2% → 86.2%`. `0.08 → 0.37`. `6/6`. A change with no number attached is a hope.

**3. "What did you change, and what did you NOT change?"**
Forces a diff. You do not read 300 lines — you read the eight that moved. `git diff`.

**4. "How would I see this fail?"**
If the answer is vague, the fix is not understood. And it hands you a test.

**5. "What did you not test?"**
The honest gap. Ask it of yourself too.

---

## Four smells you can see without reading code

| smell | what it looks like | ask |
|---|---|---|
| **Silent fallback** | an error is caught and the program carries on | "if this breaks, how do I find out?" |
| **Placeholder data** | suspiciously round numbers, generic names, `TODO` | "where did this value come from?" |
| **Quietly deleted behaviour** | the new version dropped something the old one did | "what did you remove?" |
| **Two things at once** | several changes in one step | "which one moved the number?" |

That last one is the most common way real teams fool themselves. **If three things change and the
score moves, you have learned nothing.**

---

## The rule that outranks the rest

> **Change one thing. Measure. Write the prediction down first.**

Writing the prediction first is what makes it falsifiable. Once the number is on screen it is far
too easy to author the explanation that fits it — which is exactly how a 10-point improvement that
never reproduced survived for an hour.

---

## Your role, stated plainly

You are not the person who checks the syntax. **You are the person who says "prove it."**
That role does not require coding. It requires refusing to accept a claim without a number,
and trusting a wrong output over a confident explanation.

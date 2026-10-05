# Instructor solutions

Worked answers to the 20 in-notebook exercises and the end-of-notebook
checkpoint questions.

| File | Contents |
|---|---|
| [`exercises.md`](exercises.md) | The 20 exercises, in notebook order |
| [`checkpoints.md`](checkpoints.md) | The "Checkpoint" questions that close each notebook |

---

## Confidence markers

Every answer carries one, because they are not equally certain:

| | Meaning |
|---|---|
| ✅ **Verified** | Computed or determined from the code. Safe to assert. |
| 🔬 **Expected** | Depends on the model and hardware. Describes what to expect and what counts as success — **the exact numbers will differ on the day.** |
| ✍️ **Yours** | Needs your own data, judgement, or a decision only you can make. |

**Nothing here has been run against real model weights.** Everything marked 🔬
is reasoning from the code, not observation. The first time you teach this,
correct these in place — your real numbers are worth far more than my estimates.

---

## Should students see this?

Your call. The honest trade-off:

**Arguments for publishing it:** the exercises are exploratory ("run this and
see"), not a graded problem set. Self-paced learners benefit enormously. The
thing that is actually graded — the capstone — has its own
[rubric](../docs/guides/capstone-rubric.md) and cannot be copied from here.

**Arguments for withholding it:** if you grade the exercises, or you want
students to sit with the discomfort before seeing the answer.

To withhold, pick one:

```bash
# Option 1 — keep it local only
echo "solutions/" >> .gitignore && git rm -r --cached solutions

# Option 2 — keep it on a branch students do not get
git checkout -b instructor && git push -u origin instructor
# then delete solutions/ on master

# Option 3 — a separate private repository
```

The default in this repo is **published**, on the view that a student who looks
up an exploratory answer has still read it.

---

## How to use these while teaching

1. **Read the 🔬 answers before the session.** They predict where students get
   stuck, which is the part worth preparing.
2. **Do not read answers aloud.** Most exercises are "run it and see" — the
   observation is the point and saying it first destroys it.
3. **Exercises 10-1 and 03-1 are the best live demos.** Both make something
   invisible visible in about 30 seconds.
4. **Correct these as you go.** When a real run contradicts an estimate here,
   fix the file. That is the single most valuable thing you can do with this
   document.

## Time budget

Rough, assuming students work in pairs:

| Notebook | Exercises | Minutes |
|---|---|---|
| 01 | 3 | 15 |
| 02 | 1 | 10 |
| 03 | 3 | 25 |
| 04 | 2 | 25 |
| 05 | 2 | 20 |
| 07 | 2 | 35 |
| 08 | 2 | 25 |
| 09 | 1 | 45 |
| 10 | 2 | 40 |
| 11 | 1 | 45 |
| 12 | 1 | 30 |

**If you are short of time**, the ones that carry the most weight are
**09-1** (write an eval set and measure), **10-1** (the hallucinated
`Observation:`) and **10-2** (chain vs agent). Those three teach the
discriminating skills.

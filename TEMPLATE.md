# How this repository was built, and how to build another one

A blueprint for producing a project like this with Claude Code: the prompts, the
structure, and the conventions that did most of the work.

**An honest note first.** There was no single master prompt. This repository grew
over seven conversational turns from a rough two-sentence idea. What follows is
that history, plus the *distilled* prompt I would use to reproduce it in one go —
which is more useful than the transcript.

---

## 1. The prompt that actually started it

Verbatim, typos and all:

> okay, i need to delete all this file on this directory and create a complete
> hands on workshop with python codes, preferably jupyternotebook on using quen
> open weight models for my students, they should know everything about LLMs and
> quen models, and the learning outcomes are they will be able to how to
> download, open weight models, from where and how to get started with their
> first simple demo project, for any daily life activity, all the codes needs to
> be there and a github page with in depth tutorial about agentic AI, and open
> weight models, they have 3-4 days to master the material; you can decide
> yourself the best way to create this; i'm giving you rough idea

That produced 93 files in one turn. Three things in it did the heavy lifting:

| Phrase | Why it mattered |
|---|---|
| "learning outcomes are they will be able to…" | Gave a **goal to design backwards from**, not a topic list |
| "3-4 days" | A **budget**. Forces scoping decisions instead of endless breadth |
| "you can decide yourself the best way" | Explicit **authority to make architectural calls** without asking |

The vagueness was not a problem. The *constraints* were what made it work.

---

## 2. The prompt I would use now

Fill in the bracketed parts. This is the distilled version of everything the
seven turns eventually asked for.

```text
Build a complete, self-contained [KIND OF PROJECT] on [TOPIC] for [AUDIENCE].

GOAL
By the end, they will be able to:
1. [capability]
2. [capability]
3. [capability]
Design backwards from those. If something does not serve them, leave it out.

CONSTRAINTS
- Time budget: [N days / N hours]
- They have: [hardware, prior knowledge, budget, network access]
- Must work with: [no paid APIs / offline / a specific stack]
- Must NOT require: [things you can't assume]

DELIVERABLES
- [notebooks / scripts / docs / a site] — say which, and roughly how many
- A [GitHub Pages site / README] as the front door
- Sample data so nothing is hypothetical
- Tests that run with no [GPU / network / API key]

HOW I WANT IT BUILT
- Decide the architecture yourself. Don't ask me to choose between options you
  can evaluate better than I can.
- Generate artifacts from reviewable sources. Notebooks from .py files,
  diagrams from code — never hand-edited JSON or binary blobs.
- Everything verifiable should be verified in CI: tests, link checks, and a
  check that generated files are in sync with their sources.
- Include diagrams. Author them, don't download them.
- Be honest about what you could not verify, in the final report and in the
  repo itself.

WORKING AGREEMENT
- Work on branch [NAME], commit in logical chunks, open a PR when done.
- Before telling me something works, check it. Run the tests. Render the
  images and look at them. Fetch the URLs.
- If you find a real problem with what I asked for, say so in a sentence and
  keep building under a stated assumption.
```

### Why each clause is there

- **"Design backwards from those"** — without it you get a topic survey, not a
  course.
- **"Don't ask me to choose"** — otherwise you spend the session answering
  questions instead of getting a repo.
- **"Generate artifacts from reviewable sources"** — the single highest-leverage
  instruction in the whole prompt. See §4.
- **"Render the images and look at them"** — the models-checking-their-own-work
  clause. In this project it caught an agent-loop diagram whose arrows taught
  the wrong thing, and a first attempt at theming that rendered every diagram
  solid black.
- **"Be honest about what you could not verify"** — otherwise "done" is
  ambiguous. Here, nothing was ever run against real model weights, and every
  report said so.

---

## 3. The follow-up turns, in order

Each of these was one short message. The order matters: **structure first,
polish later.**

| # | Asked for | Produced |
|---|---|---|
| 1 | The workshop (above) | 93 files: 12 notebooks, library, docs site, tests |
| 2 | "push it to my github" | Merged to master |
| 3 | "enable github pages for me" | A failed attempt, then honest documentation of why it can't be automated |
| 4 | "add visuals, flowcharts… you can decide yourself" | 18 generated SVG diagrams |
| 5 | "link all the notebooks to google colab" | Badges + the bootstrap that makes them actually run |
| 6 | "separate reading material… kind of a handbook" | 6-chapter, 10k-word handbook + 13 more diagrams |
| 7 | this file | The blueprint |

**The lesson:** do not try to specify all of this up front. Get a working
skeleton, then add one dimension per turn. Each addition was cheap *because* the
skeleton already had the conventions in place.

---

## 4. The conventions that carried the project

These are transferable to any project, not just courses.

### Generate artifacts from reviewable sources

| Artifact | Source | Why |
|---|---|---|
| `notebooks/*.ipynb` | `notebook_src/*.py` | Notebook JSON makes every diff unreadable |
| `docs/assets/diagrams/*.svg` | `tools/make_diagrams.py` | Hand-drawn images drift and can't be diffed |

Both have a `--check` mode that fails CI when the artifact is stale. Without
that, generated files rot within a week.

### Make every claim checkable by a machine

```
tools/nbbuild.py --check        notebooks match their sources
tools/make_diagrams.py --check  diagrams match the generator
tools/check_notebooks.py        every code cell parses; images resolve;
                                Colab badges point at their own notebook
tools/check_links.py            190 relative links and image paths resolve
pytest                          118 tests, no GPU/network/API key
```

Two of these came from bugs that had already shipped. **When you find a class of
error, add the check that catches it**, then prove the check works by
introducing the bug deliberately and watching it fail.

### Tests that need nothing

Every test runs with no model, no GPU, no network. CI installs `pytest`, `ruff`
and `numpy` and nothing else. That constraint keeps CI at ~15 seconds and forces
the interesting logic to be separable from the model call — which is good design
anyway.

The trick for testing LLM code: a **scripted fake client** that returns canned
responses. `tests/conftest.py` has one in 30 lines, and it tests the agent loop
including error recovery, parallel tool calls and the step limit.

### One drawing kit, many diagrams

`tools/svgkit.py` is ~250 lines: boxes, arrows, text, a palette, and a
light/dark colour contract. Thirty-one diagrams share it, so they look like a
set rather than thirty-one separate afternoons.

It also *enforces* correct use — `box()` raises if you pass a colour role into
the label slot, because that mistake renders silently wrong.

### Say what you did not verify

Every report in this project ended with the same paragraph: no notebook has been
run against real model weights, because the build container has no GPU. That
sentence is worth more than any amount of confident summary.

---

## 5. The structure, and why

```
README.md              front door: what, why, how to start
TEMPLATE.md            this file

notebooks/             the artifact people open          <- generated
notebook_src/          the source you actually edit
src/<package>/         a small, readable library the notebooks import
data/                  sample data so nothing is hypothetical
tests/                 runs with no model, GPU or network
scripts/               check_env · download · smoke_test
tools/                 generators and validators
docs/                  the GitHub Pages site
  index.md             landing
  days/                the time-boxed plan
  guides/              reference material
  handbook/            standalone background reading
  assets/diagrams/     generated figures
.github/workflows/     ci.yml · pages.yml
```

Four decisions worth stealing:

1. **A real library, not notebook cells.** Logic lives in `src/` where it can be
   tested and reused. Notebooks import it. This is why 118 tests exist at all.
2. **Three tiers of writing.** Notebooks (do), guides (look up), handbook
   (understand). Different jobs, different registers, cross-linked.
3. **Sample data in the repo.** `data/` has notes, expenses, a pantry and a fake
   inbox — including a phishing email so the prompt-injection lesson is real.
4. **Scripts for the boring parts.** `check_env.py` and `smoke_test.py` turn
   "it doesn't work on my machine" into a diagnostic paste.

---

## 6. Adapting it

The structure is domain-independent. Substitutions:

| This project | A data-science course | An internal onboarding repo | A library's docs |
|---|---|---|---|
| Qwen models | pandas/sklearn | your services | your API |
| `src/qwen_workshop/` | shared helpers | shared client code | the library |
| `data/` samples | a teaching dataset | a seeded dev database | example payloads |
| 12 notebooks | 12 notebooks | runbooks | tutorials |
| handbook | statistics background | architecture overview | concepts guide |
| `smoke_test.py` | the same | "is my access working?" | quickstart check |

What transfers unchanged: generated artifacts, `--check` in CI, dependency-free
tests, one drawing kit, and the honesty clause.

What to change: the day structure only makes sense for time-boxed teaching. For
a library, replace `days/` with a task-oriented index.

---

## 7. What to watch for

**Claude will make architectural decisions if you let it — and it should.**
Asking it to choose between six options it can evaluate better than you is how
sessions get spent on deliberation instead of output. But *check the decisions
it reports*, because they are load-bearing.

**Verification is the part to be strict about.** Almost every real bug in this
project surfaced from looking rather than assuming: rendering the diagrams,
introducing a deliberate typo to prove a check works, curling the URLs. Ask for
that explicitly; it is not free.

**Some things cannot be automated, and it's better to hear so.** Enabling GitHub
Pages needs admin permission that no workflow token can hold. The useful outcome
was not a clever workaround — it was one documented sentence and a 15-second
click.

**Scope creep is the real risk.** Each turn here added exactly one dimension.
"Add visuals, and Colab, and a handbook, and tests" in a single message would
have produced a worse version of all four.

---

## 8. The short version

1. State **outcomes**, a **budget**, and grant **authority to decide**.
2. Demand **generated artifacts from reviewable sources**.
3. Demand **CI checks for everything checkable** — and new checks for every bug
   found.
4. Demand **tests that need nothing**.
5. Demand it **look at its own output** before claiming success.
6. Demand **honesty about what was not verified**.
7. Add **one dimension per turn**, structure before polish.

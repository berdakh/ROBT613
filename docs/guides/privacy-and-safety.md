---
title: Privacy and safety
parent: Guides
nav_order: 5
---

# Privacy and safety
{: .no_toc }

Running models locally is what makes using your own data reasonable. It does not
make it automatically safe.

1. TOC
{:toc}

---

## What local actually guarantees

A model running on your machine sends nothing anywhere. That is a real and
strong guarantee, and it is the main reason to prefer open weights for personal
or institutional data.

**What it does not guarantee:**

| Risk | Still present locally? |
|---|---|
| Your prompt reaching a third party | ❌ eliminated |
| Your data in someone's training set | ❌ eliminated |
| The model inventing facts | ✅ unchanged |
| The model leaking data *to your own screen* | ✅ unchanged — think about who is watching |
| A tool exfiltrating data | ✅ **very much present** |
| Prompt injection | ✅ **very much present** |
| Logs and caches on disk | ✅ present — and often forgotten |

The third-party risk goes away. The application-security risks do not.

## Before you point it at real data

A short checklist:

- [ ] **Do you have the right to use this data this way?** Student records,
      patient data and customer data usually carry obligations that do not
      disappear because the processing is local.
- [ ] **Where do your logs go?** If you log prompts to debug an agent, you have
      just created a second copy of the sensitive data, often less protected
      than the first.
- [ ] **Is the index on an encrypted disk?** Your RAG index contains your
      documents in plain text — `data/.index/notes.json` is readable by anyone
      with the file.
- [ ] **Who else uses this machine?** A shared lab computer is not private.
- [ ] **Are you about to commit it?** `data/.index/` and `outputs/` are in
      `.gitignore` for a reason. Check `git status` before committing.

{: .warning }
> The RAG index is a plain-text copy of your documents. If the source documents
> are sensitive, the index is equally sensitive. Treat it the same way.

## The three security rules that matter most

### 1. Never execute model output

No `eval()`, no `exec()`, no `subprocess(shell=True)`, no f-string SQL. Not
behind a regex filter, not "just for the demo".

The workshop's `calculate` tool parses an AST and permits only arithmetic node
types. That is ~20 lines and it is the difference between a calculator and a
remote code execution vulnerability. Read it in
`src/qwen_workshop/demo_tools.py`.

### 2. Least privilege for every tool

An agent can only do what its tools allow. This is the entire security model,
and it is a good one — but only if you use it.

```python
# The containment check that makes a file tool safe
candidate = (ROOT / filename).resolve()
if not candidate.is_relative_to(ROOT.resolve()):
    raise ToolError("access denied")
```

Without that, `../../../../etc/passwd` works. With it, the agent physically
cannot read outside the folder no matter what it is persuaded to try.

Ask of every tool: *if an attacker controlled the arguments, what is the worst
outcome?* If the answer is bad, constrain the tool rather than trusting the model.

### 3. Assume prompt injection succeeds

Any text the model reads — a document, an email, a web page, a filename — can
contain instructions. Defences that rely on the model being sensible are not
security controls.

Design so that a successful injection does not matter:

- The agent has no tool that can send data anywhere.
- Irreversible actions require a human `confirmed="yes"`.
- Retrieved content goes in a user message, clearly delimited, never the system prompt.
- Every run is logged, so you can investigate afterwards.

Notebook 10 has a live injection demonstration. Run it.

## Hallucination is a safety issue

A confidently wrong answer about a dosage, a deadline or a legal requirement can
cause real harm. Mitigations, in order of effectiveness:

1. **RAG with citations**, so every claim is traceable to a source (notebook 09).
2. **An explicit refusal instruction**, plus a retrieval score threshold — the
   model must be *allowed* to say "I don't know".
3. **Verify citations automatically.** `cited_sources()` in this workshop checks
   that the model only cited chunks that were actually retrieved. Cheap, and it
   catches a real failure mode.
4. **Tools instead of recall** for anything computable.
5. **Show your work.** Surface which tools ran and which sources were used, so
   the user can judge.

{: .note }
> A small model with good retrieval and honest refusals is far safer than a
> large model answering from memory. Capability and safety are not the same axis.

## Things not to build

Some applications are a bad idea regardless of how carefully you engineer them:

- **Medical, legal or financial advice** presented as authoritative. Information
  and summarisation, with sources and caveats, is a different thing from advice.
- **Automated decisions about people** — grading, hiring, discipline — without a
  human who can review and overturn.
- **Anything that acts irreversibly without confirmation** — sending, paying,
  deleting, publishing.
- **Impersonation.** Generating text as a named real person is a
  misrepresentation problem, not a technical one.
- **Surveillance of individuals**, including classmates or colleagues, however
  the task is framed.

For a student capstone, the line is simple: **if a wrong answer could hurt
someone and the system cannot be checked by a human, build something else.**

## Data hygiene for this repository

Already handled in `.gitignore`:

```
data/.index/        # your RAG index - contains your documents
outputs/            # fine-tuned adapters - can memorise training data
*.gguf, *.safetensors
.env                # any keys
```

If you add your own notes to `data/notes/`, consider whether you want them in
git at all. A `data/private/` folder, gitignored, is a good habit.

## Further reading

- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) —
  the standard reference for LLM security. Short and practical.
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) —
  for institutional deployments.

"""Memory for agents: what to keep, what to drop, and what to write down.

The problem in one paragraph
----------------------------
A model has no memory. Every call is stateless, and the only thing it knows is
what is in the messages you send. "Memory" is therefore not a model feature at
all - it is **your code deciding what goes into the next request**. That makes
it an engineering problem with an engineering answer, and the answer is forced
by one hard constraint: the context window is finite, and you pay for every
token in it, twice (latency and money).

So memory is a budget problem. Four strategies, in increasing order of
sophistication and of how badly they can bite you:

======================  ======================  ==========================
Strategy                Keeps                   Loses
======================  ======================  ==========================
:class:`ConversationBuffer`  everything         nothing, until it overflows
:class:`WindowedMemory`      the last N tokens  the beginning, silently
:class:`SummarisingMemory`   a summary + recent detail in the summary
:class:`FactStore`           what you chose     everything you did not choose
======================  ======================  ==========================

The last row is the important one. A fact store is the only strategy where
*you* decide what matters instead of letting recency decide, and it is the
only one that survives a restart. It is also the only one that can be wrong in
a way that persists - a bad fact written today is still there next month.

None of these is "long-term memory" in the sense people usually mean. That is
retrieval (notebook 09) pointed at the conversation instead of at documents,
and :class:`FactStore` is deliberately the seam where you would plug it in.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ConversationBuffer",
    "FactStore",
    "Scratchpad",
    "SummarisingMemory",
    "WindowedMemory",
    "estimate_tokens",
    "format_messages",
]


# --------------------------------------------------------------------------
# Counting what you are about to pay for
# --------------------------------------------------------------------------


def estimate_tokens(text: str, tokenizer=None) -> int:
    """Roughly how many tokens `text` costs.

    With a real tokenizer this is exact. Without one it uses ~4 characters per
    token, which is close for English and **badly wrong** for Kazakh, Russian,
    Arabic or Chinese - notebook 01 measured the gap. Budgeting with the
    heuristic on non-English text will silently overflow your window, so pass
    the tokenizer when you have it.
    """
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return max(1, len(text) // 4)


def _message_tokens(message: dict, tokenizer=None) -> int:
    # Every message also carries role markers and separators in the chat
    # template. Four tokens is the usual allowance; it keeps the estimate on
    # the safe side rather than the optimistic one.
    return estimate_tokens(str(message.get("content") or ""), tokenizer) + 4


def format_messages(messages: list[dict], limit: int = 80) -> str:
    """A readable one-line-per-turn dump. For looking at, not for sending."""
    lines = []
    for message in messages:
        content = str(message.get("content") or "").replace("\n", " ")
        if len(content) > limit:
            content = content[: limit - 1] + "…"
        lines.append(f"{message['role']:>9}: {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 1. Keep everything
# --------------------------------------------------------------------------


@dataclass
class ConversationBuffer:
    """The whole conversation, in order. The right default.

    Do not skip past this one looking for something cleverer. Most assistants
    never exceed their window, and every strategy below loses information that
    this one keeps. Add complexity when you measure a problem, not before.
    """

    messages: list[dict] = field(default_factory=list)
    system: str | None = None

    def add(self, role: str, content: str) -> ConversationBuffer:
        self.messages.append({"role": role, "content": content})
        return self

    def add_user(self, content: str) -> ConversationBuffer:
        return self.add("user", content)

    def add_assistant(self, content: str) -> ConversationBuffer:
        return self.add("assistant", content)

    def render(self, tokenizer=None) -> list[dict]:
        """The messages to send, system prompt first."""
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + list(self.messages)

    def token_count(self, tokenizer=None) -> int:
        return sum(_message_tokens(m, tokenizer) for m in self.render())

    def __len__(self) -> int:
        return len(self.messages)


# --------------------------------------------------------------------------
# 2. Keep the most recent
# --------------------------------------------------------------------------


@dataclass
class WindowedMemory(ConversationBuffer):
    """Keep the newest messages that fit in `budget` tokens; drop the rest.

    Cheap, bounded, and it has a failure mode worth seeing rather than reading
    about: the agent forgets the beginning of the conversation, which is
    usually where the user stated the task. :attr:`dropped` counts what went,
    so the loss is at least visible - do not build this without that counter.

    `keep_first` pins the opening turns, which is the standard mitigation: the
    user's actual request survives even when the middle is evicted.
    """

    budget: int = 2000
    keep_first: int = 0
    dropped: int = 0

    def render(self, tokenizer=None) -> list[dict]:
        head = [{"role": "system", "content": self.system}] if self.system else []
        spent = sum(_message_tokens(m, tokenizer) for m in head)

        pinned = self.messages[: self.keep_first]
        rest = self.messages[self.keep_first :]
        spent += sum(_message_tokens(m, tokenizer) for m in pinned)

        # Walk backwards from the newest message, taking what fits.
        kept: list[dict] = []
        for message in reversed(rest):
            cost = _message_tokens(message, tokenizer)
            if spent + cost > self.budget:
                break
            kept.append(message)
            spent += cost
        kept.reverse()

        self.dropped = len(rest) - len(kept)
        return head + pinned + kept


# --------------------------------------------------------------------------
# 3. Compress the old, keep the new
# --------------------------------------------------------------------------


@dataclass
class SummarisingMemory(ConversationBuffer):
    """When over budget, summarise the oldest turns into one system message.

    This is what "context compaction" means in the tools you use. It buys a
    much longer effective conversation, and it costs you the thing summaries
    always cost: **detail you did not know you would need**. A summary is
    lossy by construction, and the loss is chosen by whatever wrote it.

    `summarise` is any callable taking the messages to compress and returning
    one string. In a notebook that is usually a model call; in a test it is a
    deterministic stub, which is how the behaviour below is verified.
    """

    budget: int = 2000
    keep_recent: int = 4
    summarise: Callable[[list[dict]], str] | None = None
    summary: str = ""
    compactions: int = 0

    def _over_budget(self, tokenizer=None) -> bool:
        return self.token_count(tokenizer) > self.budget

    def render(self, tokenizer=None) -> list[dict]:
        head = [{"role": "system", "content": self.system}] if self.system else []
        if self.summary:
            head.append({"role": "system", "content": f"Earlier conversation:\n{self.summary}"})
        return head + list(self.messages)

    def token_count(self, tokenizer=None) -> int:
        return sum(_message_tokens(m, tokenizer) for m in self.render(tokenizer))

    def compact(self, tokenizer=None) -> bool:
        """Summarise old turns if over budget. True if anything was compacted.

        Call it after each turn. It is deliberately a separate step rather than
        magic inside `render`, because compaction costs a model call and you
        should be able to see where that happens.
        """
        if self.summarise is None or not self._over_budget(tokenizer):
            return False
        if len(self.messages) <= self.keep_recent:
            # Everything left is recent. Compacting further would delete the
            # turns the next reply depends on, which trades a slow agent for a
            # broken one.
            return False

        old = self.messages[: -self.keep_recent]
        self.messages = self.messages[-self.keep_recent :]
        if self.summary:
            # Fold the previous summary in rather than appending to it. An
            # earlier version concatenated, and on a long conversation the
            # summary grew until it was larger than the transcript it was
            # supposed to be compressing - compaction that costs more context
            # than it saves. A real summariser must handle this message too.
            old = [{"role": "system",
                    "content": f"Summary of the conversation so far: {self.summary}"}] + old
        self.summary = self.summarise(old).strip()
        self.compactions += 1
        return True


# --------------------------------------------------------------------------
# 4. Write it down on purpose
# --------------------------------------------------------------------------


@dataclass
class FactStore:
    """Explicit, durable facts the agent chooses to keep.

    The other three strategies answer "what did we just say?". This one
    answers "what is true about this user?", which is a different question and
    the one that makes an assistant feel like it knows you.

    Two properties the others do not have:

    * **It survives the process.** :meth:`save` / :meth:`load` is a file.
    * **It is inspectable and editable.** You can print it, correct it, and
      delete a fact. Try doing that with a summary.

    The cost is that something has to decide what is worth storing, and that
    something is usually the model. A model that stores "the user is annoyed"
    has written a fact that will still be there in March.
    """

    facts: dict[str, str] = field(default_factory=dict)
    max_facts: int = 100

    def remember(self, key: str, value: str) -> str:
        key = key.strip().lower()
        if not key:
            raise ValueError("fact key cannot be empty")
        if key not in self.facts and len(self.facts) >= self.max_facts:
            raise ValueError(
                f"fact store is full ({self.max_facts}). Forget something first - "
                "an unbounded memory is a context-window bug waiting to happen."
            )
        existed = key in self.facts
        self.facts[key] = value
        return f"{'updated' if existed else 'stored'} {key!r}"

    def recall(self, key: str) -> str | None:
        return self.facts.get(key.strip().lower())

    def forget(self, key: str) -> bool:
        return self.facts.pop(key.strip().lower(), None) is not None

    def search(self, query: str) -> dict[str, str]:
        """Substring match over keys and values. Honest about what it is.

        Real systems embed the facts and search semantically (notebook 08).
        This is the version you can read in one line, and it is enough until
        you have a few hundred facts.
        """
        needle = query.strip().lower()
        return {
            k: v for k, v in self.facts.items()
            if needle in k or needle in v.lower()
        }

    def as_system_message(self, limit: int | None = None) -> dict | None:
        """The facts, as one system message to prepend. None if empty."""
        if not self.facts:
            return None
        items = list(self.facts.items())[:limit] if limit else list(self.facts.items())
        body = "\n".join(f"- {k}: {v}" for k, v in items)
        return {"role": "system", "content": f"What you know about this user:\n{body}"}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.facts, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> FactStore:
        file = Path(path)
        if not file.exists():
            return cls()
        return cls(facts=json.loads(file.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.facts)


# --------------------------------------------------------------------------
# Working memory, within a single run
# --------------------------------------------------------------------------


@dataclass
class Scratchpad:
    """Notes an agent takes during one task. Dies when the task ends.

    Distinct from everything above, and the distinction matters: a scratchpad
    is *working* memory. "I already checked the pantry and found no eggs" is
    worth keeping for the next three steps and worthless next week. Storing it
    in a :class:`FactStore` is the mistake this class exists to prevent.

    It is also how you stop an agent repeating a failed call: write the failure
    down, then put the scratchpad in the prompt.
    """

    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> Scratchpad:
        self.notes.append(text.strip())
        return self

    def render(self) -> str:
        if not self.notes:
            return "(nothing yet)"
        return "\n".join(f"{i}. {n}" for i, n in enumerate(self.notes, 1))

    def as_message(self) -> dict:
        return {"role": "system", "content": f"Your notes so far:\n{self.render()}"}

    def clear(self) -> Scratchpad:
        self.notes.clear()
        return self

    def __len__(self) -> int:
        return len(self.notes)

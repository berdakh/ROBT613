"""A scripted stand-in for the model, so agent logic can be run without one.

Why this exists
---------------
An agent is two things tangled together: a **model** that decides, and a
**program** that executes those decisions, keeps the transcript, enforces the
step limit and handles tool errors. Nearly every agent bug students hit is in
the second part - and the second part is ordinary Python that deserves
ordinary tests.

A real model makes that impossible. It is slow, it needs a server, and it is
non-deterministic, so a failing run may not reproduce. So we replace it with a
script::

    from qwen_workshop.scripted import ScriptedModel, calls, says

    model = ScriptedModel([
        calls("list_pantry"),
        calls("expiring_soon", {"within_days": 3}),
        says("Your yoghurt expires tomorrow."),
    ])
    run = Agent(*model.as_client(), registry).run("what is expiring?")

The agent cannot tell the difference: :class:`ScriptedClient` implements the
small slice of the OpenAI client surface that
:func:`qwen_workshop.client.complete` touches.

This is not only a teaching device. It is how you would test a real agent:
record what the model did, replay it, and assert on what your code does with
it. ``model.requests`` keeps every request the agent sent, so you can also
assert on the transcript the agent built - which is where the subtle bugs are.

What it cannot do
-----------------
It cannot tell you whether the model would *actually* make those decisions.
A green scripted test means your loop is correct, not that your agent works.
Notebook 20 is about the other half.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ScriptedClient",
    "ScriptedModel",
    "ScriptedText",
    "ScriptedTurn",
    "calls",
    "says",
]


# --------------------------------------------------------------------------
# The script: a list of turns, each either "answer" or "call these tools"
# --------------------------------------------------------------------------


@dataclass
class ScriptedTurn:
    """One model turn. Either `content` (a final answer) or `tool_calls`."""

    content: str | None = None
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.content is None and not self.tool_calls:
            raise ValueError("a ScriptedTurn needs content, tool_calls, or both")
        # Note that `says("")` is deliberately legal. A model that returns an
        # empty reply is a real failure, and being able to script it is how
        # you write a regression test for how your loop handles it.


def says(content: str) -> ScriptedTurn:
    """The model answers in plain text and stops. Ends the run."""
    return ScriptedTurn(content=content)


def calls(name: str, arguments: dict | None = None, *more: tuple[str, dict],
          thought: str | None = None) -> ScriptedTurn:
    """The model asks for one or more tool calls.

    Args:
        name: tool to call.
        arguments: its arguments. Omit for a tool that takes none.
        *more: extra ``(name, arguments)`` pairs for a parallel call.
        thought: optional text the model emits alongside the call. Real models
            often do; code that assumes `content` is empty whenever there are
            tool calls breaks on them.
    """
    pairs = [(name, dict(arguments or {}))]
    pairs += [(n, dict(a or {})) for n, a in more]
    return ScriptedTurn(content=thought, tool_calls=pairs)


# --------------------------------------------------------------------------
# The client: enough of the OpenAI SDK's shape to fool `complete()`
# --------------------------------------------------------------------------


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, index: int, name: str, arguments: dict) -> None:
        self.id = f"call_{index}"
        self.type = "function"
        # Real servers send arguments as a JSON *string*, not a dict. Keeping
        # that detail is the point: code that forgets to parse it passes
        # against a sloppy fake and fails against a server.
        self.function = _Fn(name, json.dumps(arguments))


class _Message:
    def __init__(self, turn: ScriptedTurn, call_offset: int) -> None:
        self.role = "assistant"
        self.content = turn.content
        self.tool_calls = [
            _ToolCall(call_offset + i, name, args)
            for i, (name, args) in enumerate(turn.tool_calls)
        ] or None


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.choices = [_Choice(message, finish_reason)]


class ScriptedClient:
    """Replays a list of :class:`ScriptedTurn` as if it were a model.

    Implements only ``client.chat.completions.create(**kwargs)``, which is all
    :func:`qwen_workshop.client.complete` calls.
    """

    def __init__(self, turns: list[ScriptedTurn], *,
                 exhausted: str = "(script exhausted)") -> None:
        self.turns = list(turns)
        self.exhausted = exhausted
        self.requests: list[dict] = []
        self._served = 0
        self._calls_issued = 0
        # The SDK's shape is client.chat.completions.create(...), so `chat`
        # needs a `completions` that has `create`. Point both at self.
        self.chat = type("_Chat", (), {"completions": self})()

    def create(self, **kwargs: Any) -> _Response:
        self.requests.append(kwargs)
        if self._served >= len(self.turns):
            # Running off the end of the script is itself informative: it means
            # the agent took more steps than you expected. Answer plainly so
            # the run terminates instead of looping.
            return _Response(_Message(says(self.exhausted), 0), "stop")
        turn = self.turns[self._served]
        self._served += 1
        message = _Message(turn, self._calls_issued)
        self._calls_issued += len(turn.tool_calls)
        return _Response(message, "tool_calls" if turn.tool_calls else "stop")

    # -- inspection ---------------------------------------------------------

    @property
    def turns_served(self) -> int:
        """How many scripted turns the agent actually consumed."""
        return self._served

    @property
    def exhausted_script(self) -> bool:
        """True if the agent asked for more turns than the script had."""
        return len(self.requests) > len(self.turns)

    def last_messages(self) -> list[dict]:
        """The transcript as of the most recent request. Empty before the run."""
        return list(self.requests[-1]["messages"]) if self.requests else []

    def tools_offered(self) -> list[str]:
        """Tool names the agent offered the model on its first request.

        Catches a whole family of bugs at once: an empty list means the
        registry never reached the model, however good the tools are.
        """
        if not self.requests:
            return []
        return [t["function"]["name"] for t in self.requests[0].get("tools") or []]


@dataclass
class ScriptedModel:
    """A script plus the fake backend that goes with it.

    ``Agent`` wants ``(client, backend)``, so::

        model = ScriptedModel([says("hi")])
        agent = Agent(*model.as_client(), registry)
    """

    turns: list[ScriptedTurn]
    model_name: str = "scripted"

    def as_client(self):
        """Return ``(client, backend)``, ready to hand to ``Agent``."""
        from .client import Backend

        client = ScriptedClient(self.turns)
        backend = Backend(
            name="scripted",
            base_url="scripted://local",
            api_key="none",
            model=self.model_name,
        )
        # Keep a handle so callers can inspect requests after the run.
        self.client = client
        return client, backend


class ScriptedText:
    """A scripted ``messages -> text`` callable, for :class:`~qwen_workshop.agent.ReActAgent`.

    ``ReActAgent`` takes any callable that maps a transcript to assistant text,
    so scripting it needs no fake client at all::

        generate = ScriptedText([
            "Thought: I need the date.\\nAction: current_date\\nAction Input: {}",
            "Thought: I now know the final answer\\nFinal Answer: It is Tuesday.",
        ])
        run = ReActAgent(generate, registry).run("what day is it?")

    Like :class:`ScriptedClient` it records what it was asked, so you can
    assert on the prompt the agent built - including that the Observation it
    appended is the one the tool really returned.
    """

    def __init__(self, replies: list[str], *,
                 exhausted: str = "Thought: I am out of script\nFinal Answer: (script exhausted)") -> None:
        self.replies = list(replies)
        self.exhausted = exhausted
        self.prompts: list[list[dict]] = []
        self._served = 0

    def __call__(self, messages: list[dict]) -> str:
        self.prompts.append([dict(m) for m in messages])
        if self._served >= len(self.replies):
            return self.exhausted
        reply = self.replies[self._served]
        self._served += 1
        return reply

    @property
    def turns_served(self) -> int:
        return self._served

    def last_prompt_text(self) -> str:
        """The most recent prompt flattened to text - what the model really saw."""
        if not self.prompts:
            return ""
        return "\n".join(f"[{m['role']}] {m.get('content', '')}" for m in self.prompts[-1])

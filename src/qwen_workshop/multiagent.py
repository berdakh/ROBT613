"""Several agents, and whether you needed them.

Start with the conclusion
-------------------------
**Most multi-agent systems should have been one agent with more tools.** Every
handoff is a lossy summary: agent A knows things it did not write down, and
agent B gets the summary instead of the knowledge. You have replaced a
function call with a game of telephone, and you are paying a model call for
each round of it.

Multiple agents genuinely help in three situations, and they are narrower than
the literature suggests:

1. **Different tools, and the sets must not mix.** The agent that can read your
   email should not also be the one that can send money. Separation here is a
   security boundary, not an architecture preference - and it is the one
   reason on this list that nothing else can provide.
2. **Different context, and one window cannot hold both.** Four specialists
   each reading a different 50-page document is four context windows. One
   agent reading all four is a truncation bug.
3. **Adversarial roles.** A critic that has not seen the writer's reasoning
   catches things the writer cannot, precisely because it lacks that context.
   Here the information loss *is* the mechanism.

If your reason is "it seems more organised", you want functions.

What is actually new
--------------------
Only three things, which is why this module is short:

* **Routing** - choosing who handles a task (:class:`Router`).
* **A handoff format** - what crosses the boundary (:class:`Message`).
* **A termination rule** - when the exchange stops (every loop below has one,
  because the characteristic multi-agent failure is two agents being polite at
  each other until the budget runs out).

A :class:`Specialist` is anything that maps a task string to a result string,
so a specialist can be an :class:`~qwen_workshop.agent.Agent`, a plain Python
function, or another whole team. That is the useful part of the abstraction
and roughly the whole of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "CriticLoop",
    "Message",
    "Orchestrator",
    "Router",
    "Specialist",
    "Team",
]


@dataclass
class Message:
    """One handoff. The thing that crosses an agent boundary.

    Logging these is not optional. In a single agent the transcript tells you
    what happened; in a team the transcript is scattered across members, and
    this is the only record of what they actually told each other - which is
    where multi-agent systems go wrong.
    """

    sender: str
    recipient: str
    content: str
    kind: str = "task"  # task | result | error | critique

    def __str__(self) -> str:
        # The arrow always points sender -> recipient. An earlier version
        # flipped it for results, which made every reply read backwards in the
        # trace; the message kind is a tag, not a direction.
        tag = {"task": "task", "result": "done", "error": "ERR ", "critique": "crit"}[self.kind]
        body = self.content.replace("\n", " ")
        if len(body) > 86:
            body = body[:85] + "…"
        return f"{self.sender:>12} -> {self.recipient:<12} [{tag}] {body}"


@dataclass
class Specialist:
    """A named worker with a description a router can read.

    The description is not documentation - it is the routing signal, and it is
    the single most common reason routing fails. "Handles data" tells a router
    nothing. Write it the way you write a tool description (notebook 07): what
    it does, and when to pick it over the others.
    """

    name: str
    description: str
    run: Callable[[str], str]
    tools: list[str] = field(default_factory=list)

    def __call__(self, task: str) -> str:
        return self.run(task)


@dataclass
class Team:
    """Specialists plus the shared transcript of what they said to each other."""

    members: list[Specialist] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)

    def add(self, *specialists: Specialist) -> Team:
        self.members.extend(specialists)
        return self

    @property
    def by_name(self) -> dict[str, Specialist]:
        return {s.name: s for s in self.members}

    def roster(self) -> str:
        """The members, as a router (or a model) would see them."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self.members)

    def send(self, sender: str, recipient: str, content: str,
             kind: str = "task") -> Message:
        message = Message(sender, recipient, content, kind)
        self.transcript.append(message)
        return message

    def dispatch(self, sender: str, recipient: str, task: str) -> str:
        """Send `task` to `recipient`, record both directions, return the result.

        A specialist that raises is recorded as an error message rather than
        being allowed to kill the whole team. One worker failing should cost
        you one result, not the run.
        """
        self.send(sender, recipient, task, "task")
        member = self.by_name.get(recipient)
        if member is None:
            error = f"no such specialist: {recipient!r}. Available: {', '.join(self.by_name)}"
            self.send("system", sender, error, "error")
            return error
        try:
            result = member(task)
            self.send(recipient, sender, result, "result")
            return result
        except Exception as exc:  # noqa: BLE001 - a specialist may raise anything
            error = f"{type(exc).__name__}: {exc}"
            self.send(recipient, sender, error, "error")
            return error

    def trace(self) -> str:
        return "\n".join(str(m) for m in self.transcript)

    @property
    def n_handoffs(self) -> int:
        """Messages sent. The cost metric nobody reports and everybody pays."""
        return len(self.transcript)


class Router:
    """Picks one specialist for a task. The cheapest useful pattern.

    One model call to choose, one to do the work. Compare that with an
    orchestrator, which pays for every worker whether or not their output is
    used.

    `choose` maps ``(task, roster) -> name``. In a notebook that is a model
    call; in a test it is a dict lookup, which is how the behaviour here is
    verified.
    """

    def __init__(self, team: Team, choose: Callable[[str, str], str],
                 *, fallback: str | None = None) -> None:
        self.team = team
        self.choose = choose
        self.fallback = fallback

    def run(self, task: str) -> str:
        name = (self.choose(task, self.team.roster()) or "").strip()
        if name not in self.team.by_name:
            # A router that picks a non-existent specialist is a routine
            # failure, not an exceptional one - models invent plausible names.
            # Fall back visibly rather than crashing or silently guessing.
            self.team.send("router", "system",
                           f"invalid choice {name!r}; falling back to {self.fallback!r}",
                           "error")
            if self.fallback is None:
                return f"No specialist could be selected for: {task}"
            name = self.fallback
        return self.team.dispatch("router", name, task)


class Orchestrator:
    """Fans subtasks out to specialists, then synthesises one answer.

    The pattern people mean by "multi-agent". It is genuinely useful when the
    subtasks are **independent** - four searches over four different sources -
    because then the handoffs are cheap and the context separation is real.

    It is a poor fit when subtask 2 needs what subtask 1 found. That is a
    *plan*, with dependencies, and :mod:`qwen_workshop.planning` already
    models it properly.
    """

    def __init__(self, team: Team, synthesise: Callable[[str, dict[str, str]], str],
                 *, name: str = "orchestrator") -> None:
        self.team = team
        self.synthesise = synthesise
        self.name = name

    def run(self, goal: str, assignments: list[tuple[str, str]]) -> str:
        """Run ``(specialist_name, subtask)`` pairs, then synthesise.

        Args:
            goal: the original question, passed to `synthesise` so the final
                answer is about what was asked rather than about the parts.
            assignments: who does what. Produced by a planner, or written by
                hand - writing them by hand is underrated and often enough.
        """
        results: dict[str, str] = {}
        for specialist_name, subtask in assignments:
            results[specialist_name] = self.team.dispatch(self.name, specialist_name, subtask)
        answer = self.synthesise(goal, results)
        self.team.send(self.name, "user", answer, "result")
        return answer


class CriticLoop:
    """A writer and a critic, until the critic accepts or the rounds run out.

    The one pattern where losing context is the *point*: a critic that has not
    seen the writer's reasoning judges the output on its merits, the way a
    reader will.

    Two rules, both learned the hard way:

    * **A round cap.** Without it the pair will trade refinements forever, and
      they will sound productive the whole time.
    * **An explicit accept token.** "Looks good to me, although…" is not an
      acceptance, and prose has no reliable way to say so. The critic must
      emit `accept_token` and nothing else counts.
    """

    def __init__(self, team: Team, writer: str, critic: str, *,
                 max_rounds: int = 3, accept_token: str = "APPROVED") -> None:
        self.team = team
        self.writer = writer
        self.critic = critic
        self.max_rounds = max_rounds
        self.accept_token = accept_token

    def run(self, task: str) -> dict:
        """Returns the draft, whether it was accepted, and how many rounds it took."""
        draft = self.team.dispatch("critic-loop", self.writer, task)
        for round_number in range(1, self.max_rounds + 1):
            verdict = self.team.dispatch("critic-loop", self.critic, draft)
            # Retag the critic's reply so the trace distinguishes a judgement
            # from an ordinary result. `dispatch` cannot know which it is.
            self.team.transcript[-1].kind = "critique"
            if self.accept_token in verdict:
                return {"draft": draft, "accepted": True, "rounds": round_number,
                        "last_critique": verdict}
            draft = self.team.dispatch(
                "critic-loop", self.writer,
                f"Revise this draft.\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{verdict}",
            )
        return {"draft": draft, "accepted": False, "rounds": self.max_rounds,
                "last_critique": verdict}

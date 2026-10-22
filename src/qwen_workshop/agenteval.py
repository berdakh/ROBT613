"""Evaluating agents: did it get the right answer, and did it get there sanely?

Why agents need their own evaluation
------------------------------------
For a single generation you check the output. For an agent that is not enough,
because an agent can reach the right answer the wrong way:

* It guessed, and happened to be right. Next week it guesses wrong.
* It called eleven tools where two would do. Correct, and forty times the cost.
* It called ``send_message`` while answering a read-only question. Correct
  answer, unacceptable behaviour.
* It looped until ``max_steps`` and then bluffed a confident answer.

So agent evaluation has two halves:

**Outcome** - is the final answer right? What you actually care about.

**Trajectory** - were the right tools called, in a sane order, without doing
anything it should not? What tells you *why*, and what catches the three
failures above, every one of which scores a perfect outcome.

Measure both. Outcome alone cannot distinguish a working agent from a lucky
one, and on a non-deterministic system that distinction is the whole game.

The harness here is deliberately boring: a list of :class:`AgentTestCase`, a
runner, a report. It needs no model to test *itself*, which is why the
examples in notebook 20 are reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "AgentTestCase",
    "CaseResult",
    "EvalReport",
    "evaluate",
    "judge_with_model",
    "tools_used",
]


def tools_used(run) -> list[str]:
    """Tool names an :class:`~qwen_workshop.agent.AgentRun` called, in order."""
    return [result.name for step in run.steps for result in step.tool_results]


# --------------------------------------------------------------------------
# A test case
# --------------------------------------------------------------------------


@dataclass
class AgentTestCase:
    """One thing the agent should be able to do, and what "right" means.

    Every field beyond `question` is optional, and a case with none of them is
    a case that cannot fail. Write at least one expectation.

    Args:
        question: what to ask.
        expect_substrings: all of these must appear in the answer (case
            insensitive). Crude and surprisingly effective - "412.50" either
            appears or it does not.
        expect_pattern: a regex the answer must match, for formats.
        expect_tools: tools that must be called. Order is not checked; use
            `expect_tool_order` for that.
        expect_tool_order: tools that must appear in this relative order,
            ignoring anything in between. Catches "answered before looking".
        forbid_tools: tools that must NOT be called. **The most valuable
            field in this class.** A read-only question that triggers
            `send_message` is a failure no outcome check will ever see.
        max_tool_calls: cost ceiling. A correct answer in 14 calls is a
            regression from a correct answer in 3.
        min_tool_calls: floor. Use this instead of `expect_tools` when several
            tools would legitimately do - "it must consult *something*" is
            often the real requirement, and naming one tool fails the runs
            that found a better route.
        must_refuse: the agent should decline or say it cannot answer. Needs
            `refusal_markers` to be meaningful.
        notes: why this case exists. Write it. In six months the case will
            fail and nobody will remember whether it should.
    """

    question: str
    expect_substrings: list[str] = field(default_factory=list)
    expect_pattern: str | None = None
    expect_tools: list[str] = field(default_factory=list)
    min_tool_calls: int | None = None
    expect_tool_order: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    max_tool_calls: int | None = None
    must_refuse: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not any([self.expect_substrings, self.expect_pattern, self.expect_tools,
                    self.expect_tool_order, self.forbid_tools, self.must_refuse,
                    self.max_tool_calls is not None, self.min_tool_calls is not None]):
            raise ValueError(
                f"test case {self.question!r} asserts nothing, so it can never fail. "
                "Give it at least one expectation."
            )


REFUSAL_MARKERS = (
    "cannot", "can't", "unable", "do not have", "don't have", "no information",
    "not available", "insufficient", "i don't know", "i do not know",
)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class CaseResult:
    """How one case went, split into outcome and trajectory."""

    case: AgentTestCase
    answer: str
    tools: list[str]
    outcome_failures: list[str] = field(default_factory=list)
    trajectory_failures: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def outcome_ok(self) -> bool:
        return not self.outcome_failures and self.error is None

    @property
    def trajectory_ok(self) -> bool:
        return not self.trajectory_failures and self.error is None

    @property
    def passed(self) -> bool:
        return self.outcome_ok and self.trajectory_ok

    def explain(self) -> str:
        if self.passed:
            return f"PASS  {self.case.question[:60]}  [{len(self.tools)} calls]"
        lines = [f"FAIL  {self.case.question[:60]}"]
        if self.error:
            lines.append(f"        crashed: {self.error}")
        for failure in self.outcome_failures:
            lines.append(f"        outcome:    {failure}")
        for failure in self.trajectory_failures:
            lines.append(f"        trajectory: {failure}")
        lines.append(f"        tools: {self.tools or '(none)'}")
        lines.append(f"        answer: {self.answer[:140]!r}")
        return "\n".join(lines)


@dataclass
class EvalReport:
    """The whole suite. Report both rates - one number hides the interesting half."""

    results: list[CaseResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def outcome_rate(self) -> float:
        return sum(r.outcome_ok for r in self.results) / self.n if self.n else 0.0

    @property
    def trajectory_rate(self) -> float:
        return sum(r.trajectory_ok for r in self.results) / self.n if self.n else 0.0

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / self.n if self.n else 0.0

    @property
    def mean_tool_calls(self) -> float:
        return sum(len(r.tools) for r in self.results) / self.n if self.n else 0.0

    @property
    def lucky(self) -> list[CaseResult]:
        """Right answer, wrong path. The cases worth your attention.

        These are the ones that will break when the model changes, the data
        changes, or the question is phrased slightly differently - and they
        look like passes in any outcome-only report.
        """
        return [r for r in self.results if r.outcome_ok and not r.trajectory_ok]

    def summary(self) -> str:
        lines = [
            f"{self.n} cases",
            f"  outcome    {self.outcome_rate:>6.0%}   (is the answer right)",
            f"  trajectory {self.trajectory_rate:>6.0%}   (did it get there sanely)",
            f"  both       {self.pass_rate:>6.0%}",
            f"  mean tool calls: {self.mean_tool_calls:.1f}",
        ]
        if self.lucky:
            lines.append(f"  !! {len(self.lucky)} lucky pass(es) - right answer, wrong path:")
            for result in self.lucky:
                lines.append(f"       {result.case.question[:56]}")
                for failure in result.trajectory_failures:
                    lines.append(f"         {failure}")
        return "\n".join(lines)

    def failures(self) -> str:
        bad = [r for r in self.results if not r.passed]
        return "\n".join(r.explain() for r in bad) if bad else "(no failures)"


# --------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------


def _check_order(actual: Sequence[str], required: Sequence[str]) -> bool:
    """Is `required` a subsequence of `actual`? Order matters, gaps do not."""
    iterator = iter(actual)
    return all(name in iterator for name in required)


def evaluate(
    run_one: Callable[[str], object],
    cases: list[AgentTestCase],
    *,
    judge: Callable[[AgentTestCase, str], str | None] | None = None,
) -> EvalReport:
    """Run every case and report.

    Args:
        run_one: takes a question, returns an ``AgentRun`` (anything with
            `.answer` and `.steps`). Keeping this a callable means the same
            harness evaluates a real agent, a scripted one, or a recorded
            trace replayed from disk.
        cases: what to test.
        judge: optional extra outcome check, for answers substrings cannot
            score. Returns a failure string, or None to pass. See
            :func:`judge_with_model`.

    An agent that raises is recorded as a failed case, not propagated - one
    broken case should not cost you the other nineteen results.
    """
    results: list[CaseResult] = []
    for case in cases:
        try:
            run = run_one(case.question)
        except Exception as exc:  # noqa: BLE001 - the agent may raise anything
            results.append(CaseResult(case, "", [], ["agent raised"], [],
                                      error=f"{type(exc).__name__}: {exc}"))
            continue

        answer = getattr(run, "answer", "") or ""
        tools = tools_used(run)
        lowered = answer.lower()
        outcome: list[str] = []
        trajectory: list[str] = []

        # -- outcome --
        for needle in case.expect_substrings:
            if needle.lower() not in lowered:
                outcome.append(f"missing {needle!r}")
        if case.expect_pattern and not re.search(case.expect_pattern, answer, re.S):
            outcome.append(f"does not match /{case.expect_pattern}/")
        if case.must_refuse and not any(m in lowered for m in REFUSAL_MARKERS):
            outcome.append("should have refused or said it could not answer")
        if judge is not None:
            if (verdict := judge(case, answer)) is not None:
                outcome.append(f"judge: {verdict}")

        # -- trajectory --
        for name in case.expect_tools:
            if name not in tools:
                trajectory.append(f"never called {name!r}")
        if case.expect_tool_order and not _check_order(tools, case.expect_tool_order):
            trajectory.append(
                f"expected order {case.expect_tool_order}, got {tools}"
            )
        for name in case.forbid_tools:
            if name in tools:
                trajectory.append(f"called forbidden tool {name!r}")
        if case.max_tool_calls is not None and len(tools) > case.max_tool_calls:
            trajectory.append(
                f"{len(tools)} tool calls, budget was {case.max_tool_calls}"
            )
        if case.min_tool_calls is not None and len(tools) < case.min_tool_calls:
            trajectory.append(
                f"only {len(tools)} tool call(s); expected at least "
                f"{case.min_tool_calls} - did it answer without looking?"
            )
        if getattr(run, "stopped_because", "finished") != "finished":
            trajectory.append(str(run.stopped_because))

        results.append(CaseResult(case, answer, tools, outcome, trajectory))
    return EvalReport(results)


# --------------------------------------------------------------------------
# Using a model as the judge
# --------------------------------------------------------------------------


JUDGE_PROMPT = """You are grading one answer from an assistant.

QUESTION: {question}
CRITERION: {criterion}
ANSWER: {answer}

Reply with exactly one word, PASS or FAIL, then a newline, then one short
sentence of justification. Judge only the stated criterion. Do not reward
length, confidence or politeness."""


def judge_with_model(generate: Callable[[str], str],
                     criteria: dict[str, str]) -> Callable[[AgentTestCase, str], str | None]:
    """Build a judge that asks a model whether the answer meets a criterion.

    Args:
        generate: ``prompt -> text``. A model call, or a stub in tests.
        criteria: question -> the criterion to judge it against. A question
            with no criterion is skipped rather than judged against nothing.

    Use this only for what substrings genuinely cannot check - tone, whether a
    refusal was appropriate, whether a summary is faithful. Three warnings,
    all of which show up in practice:

    * The judge **prefers long, confident answers**, which is exactly the bias
      preference tuning instals (notebook 15). It will reward verbosity.
    * The judge is a model, so it is wrong sometimes. A judged suite has a
      noise floor; do not read 2% movements.
    * Never judge with the same model you are evaluating. It grades its own
      style generously.
    """

    def judge(case: AgentTestCase, answer: str) -> str | None:
        criterion = criteria.get(case.question)
        if not criterion:
            return None
        verdict = generate(JUDGE_PROMPT.format(
            question=case.question, criterion=criterion, answer=answer or "(empty)"))
        head = verdict.strip().splitlines()[0].strip().upper() if verdict.strip() else ""
        if head.startswith("PASS"):
            return None
        reason = " ".join(verdict.strip().splitlines()[1:]).strip()
        return reason or "judged FAIL"

    return judge

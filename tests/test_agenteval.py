"""The agent evaluation harness - especially that it catches lucky passes."""

from __future__ import annotations

import pytest

from qwen_workshop.agenteval import (
    AgentTestCase,
    evaluate,
    judge_with_model,
    tools_used,
)


class FakeResult:
    def __init__(self, name):
        self.name = name


class FakeStep:
    def __init__(self, names):
        self.tool_results = [FakeResult(n) for n in names]


class FakeRun:
    """Minimal stand-in for AgentRun: an answer, some tool calls, a stop reason."""

    def __init__(self, answer, tools=(), stopped_because="finished"):
        self.answer = answer
        self.steps = [FakeStep(tools)] if tools else []
        self.stopped_because = stopped_because


def runner(mapping):
    return lambda question: mapping[question]


def test_case_with_no_expectations_is_rejected():
    with pytest.raises(ValueError, match="asserts nothing"):
        AgentTestCase("a question")


def test_tools_used_preserves_order():
    run = FakeRun("x", ["a", "b", "a"])
    assert tools_used(run) == ["a", "b", "a"]


def test_outcome_and_trajectory_both_pass():
    case = AgentTestCase("q", expect_substrings=["42"], expect_tools=["calc"])
    report = evaluate(runner({"q": FakeRun("the answer is 42", ["calc"])}), [case])
    assert report.pass_rate == 1.0
    assert report.lucky == []


def test_lucky_pass_is_caught():
    """Right answer, never looked at the data. The case this harness exists for."""
    case = AgentTestCase("q", expect_substrings=["42"], expect_tools=["calc"])
    report = evaluate(runner({"q": FakeRun("the answer is 42")}), [case])
    assert report.outcome_rate == 1.0
    assert report.trajectory_rate == 0.0
    assert len(report.lucky) == 1
    assert "never called 'calc'" in report.lucky[0].trajectory_failures[0]


def test_forbidden_tool_fails_a_correct_answer():
    case = AgentTestCase("q", expect_substrings=["sent"], forbid_tools=["send_message"])
    report = evaluate(runner({"q": FakeRun("sent", ["send_message"])}), [case])
    assert report.outcome_rate == 1.0
    assert report.pass_rate == 0.0


def test_tool_order_allows_gaps_but_not_inversion():
    ordered = AgentTestCase("q", expect_tool_order=["search", "read"])
    assert evaluate(runner({"q": FakeRun("x", ["search", "calc", "read"])}),
                    [ordered]).pass_rate == 1.0
    assert evaluate(runner({"q": FakeRun("x", ["read", "search"])}),
                    [ordered]).pass_rate == 0.0


def test_tool_budget_is_enforced():
    case = AgentTestCase("q", max_tool_calls=2)
    assert evaluate(runner({"q": FakeRun("x", ["a", "b", "c"])}), [case]).pass_rate == 0.0
    assert evaluate(runner({"q": FakeRun("x", ["a", "b"])}), [case]).pass_rate == 1.0


def test_hitting_max_steps_is_a_trajectory_failure():
    case = AgentTestCase("q", expect_substrings=["x"])
    report = evaluate(runner({"q": FakeRun("x", stopped_because="hit max_steps=6")}), [case])
    assert report.outcome_rate == 1.0
    assert report.trajectory_rate == 0.0


def test_refusal_is_checked():
    case = AgentTestCase("q", must_refuse=True)
    assert evaluate(runner({"q": FakeRun("I cannot answer that.")}), [case]).pass_rate == 1.0
    assert evaluate(runner({"q": FakeRun("The capital of Mars is Olympus.")}),
                    [case]).pass_rate == 0.0


def test_pattern_expectation():
    case = AgentTestCase("q", expect_pattern=r"^\d+\.\d{2}$")
    assert evaluate(runner({"q": FakeRun("412.50")}), [case]).pass_rate == 1.0
    assert evaluate(runner({"q": FakeRun("about 412")}), [case]).pass_rate == 0.0


def test_substring_check_is_case_insensitive():
    case = AgentTestCase("q", expect_substrings=["Yoghurt"])
    assert evaluate(runner({"q": FakeRun("your yoghurt expires")}), [case]).pass_rate == 1.0


def test_a_crashing_agent_fails_only_its_own_case():
    def run_one(question):
        if question == "boom":
            raise RuntimeError("agent died")
        return FakeRun("fine")

    cases = [AgentTestCase("boom", expect_substrings=["x"]),
             AgentTestCase("ok", expect_substrings=["fine"])]
    report = evaluate(run_one, cases)
    assert report.n == 2
    assert report.pass_rate == 0.5
    assert "agent died" in report.results[0].error


def test_report_summary_names_the_lucky_cases():
    case = AgentTestCase("q", expect_substrings=["42"], expect_tools=["calc"])
    summary = evaluate(runner({"q": FakeRun("42")}), [case]).summary()
    assert "lucky" in summary
    assert "trajectory" in summary


def test_judge_passes_and_fails():
    cases = [AgentTestCase("q", expect_substrings=["42"])]
    run = runner({"q": FakeRun("42")})

    passing = judge_with_model(lambda p: "PASS\nFine.", {"q": "is it grounded"})
    assert evaluate(run, cases, judge=passing).pass_rate == 1.0

    failing = judge_with_model(lambda p: "FAIL\nNot grounded in the data.",
                               {"q": "is it grounded"})
    report = evaluate(run, cases, judge=failing)
    assert report.pass_rate == 0.0
    assert "Not grounded" in report.results[0].outcome_failures[0]


def test_judge_skips_questions_with_no_criterion():
    """A question with no criterion must not be judged against nothing."""
    cases = [AgentTestCase("q", expect_substrings=["42"])]
    judge = judge_with_model(lambda p: pytest.fail("judge should not be called"), {})
    assert evaluate(runner({"q": FakeRun("42")}), cases, judge=judge).pass_rate == 1.0


def test_mean_tool_calls_is_reported():
    cases = [AgentTestCase("a", expect_substrings=["x"]),
             AgentTestCase("b", expect_substrings=["x"])]
    report = evaluate(runner({"a": FakeRun("x", ["t"]), "b": FakeRun("x", ["t", "t", "t"])}),
                      cases)
    assert report.mean_tool_calls == 2.0


def test_min_tool_calls_catches_answering_without_looking():
    """Use this when several tools would legitimately do the job."""
    case = AgentTestCase("q", min_tool_calls=1)
    assert evaluate(runner({"q": FakeRun("42")}), [case]).pass_rate == 0.0
    assert evaluate(runner({"q": FakeRun("42", ["anything"])}), [case]).pass_rate == 1.0


def test_min_tool_calls_alone_is_enough_of_an_expectation():
    AgentTestCase("q", min_tool_calls=1)  # must not raise

"""The scripted model stand-in, and that it really behaves like a server."""

from __future__ import annotations

import json

import pytest

from qwen_workshop.agent import Agent, ReActAgent
from qwen_workshop.scripted import (
    ScriptedClient,
    ScriptedModel,
    ScriptedText,
    calls,
    says,
)
from qwen_workshop.tools import ToolRegistry, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("tool exploded")


@pytest.fixture
def registry():
    return ToolRegistry().add(add, boom)


def test_empty_turn_is_rejected():
    from qwen_workshop.scripted import ScriptedTurn

    with pytest.raises(ValueError):
        ScriptedTurn()  # neither content nor tool calls is not a turn


def test_empty_answer_is_scriptable(registry):
    """`says("")` stays legal on purpose: a model returning nothing is a real
    failure mode, and you want to be able to reproduce it deliberately."""
    model = ScriptedModel([says("")])
    run = Agent(*model.as_client(), registry).run("q")
    assert run.answer == ""
    assert run.stopped_because == "finished"


def test_arguments_are_json_strings_like_a_real_server():
    """A dict here would let student code skip json.loads and still pass."""
    client = ScriptedClient([calls("add", {"a": 1, "b": 2})])
    message = client.create(messages=[]).choices[0].message
    raw = message.tool_calls[0].function.arguments
    assert isinstance(raw, str)
    assert json.loads(raw) == {"a": 1, "b": 2}


def test_finish_reason_matches_the_turn():
    client = ScriptedClient([calls("add", {"a": 1, "b": 2}), says("3")])
    assert client.create(messages=[]).choices[0].finish_reason == "tool_calls"
    assert client.create(messages=[]).choices[0].finish_reason == "stop"


def test_tool_call_ids_are_unique_across_turns():
    """Duplicate ids make the tool replies ambiguous to a real server."""
    client = ScriptedClient([calls("add", {"a": 1, "b": 1}), calls("add", {"a": 2, "b": 2})])
    first = client.create(messages=[]).choices[0].message.tool_calls[0].id
    second = client.create(messages=[]).choices[0].message.tool_calls[0].id
    assert first != second


def test_agent_runs_a_script(registry):
    model = ScriptedModel([calls("add", {"a": 2, "b": 3}), says("It is 5.")])
    run = Agent(*model.as_client(), registry).run("2+3?")
    assert run.answer == "It is 5."
    assert run.n_tool_calls == 1
    assert run.stopped_because == "finished"


def test_parallel_calls_in_one_turn(registry):
    model = ScriptedModel([calls("add", {"a": 1, "b": 1}, ("add", {"a": 2, "b": 2})),
                           says("done")])
    run = Agent(*model.as_client(), registry).run("two sums")
    assert run.n_tool_calls == 2


def test_thought_alongside_tool_calls_is_preserved(registry):
    model = ScriptedModel([calls("add", {"a": 1, "b": 1}, thought="Let me add."),
                           says("2")])
    run = Agent(*model.as_client(), registry).run("q")
    assert run.steps[0].thought == "Let me add."


def test_tool_error_reaches_the_transcript(registry):
    model = ScriptedModel([calls("boom"), says("That tool failed.")])
    run = Agent(*model.as_client(), registry).run("q")
    assert run.steps[0].tool_results[0].ok is False
    assert "exploded" in run.steps[0].tool_results[0].content


def test_tools_offered_is_inspectable(registry):
    model = ScriptedModel([says("hi")])
    Agent(*model.as_client(), registry).run("q")
    assert set(model.client.tools_offered()) == {"add", "boom"}


def test_running_off_the_end_terminates(registry):
    """An over-short script must not hang or loop."""
    model = ScriptedModel([calls("add", {"a": 1, "b": 1})])
    run = Agent(*model.as_client(), registry, max_steps=4).run("q")
    assert model.client.exhausted_script
    assert run.answer  # it still produced something


def test_max_steps_is_enforced(registry):
    model = ScriptedModel([calls("add", {"a": 1, "b": 1})] * 20)
    run = Agent(*model.as_client(), registry, max_steps=3).run("q")
    assert len(run.steps) == 3
    assert "max_steps=3" in run.stopped_because


def test_scripted_text_drives_react(registry):
    generate = ScriptedText([
        "Thought: add them.\nAction: add\nAction Input: {\"a\": 4, \"b\": 5}",
        "Thought: I now know the final answer\nFinal Answer: 9",
    ])
    run = ReActAgent(generate, registry).run("4+5?")
    assert run.answer.strip() == "9"
    assert generate.turns_served == 2
    assert "Observation: 9" in generate.last_prompt_text()


def test_scripted_text_exhaustion_ends_the_run(registry):
    generate = ScriptedText([])
    run = ReActAgent(generate, registry).run("q")
    assert "exhausted" in run.answer

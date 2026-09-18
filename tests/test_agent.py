"""The agent loop, driven by a scripted fake model (no network, no weights)."""

from __future__ import annotations

import pytest
from conftest import FakeClient, FakeMessage, FakeToolCall

from qwen_workshop.agent import Agent, ReActAgent
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
def fail_once(x: int) -> str:
    """A tool that rejects negative input.

    Args:
        x: Must be positive.
    """
    if x < 0:
        raise ValueError("x must be positive")
    return f"ok:{x}"


@pytest.fixture
def registry():
    return ToolRegistry().add(add, fail_once)


def test_agent_answers_without_tools(backend, registry):
    client = FakeClient([FakeMessage(content="Hello, no tools needed.")])
    run = Agent(client, backend, registry).run("hi")
    assert run.answer == "Hello, no tools needed."
    assert run.n_tool_calls == 0
    assert run.stopped_because == "finished"


def test_agent_calls_tool_then_answers(backend, registry):
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("add", '{"a": 2, "b": 3}')]),
            FakeMessage(content="The answer is 5."),
        ]
    )
    run = Agent(client, backend, registry).run("what is 2+3?")
    assert run.answer == "The answer is 5."
    assert run.n_tool_calls == 1
    assert run.steps[0].tool_results[0].content == "5"


def test_tool_result_is_appended_to_transcript(backend, registry):
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("add", '{"a": 1, "b": 1}')]),
            FakeMessage(content="2"),
        ]
    )
    run = Agent(client, backend, registry).run("1+1")
    roles = [m["role"] for m in run.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_message = run.messages[3]
    assert tool_message["tool_call_id"] == "call_1"
    assert tool_message["content"] == "2"


def test_tools_are_advertised_to_the_model(backend, registry):
    client = FakeClient([FakeMessage(content="done")])
    Agent(client, backend, registry).run("hi")
    advertised = {t["function"]["name"] for t in client.requests[0]["tools"]}
    assert advertised == {"add", "fail_once"}


def test_agent_recovers_from_tool_error(backend, registry):
    """A failing tool must feed its error back, not crash the loop."""
    client = FakeClient(
        [
            FakeMessage(tool_calls=[FakeToolCall("fail_once", '{"x": -5}')]),
            FakeMessage(tool_calls=[FakeToolCall("fail_once", '{"x": 5}', "call_2")]),
            FakeMessage(content="Fixed it: ok:5"),
        ]
    )
    run = Agent(client, backend, registry).run("try it")
    assert run.n_tool_calls == 2
    assert not run.steps[0].tool_results[0].ok
    assert run.steps[1].tool_results[0].ok
    assert run.answer == "Fixed it: ok:5"


def test_agent_stops_at_max_steps(backend, registry):
    """A model stuck in a loop must be cut off, not run forever."""
    looping = [FakeMessage(tool_calls=[FakeToolCall("add", '{"a": 1, "b": 1}')]) for _ in range(20)]
    client = FakeClient(looping + [FakeMessage(content="Partial answer.")])
    run = Agent(client, backend, registry, max_steps=3).run("loop forever")
    assert len(run.steps) == 3
    assert run.stopped_because == "hit max_steps=3"
    # The forced final call must not offer tools again.
    assert "tools" not in client.requests[-1]


def test_agent_handles_parallel_tool_calls(backend, registry):
    client = FakeClient(
        [
            FakeMessage(
                tool_calls=[
                    FakeToolCall("add", '{"a": 1, "b": 2}', "c1"),
                    FakeToolCall("add", '{"a": 10, "b": 20}', "c2"),
                ]
            ),
            FakeMessage(content="3 and 30."),
        ]
    )
    run = Agent(client, backend, registry).run("both")
    assert run.n_tool_calls == 2
    assert [r.content for r in run.steps[0].tool_results] == ["3", "30"]


def test_trace_is_readable(backend, registry):
    client = FakeClient(
        [FakeMessage(tool_calls=[FakeToolCall("add", '{"a": 2, "b": 2}')]), FakeMessage(content="4")]
    )
    trace = Agent(client, backend, registry).run("2+2").trace()
    assert "add(" in trace and "A: 4" in trace


# --------------------------------------------------------------------------
# ReAct (text-protocol) agent
# --------------------------------------------------------------------------


def test_react_agent_full_loop(registry):
    replies = iter(
        [
            'Thought: I should add them.\nAction: add\nAction Input: {"a": 4, "b": 6}',
            "Thought: I now know the final answer\nFinal Answer: 10",
        ]
    )
    agent = ReActAgent(lambda messages: next(replies), registry)
    run = agent.run("what is 4+6?")
    assert run.answer == "10"
    assert run.n_tool_calls == 1
    assert run.messages[-1]["content"] == "Observation: 10"


def test_react_agent_gives_up_gracefully(registry):
    agent = ReActAgent(
        lambda messages: 'Thought: again\nAction: add\nAction Input: {"a": 1, "b": 1}',
        registry,
        max_steps=2,
    )
    run = agent.run("loop")
    assert "ran out of steps" in run.answer
    assert run.stopped_because == "hit max_steps=2"


def test_react_tool_descriptions_reach_the_prompt(registry):
    captured = {}

    def generate(messages):
        captured["system"] = messages[0]["content"]
        return "Final Answer: done"

    ReActAgent(generate, registry).run("hi")
    assert "add(a: integer, b: integer)" in captured["system"]

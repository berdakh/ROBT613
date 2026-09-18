"""Shared fixtures.

These tests deliberately use **no model and no network**. They cover the
plumbing - schemas, parsing, chunking, the agent loop - which is exactly the
part students break when they adapt the code. Model quality is evaluated in
notebook 09/12, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class FakeToolCall:
    """Mimics the OpenAI SDK's tool-call object shape."""

    def __init__(self, name: str, arguments: str, call_id: str = "call_1") -> None:
        self.id = call_id
        self.type = "function"
        self.function = type("Fn", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


class FakeResponse:
    def __init__(self, message) -> None:
        self.choices = [type("Choice", (), {"message": message, "finish_reason": "stop"})()]


class FakeClient:
    """A scripted OpenAI-compatible client.

    Give it a list of FakeMessages; it returns them in order and records every
    request it received so tests can assert on the transcript.
    """

    def __init__(self, scripted: list[FakeMessage]) -> None:
        self.scripted = list(scripted)
        self.requests: list[dict] = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.scripted:
            return FakeResponse(FakeMessage(content="(no more scripted replies)"))
        return FakeResponse(self.scripted.pop(0))


@pytest.fixture
def backend():
    from qwen_workshop.client import Backend

    return Backend(name="fake", base_url="http://localhost:0/v1", api_key="x", model="qwen3-test")

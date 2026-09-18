"""Chat-template helpers and Qwen thinking-mode parsing."""

from __future__ import annotations

from qwen_workshop.chat import (
    build_prompt,
    extract_tool_calls,
    split_thinking,
    supports_thinking_switch,
)


class FakeTokenizer:
    """Records how apply_chat_template was called."""

    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self, accepts_thinking: bool = True) -> None:
        self.accepts_thinking = accepts_thinking
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        if "enable_thinking" in kwargs and not self.accepts_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        self.calls.append({"messages": messages, **kwargs})
        return "PROMPT:" + "|".join(m["role"] for m in messages)


def test_split_thinking_with_both_tags():
    thinking, answer = split_thinking("<think>2+2 is 4</think>The answer is 4.")
    assert thinking == "2+2 is 4"
    assert answer == "The answer is 4."


def test_split_thinking_when_open_tag_came_from_the_template():
    """The template often emits <think> itself, so only </think> is decoded."""
    thinking, answer = split_thinking("counting on fingers</think>Four.")
    assert thinking == "counting on fingers"
    assert answer == "Four."


def test_split_thinking_without_any_tags():
    assert split_thinking("  Just an answer.  ") == (None, "Just an answer.")


def test_empty_thinking_block_becomes_none():
    thinking, answer = split_thinking("<think></think>Answer.")
    assert thinking is None
    assert answer == "Answer."


def test_build_prompt_forwards_thinking_flag():
    tokenizer = FakeTokenizer()
    build_prompt(tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert tokenizer.calls[0]["enable_thinking"] is True
    assert tokenizer.calls[0]["add_generation_prompt"] is True


def test_build_prompt_retries_without_flag_on_older_templates():
    """Qwen2.5 and the 2507 splits have no thinking switch - must not crash."""
    tokenizer = FakeTokenizer(accepts_thinking=False)
    prompt = build_prompt(tokenizer, [{"role": "user", "content": "hi"}], enable_thinking=True)
    assert prompt == "PROMPT:user"
    assert "enable_thinking" not in tokenizer.calls[0]


def test_build_prompt_passes_tools_through():
    tokenizer = FakeTokenizer()
    schemas = [{"type": "function", "function": {"name": "f"}}]
    build_prompt(tokenizer, [{"role": "user", "content": "hi"}], tools=schemas)
    assert tokenizer.calls[0]["tools"] == schemas


def test_supports_thinking_switch_detection():
    assert supports_thinking_switch(FakeTokenizer())

    class Old:
        chat_template = "{{ messages }}"

    assert not supports_thinking_switch(Old())


def test_extract_tool_calls_parses_hermes_blocks():
    raw = (
        'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Astana"}}\n'
        "</tool_call>"
    )
    calls = extract_tool_calls(raw)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Astana"}}]


def test_extract_tool_calls_skips_malformed_blocks():
    raw = "<tool_call>not json</tool_call><tool_call>{\"name\": \"ok\"}</tool_call>"
    calls = extract_tool_calls(raw)
    assert len(calls) == 1
    assert calls[0] == {"name": "ok", "arguments": {}}


def test_extract_tool_calls_handles_multiple():
    raw = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    assert [c["name"] for c in extract_tool_calls(raw)] == ["a", "b"]

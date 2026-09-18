"""Chat helpers: templates, generation, streaming and thinking-mode parsing.

The single most important idea in this file: **a chat model does not take a
list of messages, it takes a string.** The tokenizer's chat template is what
turns one into the other. If your output looks broken, print the templated
string first - nine times out of ten the bug is visible there.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .config import NON_THINKING_SAMPLING, THINKING_SAMPLING, SamplingParams

Message = dict  # {"role": "system"|"user"|"assistant"|"tool", "content": str}

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class ChatResult:
    """Everything notebook 03/04 wants to inspect after a generation."""

    text: str
    thinking: str | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    raw: str

    @property
    def tokens_per_second(self) -> float:
        if self.seconds <= 0:
            return float("nan")
        return self.completion_tokens / self.seconds

    def __str__(self) -> str:
        return self.text


def split_thinking(raw: str) -> tuple[str | None, str]:
    """Separate a Qwen3 reasoning block from the final answer.

    Qwen3 hybrid models emit ``<think> ... </think>`` before the answer. The
    opening tag is sometimes injected by the chat template rather than
    generated, so it may be missing from the decoded text while the closing
    tag is present - handle both shapes.

    Returns:
        ``(thinking_or_None, answer)``.
    """
    if THINK_CLOSE not in raw:
        return None, raw.strip()
    head, _, tail = raw.partition(THINK_CLOSE)
    thinking = head.replace(THINK_OPEN, "").strip()
    return (thinking or None), tail.strip()


def build_prompt(
    tokenizer,
    messages: Iterable[Message],
    *,
    enable_thinking: bool | None = None,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages into the exact string the model will see.

    ``enable_thinking`` only exists on Qwen3 *hybrid* checkpoints (Qwen3-0.6B,
    -4B, -8B, ...). The split ``-Instruct-2507`` / ``-Thinking-2507`` releases
    dropped the switch, and older families (Qwen2.5) never had it - so we pass
    it optimistically and fall back if the template rejects it.
    """
    kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return tokenizer.apply_chat_template(list(messages), **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(list(messages), **kwargs)


def supports_thinking_switch(tokenizer) -> bool:
    """True if this tokenizer's chat template understands `enable_thinking`."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "enable_thinking" in template


def chat(
    loaded,
    messages: list[Message] | str,
    *,
    sampling: SamplingParams | None = None,
    enable_thinking: bool = False,
    tools: list[dict] | None = None,
    system: str | None = None,
) -> ChatResult:
    """Run one turn of chat and return a :class:`ChatResult`.

    Args:
        loaded: a :class:`~qwen_workshop.loading.LoadedModel`.
        messages: a message list, or a bare string treated as one user turn.
        sampling: decoding settings. Defaults to Qwen's recommended preset for
            the mode you chose, which matters more than students expect.
        enable_thinking: turn on Qwen3 reasoning mode.
        tools: OpenAI-style tool schemas to expose in the prompt (notebook 07).
        system: convenience system prompt, prepended if `messages` has none.
    """
    import time

    import torch

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages = list(messages)
    if system and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": system}, *messages]

    sampling = sampling or (THINKING_SAMPLING if enable_thinking else NON_THINKING_SAMPLING)

    prompt = build_prompt(
        loaded.tokenizer, messages, enable_thinking=enable_thinking, tools=tools
    )
    inputs = loaded.tokenizer([prompt], return_tensors="pt").to(loaded.model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = loaded.model.generate(
            **inputs,
            pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
            **sampling.as_dict(),
        )
    seconds = time.perf_counter() - start

    # Slice off the prompt: generate() returns prompt + completion.
    new_ids = output_ids[0][prompt_tokens:]
    raw = loaded.tokenizer.decode(new_ids, skip_special_tokens=True)
    thinking, answer = split_thinking(raw)

    return ChatResult(
        text=answer,
        thinking=thinking,
        prompt_tokens=prompt_tokens,
        completion_tokens=int(new_ids.shape[-1]),
        seconds=seconds,
        raw=raw,
    )


def stream_chat(
    loaded,
    messages: list[Message] | str,
    *,
    sampling: SamplingParams | None = None,
    enable_thinking: bool = False,
    system: str | None = None,
) -> Iterator[str]:
    """Yield text chunks as they are generated.

    Streaming is not a performance trick - total time is identical. It is a
    *perceived latency* trick, and it is why every chat UI does it.
    """
    from threading import Thread

    from transformers import TextIteratorStreamer

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    messages = list(messages)
    if system and not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": system}, *messages]

    sampling = sampling or (THINKING_SAMPLING if enable_thinking else NON_THINKING_SAMPLING)
    prompt = build_prompt(loaded.tokenizer, messages, enable_thinking=enable_thinking)
    inputs = loaded.tokenizer([prompt], return_tensors="pt").to(loaded.model.device)

    streamer = TextIteratorStreamer(
        loaded.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
        **sampling.as_dict(),
    )
    # generate() blocks, so it runs on a worker thread while we consume.
    thread = Thread(target=loaded.model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()
    try:
        yield from streamer
    finally:
        thread.join()


def print_stream(chunks: Iterator[str], hide_thinking: bool = False) -> str:
    """Print a stream token-by-token and return the assembled text."""
    pieces: list[str] = []
    in_think = False
    for chunk in chunks:
        pieces.append(chunk)
        if hide_thinking:
            joined = "".join(pieces)
            if THINK_OPEN in joined and THINK_CLOSE not in joined:
                if not in_think:
                    print("[thinking...]", end="", flush=True)
                    in_think = True
                continue
            if in_think and THINK_CLOSE in joined:
                in_think = False
                print()
                continue
        print(chunk, end="", flush=True)
    print()
    return "".join(pieces)


def extract_tool_calls(raw: str) -> list[dict]:
    """Pull Hermes-style ``<tool_call>`` blocks out of raw Qwen output.

    Qwen3 emits tool calls as::

        <tool_call>
        {"name": "get_weather", "arguments": {"city": "Astana"}}
        </tool_call>

    When you use an OpenAI-compatible server (vLLM/Ollama) with a tool parser
    enabled you get structured ``message.tool_calls`` instead and never need
    this. It exists so notebook 07 can show what is really on the wire.
    """
    import json

    calls: list[dict] = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, flags=re.DOTALL):
        try:
            payload = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "name" in payload:
            payload.setdefault("arguments", {})
            calls.append(payload)
    return calls

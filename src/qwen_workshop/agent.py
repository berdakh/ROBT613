"""A minimal but honest agent loop.

An "agent" is far less magical than the marketing suggests. It is this::

    while not done and steps < limit:
        reply = model(transcript)          # 1. think
        if reply has tool calls:           # 2. act
            run them, append results       # 3. observe
        else:
            done = True

Everything else - planning, memory, multi-agent, MCP - is a variation on that
loop. Read :meth:`Agent.run` once and you will recognise the pattern inside
every agent framework you meet afterwards.

Two front-ends are provided:

* :class:`Agent` - uses **native tool calling** over an OpenAI-compatible
  server. This is what you should ship.
* :class:`ReActAgent` - parses tool calls out of **plain text**. Slower and
  more fragile, but it works with any model and shows you what frameworks are
  doing under the hood.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .tools import ToolRegistry, ToolResult

DEFAULT_SYSTEM_PROMPT = """You are a careful, concise assistant with access to tools.

Rules:
- Use a tool whenever it gives you a fact you cannot be sure of (dates, maths,
  files, anything about the user's own data). Do not guess.
- Call one tool at a time and read the result before deciding the next step.
- If a tool returns an error, read the error and try a corrected call. Do not
  repeat the identical call.
- When you have enough information, stop calling tools and answer in plain
  language. Say which tool results you relied on.
- If the tools cannot answer the question, say so plainly."""


@dataclass
class Step:
    """One iteration of the loop - the unit you inspect when debugging."""

    index: int
    thought: str | None
    tool_results: list[ToolResult] = field(default_factory=list)
    final: str | None = None

    def summary(self) -> str:
        if self.final is not None:
            return f"step {self.index}: FINAL ANSWER"
        calls = ", ".join(f"{r.name}({_short(r.arguments)})" for r in self.tool_results)
        return f"step {self.index}: called {calls or 'nothing'}"


@dataclass
class AgentRun:
    """The full trace of one `agent.run(...)`.

    Keeping the trace is not optional extra credit. An agent you cannot trace
    is an agent you cannot debug, and non-determinism means you often get one
    shot at seeing a failure.
    """

    question: str
    answer: str
    steps: list[Step]
    messages: list[dict]
    stopped_because: str = "finished"

    @property
    def n_tool_calls(self) -> int:
        return sum(len(step.tool_results) for step in self.steps)

    def trace(self) -> str:
        lines = [f"Q: {self.question}", f"({len(self.steps)} steps, {self.n_tool_calls} tool calls, stopped: {self.stopped_because})"]
        for step in self.steps:
            lines.append(f"  {step.summary()}")
            for result in step.tool_results:
                status = "ok" if result.ok else "ERR"
                lines.append(f"      [{status}] {_short(result.content, 160)}")
        lines.append(f"A: {self.answer}")
        return "\n".join(lines)


class Agent:
    """Tool-calling agent on top of an OpenAI-compatible endpoint.

    Args:
        client: an ``openai.OpenAI`` instance (see :func:`qwen_workshop.client.get_client`).
        backend: the :class:`~qwen_workshop.client.Backend` that came with it.
        registry: the tools this agent may use. Nothing outside it can run.
        system: system prompt.
        max_steps: hard cap on loop iterations. **Always set one.** A model
            stuck in a call/error cycle will otherwise burn your GPU all night.
        on_step: optional callback, called with each :class:`Step` as it
            completes - handy for live progress in a notebook.
    """

    def __init__(
        self,
        client,
        backend,
        registry: ToolRegistry,
        *,
        system: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
        temperature: float = 0.6,
        max_tokens: int = 1024,
        enable_thinking: bool | None = False,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.client = client
        self.backend = backend
        self.registry = registry
        self.system = system
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.on_step = on_step

    def run(self, question: str, *, history: list[dict] | None = None) -> AgentRun:
        """Answer `question`, calling tools as needed."""
        from .client import complete

        messages: list[dict] = [{"role": "system", "content": self.system}]
        messages += list(history or [])
        messages.append({"role": "user", "content": question})

        steps: list[Step] = []
        stopped = "finished"

        for index in range(1, self.max_steps + 1):
            response = complete(
                self.client,
                self.backend,
                messages,
                tools=self.registry.schemas() or None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            # Echo the assistant turn back into the transcript verbatim -
            # the server needs the tool_calls to match the tool replies.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    **({"tool_calls": [_dump(c) for c in tool_calls]} if tool_calls else {}),
                }
            )

            if not tool_calls:
                step = Step(index=index, thought=None, final=message.content or "")
                steps.append(step)
                if self.on_step:
                    self.on_step(step)
                return AgentRun(question, step.final, steps, messages, stopped)

            results: list[ToolResult] = []
            for call in tool_calls:
                result = self.registry.call(call.function.name, call.function.arguments)
                results.append(result)
                messages.append(result.to_message(call_id=getattr(call, "id", None)))

            step = Step(index=index, thought=message.content or None, tool_results=results)
            steps.append(step)
            if self.on_step:
                self.on_step(step)

        # Ran out of steps: ask once more, without tools, for a best-effort answer.
        stopped = f"hit max_steps={self.max_steps}"
        messages.append(
            {
                "role": "user",
                "content": "Stop calling tools. Answer now using what you already found, "
                "and say clearly if the answer is incomplete.",
            }
        )
        final = complete(
            self.client, self.backend, messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
            enable_thinking=False,
        )
        answer = final.choices[0].message.content or "(no answer)"
        messages.append({"role": "assistant", "content": answer})
        return AgentRun(question, answer, steps, messages, stopped)


REACT_SYSTEM_PROMPT = """You solve problems by reasoning and using tools.

Available tools:
{tool_descriptions}

Answer in EXACTLY this format, one block at a time:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <a JSON object of arguments>

Then STOP and wait. You will receive:

Observation: <the tool result>

Repeat Thought/Action/Action Input as many times as you need. When you can
answer, write:

Thought: I now know the final answer
Final Answer: <your answer to the user>

Never invent an Observation yourself. Never use a tool that is not listed."""


class ReActAgent:
    """Text-parsing agent, for models or runtimes without native tool calling.

    ReAct = *Reason + Act* (Yao et al., 2022). It gets the same behaviour out
    of a plain completion model by defining a strict text protocol and parsing
    it. Use it to understand the mechanism; use :class:`Agent` in production,
    because parsing free text is exactly as brittle as it sounds.
    """

    def __init__(
        self,
        generate: Callable[[list[dict]], str],
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
    ) -> None:
        # `generate` is any callable: messages -> assistant text. That keeps
        # this class independent of transformers, Ollama, vLLM, everything.
        self.generate = generate
        self.registry = registry
        self.max_steps = max_steps

    def _system(self) -> str:
        lines = []
        for schema in self.registry.schemas():
            fn = schema["function"]
            params = ", ".join(
                f"{k}: {v.get('type','string')}" for k, v in fn["parameters"]["properties"].items()
            )
            lines.append(f"- {fn['name']}({params}): {fn['description']}")
        return REACT_SYSTEM_PROMPT.format(tool_descriptions="\n".join(lines))

    @staticmethod
    def parse(text: str) -> dict:
        """Extract Thought / Action / Action Input / Final Answer from text."""
        if match := re.search(r"Final Answer\s*:\s*(.+)", text, re.DOTALL):
            return {"final": match.group(1).strip()}

        thought_match = re.search(r"Thought\s*:\s*(.+?)(?=\n\s*Action\s*:|\Z)", text, re.DOTALL)
        action_match = re.search(r"Action\s*:\s*(.+?)(?=\n|$)", text)
        input_match = re.search(
            r"Action Input\s*:\s*(\{.*?\})(?=\n\s*(?:Thought|Observation|Action)\s*:|\Z)",
            text,
            re.DOTALL,
        )
        if not action_match:
            return {"final": text.strip()}

        arguments: dict = {}
        if input_match:
            try:
                arguments = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                arguments = {}
        return {
            "thought": thought_match.group(1).strip() if thought_match else None,
            "action": action_match.group(1).strip().strip("`\"'"),
            "arguments": arguments,
        }

    def run(self, question: str) -> AgentRun:
        messages = [
            {"role": "system", "content": self._system()},
            {"role": "user", "content": question},
        ]
        steps: list[Step] = []
        stopped = "finished"

        for index in range(1, self.max_steps + 1):
            text = self.generate(messages)
            parsed = self.parse(text)

            if "final" in parsed:
                step = Step(index=index, thought=None, final=parsed["final"])
                steps.append(step)
                return AgentRun(question, parsed["final"], steps, messages, stopped)

            result = self.registry.call(parsed["action"], parsed.get("arguments"))
            steps.append(Step(index=index, thought=parsed.get("thought"), tool_results=[result]))
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {result.content}"})

        stopped = f"hit max_steps={self.max_steps}"
        return AgentRun(question, "(agent ran out of steps)", steps, messages, stopped)


def _dump(call) -> dict:
    """Normalise an SDK tool-call object into a plain dict."""
    return {
        "id": getattr(call, "id", None),
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }


def _short(value, limit: int = 60) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."

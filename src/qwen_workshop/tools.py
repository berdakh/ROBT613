"""Turning plain Python functions into tools an LLM can call.

A "tool" is three things glued together:

1. a **JSON schema** the model reads, so it knows the tool exists and what
   arguments it takes,
2. a **name -> callable** mapping, so your code can execute what the model asked
   for,
3. a **safety boundary**, because the model's arguments are untrusted input.

Nothing here is Qwen-specific: the schema format is the OpenAI function-calling
shape, which Qwen3, vLLM, Ollama and llama.cpp all understand.
"""

from __future__ import annotations

import builtins
import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Python type -> JSON Schema type
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolError(Exception):
    """Raised when a tool cannot run. The message is fed back to the model.

    This is deliberate: a well-written error message is a *prompt*. "City must
    be one of: Astana, Almaty" lets the model fix itself on the next turn,
    while "KeyError" makes it guess again.
    """


def _json_type(annotation: Any) -> dict:
    """Map a Python annotation onto a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] / X | None -> schema of X (nullability is handled by "required")
    if origin is typing.Union or (origin is not None and str(origin) == "<class 'types.UnionType'>"):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"type": [_json_type(a).get("type", "string") for a in non_none]}

    # Literal["a", "b"] -> enum, the single most useful constraint you can give
    if origin is typing.Literal:
        return {"type": "string", "enum": [str(a) for a in args]}

    if origin in (list, set, tuple):
        item = _json_type(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    return {"type": _JSON_TYPES.get(annotation, "string")}


def _resolved_hints(func: Callable) -> dict[str, Any]:
    """Resolve annotations to real objects, tolerating un-importable names.

    ``typing.get_type_hints`` evaluates string annotations against the
    function's module globals. It is all-or-nothing: one name it cannot resolve
    (defined under ``TYPE_CHECKING``, or local to an enclosing scope) and the
    whole call raises, which would silently degrade *every* argument to
    ``"string"``.

    So on failure we resolve each annotation independently and keep whatever
    works.

    Note: this evaluates the *developer's own* annotation source, exactly as
    ``get_type_hints`` does internally. It never touches model output - see the
    security notes in ``demo_tools.calculate`` for that rule.
    """
    try:
        return typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - fall through to per-argument resolution
        pass

    raw = dict(getattr(func, "__annotations__", {}) or {})
    namespace: dict[str, Any] = {
        **vars(builtins),
        **vars(typing),
        **getattr(func, "__globals__", {}),
    }
    resolved: dict[str, Any] = {}
    for name, annotation in raw.items():
        if not isinstance(annotation, str):
            resolved[name] = annotation
            continue
        try:
            resolved[name] = eval(annotation, namespace)  # noqa: S307
        except Exception:  # noqa: BLE001 - this one argument stays a string
            resolved[name] = annotation
    return resolved


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into a summary and per-argument help.

    The argument descriptions end up in the schema, which means **your
    docstring is part of your prompt**. Vague docstrings produce vague tool
    calls.
    """
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).split("\n")

    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    summary_done = False
    current: str | None = None

    for line in lines:
        stripped = line.strip()

        if re.match(r"^(Args|Arguments|Parameters)\s*:$", stripped):
            in_args, summary_done, current = True, True, None
            continue
        if re.match(r"^(Returns|Raises|Yields|Examples?|Notes?)\s*:$", stripped):
            in_args, summary_done, current = False, True, None
            continue

        if in_args:
            # "name: description" or "name (type): description"
            match = re.match(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", stripped)
            if match:
                current = match.group(1).lstrip("*")
                params[current] = match.group(2).strip()
            elif current and stripped:
                # continuation of the previous argument's description
                params[current] = (params[current] + " " + stripped).strip()
            continue

        if summary_done:
            continue
        if stripped:
            summary_lines.append(stripped)
        elif summary_lines:
            # A blank line ends the summary paragraph, but we keep scanning
            # for an "Args:" section further down.
            summary_done = True

    return " ".join(summary_lines).strip(), params


def function_schema(func: Callable) -> dict:
    """Build an OpenAI-style function schema by introspecting `func`.

    Example::

        >>> def add(a: int, b: int = 0) -> int:
        ...     '''Add two numbers.
        ...
        ...     Args:
        ...         a: First number.
        ...         b: Second number.
        ...     '''
        >>> function_schema(add)["function"]["name"]
        'add'
    """
    signature = inspect.signature(func)
    summary, param_docs = _parse_docstring(func.__doc__)
    hints = _resolved_hints(func)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Prefer the resolved hint: under `from __future__ import annotations`
        # (and in Python 3.14+) param.annotation is the *string* "int", which
        # would silently degrade every argument to type "string".
        schema = _json_type(hints.get(name, param.annotation))
        if name in param_docs:
            schema["description"] = param_docs[name]
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(name)
        properties[name] = schema

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": summary or f"Call the {func.__name__} function.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool(func: Callable) -> Callable:
    """Decorator that attaches a generated JSON schema to a function.

    Usage::

        @tool
        def get_weather(city: str) -> str:
            '''Get today's weather.

            Args:
                city: City name, e.g. "Astana".
            '''
            ...

        get_weather.schema  # -> OpenAI function schema
    """
    func.schema = function_schema(func)  # type: ignore[attr-defined]
    func.is_tool = True  # type: ignore[attr-defined]
    return func


@dataclass
class ToolResult:
    """Outcome of one tool invocation - the unit an agent loop works in."""

    name: str
    arguments: dict
    ok: bool
    content: str
    seconds: float = 0.0

    def to_message(self, call_id: str | None = None) -> dict:
        """Format as a ``role: "tool"`` message to append to the transcript."""
        message = {"role": "tool", "name": self.name, "content": self.content}
        if call_id:
            message["tool_call_id"] = call_id
        return message


@dataclass
class ToolRegistry:
    """A named collection of callables plus their schemas.

    The registry is the *only* thing the agent is allowed to execute. If a
    function is not registered, the agent cannot call it - which is the whole
    security model. Keep the registry small and boring.
    """

    tools: dict[str, Callable] = field(default_factory=dict)

    def register(self, func: Callable, name: str | None = None) -> Callable:
        if not hasattr(func, "schema"):
            func = tool(func)
        key = name or func.schema["function"]["name"]
        self.tools[key] = func
        return func

    def add(self, *funcs: Callable) -> ToolRegistry:
        for func in funcs:
            self.register(func)
        return self

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def schemas(self) -> list[dict]:
        """The list you pass as ``tools=`` to a chat template or API call."""
        return [self.tools[name].schema for name in sorted(self.tools)]

    def describe(self) -> str:
        lines = [f"{len(self.tools)} tool(s) registered:"]
        for name in sorted(self.tools):
            desc = self.tools[name].schema["function"]["description"]
            params = self.tools[name].schema["function"]["parameters"]["properties"]
            lines.append(f"  - {name}({', '.join(params)}): {desc}")
        return "\n".join(lines)

    def call(self, name: str, arguments: dict | str | None = None) -> ToolResult:
        """Execute a tool by name. Never raises - failures come back as text.

        An agent loop that crashes on a bad tool call is useless; an agent loop
        that *tells the model what went wrong* usually recovers on the next
        iteration.
        """
        import time

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(name, {}, False, f"Error: arguments were not valid JSON ({exc}).")
        arguments = arguments or {}

        if name not in self.tools:
            available = ", ".join(self.names()) or "none"
            return ToolResult(
                name, arguments, False,
                f"Error: no tool named '{name}'. Available tools: {available}.",
            )

        func = self.tools[name]
        # Drop arguments the function does not accept rather than TypeError-ing:
        # small models routinely hallucinate one extra keyword.
        valid = set(inspect.signature(func).parameters)
        cleaned = {k: v for k, v in arguments.items() if k in valid}
        dropped = sorted(set(arguments) - valid)

        start = time.perf_counter()
        try:
            result = func(**cleaned)
            ok, content = True, _stringify(result)
        except ToolError as exc:
            ok, content = False, f"Error: {exc}"
        except TypeError as exc:
            ok, content = False, f"Error: wrong arguments for '{name}': {exc}"
        except Exception as exc:  # noqa: BLE001 - surfaced to the model, not swallowed
            ok, content = False, f"Error: {type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - start

        if dropped and ok:
            content += f"\n(note: ignored unknown argument(s): {', '.join(dropped)})"
        return ToolResult(name, cleaned, ok, content, seconds)


def _stringify(value: Any, limit: int = 4000) -> str:
    """Render a tool's return value as text for the model, with a size cap.

    The cap matters: one tool returning a 200 KB web page will blow your
    context window and silently truncate the conversation instead.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text

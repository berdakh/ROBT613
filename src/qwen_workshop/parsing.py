"""Tolerant parsing of model output.

Models are trained on text written by people, and people wrap JSON in prose and
markdown fences. These helpers accept that reality instead of fighting it.

Use them as a *fallback*. When your server supports constrained decoding
(notebook 07), prefer that - it makes malformed output impossible rather than
merely recoverable.
"""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list | None:
    """Pull the first JSON value out of a possibly chatty response.

    Tries, in order:

    1. the whole string,
    2. a ```json ...``` fenced block,
    3. the first balanced ``{...}`` or ``[...]`` span - whichever bracket
       appears first - respecting strings and escapes, so that a ``}`` inside
       a string does not end the scan early.

    Returns:
        The parsed value, or ``None`` if nothing parsed.
    """
    if not text:
        return None
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Consider whichever bracket appears first: a top-level array whose
    # elements are objects must not be mistaken for its own first object.
    candidates = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if o in text]
    for start, opener, closer in sorted(candidates):
        depth, in_string, escaped = 0, False, False
        for i, char in enumerate(text[start:], start):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def strip_code_fence(text: str) -> str:
    """Remove a surrounding ```lang ... ``` fence, if present."""
    match = re.fullmatch(r"\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text


def first_number(text: str) -> float | None:
    """Return the first number in the text - for 'reply with only the number' prompts."""
    match = re.search(r"-?\d+(?:[.,]\d+)?", text.replace(" ", ""))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None

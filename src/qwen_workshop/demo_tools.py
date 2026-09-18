"""A small library of safe, useful tools for the agent notebooks.

These operate on the sample data in ``data/``. They are written the way you
should write real tools:

* one job each, with a name and docstring the model can understand;
* every argument validated;
* :class:`~qwen_workshop.tools.ToolError` for expected failures, with a message
  that tells the model how to fix its call;
* file access confined to one directory;
* **no** ``eval``, ``exec``, shell, or unbounded network access.

Swap ``DATA_DIR`` for your own folder and these become genuinely useful.
"""

from __future__ import annotations

import ast
import csv
import datetime
import json
import operator
import re
from pathlib import Path
from typing import Literal

from .tools import ToolError, ToolRegistry, tool

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
NOTES_DIR = DATA_DIR / "notes"


# --------------------------------------------------------------------------
# General-purpose
# --------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@tool
def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(1200 + 800) * 3 / 2'.

    Args:
        expression: Arithmetic using numbers and + - * / // % ** and parentheses.
    """
    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ToolError("only numbers are allowed")
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            left, right = evaluate(node.left), evaluate(node.right)
            # Guard against 9**9**9 freezing the process.
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ToolError("exponent too large")
            return _ALLOWED_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](evaluate(node.operand))
        raise ToolError(
            "only arithmetic is supported - no names, calls, attributes or comparisons"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError(f"could not parse {expression!r} as an arithmetic expression") from None
    try:
        return evaluate(tree.body)
    except ZeroDivisionError:
        raise ToolError("division by zero") from None
    except OverflowError:
        raise ToolError("the result is too large") from None


@tool
def count_letters(word: str, letter: str) -> int:
    """Count how many times a letter appears in a word.

    Args:
        word: The word to inspect.
        letter: A single letter to count.
    """
    if len(letter) != 1:
        raise ToolError("letter must be exactly one character")
    return word.lower().count(letter.lower())


@tool
def current_date() -> str:
    """Return today's date as YYYY-MM-DD. Use this before any date arithmetic."""
    return datetime.date.today().isoformat()


@tool
def days_between(start: str, end: str) -> int:
    """Count days from `start` to `end`.

    Args:
        start: ISO date, YYYY-MM-DD.
        end: ISO date, YYYY-MM-DD.
    """
    try:
        first = datetime.date.fromisoformat(start)
        second = datetime.date.fromisoformat(end)
    except ValueError as exc:
        raise ToolError(f"dates must be YYYY-MM-DD ({exc})") from None
    return (second - first).days


# --------------------------------------------------------------------------
# Personal notes  (read-only, confined to NOTES_DIR)
# --------------------------------------------------------------------------


def _safe_note_path(filename: str) -> Path:
    """Resolve a note path, refusing anything outside the notes folder."""
    root = NOTES_DIR.resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        raise ToolError("access denied: path escapes the notes folder")
    return candidate


@tool
def list_notes() -> list[str]:
    """List the names of the user's available personal notes."""
    if not NOTES_DIR.exists():
        raise ToolError(f"notes folder not found at {NOTES_DIR}")
    return sorted(p.name for p in NOTES_DIR.glob("*.md"))


@tool
def read_note(filename: str) -> str:
    """Read one of the user's personal notes in full.

    Args:
        filename: Note file name, e.g. "cooking.md". Use list_notes first.
    """
    path = _safe_note_path(filename)
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in NOTES_DIR.glob("*.md")))
        raise ToolError(f"no note named {filename!r}. Available: {available}")
    return path.read_text(encoding="utf-8")[:6000]


@tool
def search_notes(query: str, max_results: int = 5) -> list[dict]:
    """Search the user's notes for a keyword and return matching lines.

    Args:
        query: Word or phrase to look for, case-insensitive.
        max_results: Maximum number of matching lines to return.
    """
    if not query.strip():
        raise ToolError("query must not be empty")
    pattern = re.compile(re.escape(query.strip()), re.IGNORECASE)

    results: list[dict] = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                results.append({"file": path.name, "line": number, "text": line.strip()})
                if len(results) >= max_results:
                    return results
    return results


# --------------------------------------------------------------------------
# Kitchen and budget
# --------------------------------------------------------------------------


@tool
def list_pantry() -> list[dict]:
    """List what is currently in the user's kitchen, with expiry dates."""
    path = DATA_DIR / "pantry.txt"
    if not path.exists():
        raise ToolError(f"pantry file not found at {path}")

    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            items.append({"item": parts[0], "quantity": parts[1],
                          "expires": None if parts[2] == "-" else parts[2]})
    return items


@tool
def expiring_soon(within_days: int = 3, today: str | None = None) -> list[dict]:
    """List pantry items expiring within a number of days.

    Args:
        within_days: How many days ahead to look.
        today: Reference date as YYYY-MM-DD. Defaults to the real today.
    """
    reference = datetime.date.fromisoformat(today) if today else datetime.date.today()
    soon = []
    for item in list_pantry():
        if not item["expires"]:
            continue
        try:
            expiry = datetime.date.fromisoformat(item["expires"])
        except ValueError:
            continue
        days = (expiry - reference).days
        if days <= within_days:
            soon.append({**item, "days_left": days})
    return sorted(soon, key=lambda row: row["days_left"])


def _load_expenses() -> list[dict]:
    path = DATA_DIR / "expenses.csv"
    if not path.exists():
        raise ToolError(f"expenses file not found at {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["amount_kzt"] = float(row["amount_kzt"])
    return rows


@tool
def spending_by_category(month: str | None = None) -> dict:
    """Total the user's spending per category, in tenge.

    Args:
        month: Restrict to one month, as YYYY-MM. Omit for all data.
    """
    rows = _load_expenses()
    if month:
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise ToolError("month must look like 2026-01")
        rows = [r for r in rows if r["date"].startswith(month)]
        if not rows:
            months = sorted({r["date"][:7] for r in _load_expenses()})
            raise ToolError(f"no expenses in {month}. Available months: {', '.join(months)}")

    totals: dict[str, float] = {}
    for row in rows:
        totals[row["category"]] = totals.get(row["category"], 0.0) + row["amount_kzt"]
    return {
        "month": month or "all",
        "total_kzt": round(sum(totals.values()), 2),
        "by_category": {k: round(v, 2) for k, v in sorted(totals.items(), key=lambda i: -i[1])},
        "transactions": len(rows),
    }


@tool
def largest_expenses(limit: int = 5, category: str | None = None) -> list[dict]:
    """List the user's biggest individual purchases.

    Args:
        limit: How many to return.
        category: Optional category filter, e.g. "groceries".
    """
    rows = _load_expenses()
    if category:
        known = sorted({r["category"] for r in rows})
        if category not in known:
            raise ToolError(f"unknown category {category!r}. Known: {', '.join(known)}")
        rows = [r for r in rows if r["category"] == category]
    rows.sort(key=lambda r: -r["amount_kzt"])
    return rows[: max(1, min(limit, 50))]


@tool
def read_inbox(limit: int = 10) -> list[dict]:
    """Read the user's recent email messages.

    Args:
        limit: Maximum number of messages to return.
    """
    path = DATA_DIR / "inbox.jsonl"
    if not path.exists():
        raise ToolError(f"inbox file not found at {path}")
    messages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return messages[: max(1, min(limit, 100))]


# --------------------------------------------------------------------------
# A tool that deliberately requires confirmation
# --------------------------------------------------------------------------

SENT_MESSAGES: list[dict] = []


@tool
def send_message(recipient: str, body: str, confirmed: Literal["yes", "no"] = "no") -> str:
    """Send a message to someone. Requires explicit confirmation first.

    Args:
        recipient: Email address or name of the person.
        body: The message text to send.
        confirmed: Must be "yes" to actually send. Ask the user before setting it.
    """
    if confirmed != "yes":
        return (f"NOT SENT. Confirmation required. Show the user this draft to "
                f"{recipient}:\n{body}\nThen call again with confirmed='yes'.")
    SENT_MESSAGES.append({"to": recipient, "body": body,
                          "at": datetime.datetime.now().isoformat(timespec="seconds")})
    return f"Sent to {recipient}. (Simulated - nothing left this machine.)"


# --------------------------------------------------------------------------
# Ready-made registries
# --------------------------------------------------------------------------

def basic_registry() -> ToolRegistry:
    """Arithmetic, dates and letter counting - the notebook 10 starter set."""
    return ToolRegistry().add(calculate, count_letters, current_date, days_between)


def daily_life_registry() -> ToolRegistry:
    """Everything a personal assistant needs for the capstone."""
    return ToolRegistry().add(
        calculate, count_letters, current_date, days_between,
        list_notes, read_note, search_notes,
        list_pantry, expiring_soon,
        spending_by_category, largest_expenses,
        read_inbox, send_message,
    )

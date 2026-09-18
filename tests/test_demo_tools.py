"""The daily-life tools, with emphasis on the ways they must refuse."""

from __future__ import annotations

import pytest

from qwen_workshop.demo_tools import SENT_MESSAGES, daily_life_registry
from qwen_workshop.tools import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return daily_life_registry()


# --------------------------------------------------------------------------
# calculate: the security-critical one
# --------------------------------------------------------------------------


def test_calculate_does_arithmetic(registry):
    assert registry.call("calculate", {"expression": "(1200 + 800) * 3 / 2"}).content == "3000.0"


def test_calculate_respects_precedence(registry):
    assert registry.call("calculate", {"expression": "2 + 3 * 4"}).content == "14"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "exec('x=1')",
        "[i for i in range(10)]",
        "lambda: 1",
        "1 if True else 2",
        "print(1)",
    ],
)
def test_calculate_refuses_anything_that_is_not_arithmetic(registry, expression):
    """No model output should ever reach an interpreter. These must all fail."""
    result = registry.call("calculate", {"expression": expression})
    assert not result.ok
    assert "Error" in result.content


def test_calculate_refuses_huge_exponents(registry):
    """9**9**9 would otherwise hang the process."""
    result = registry.call("calculate", {"expression": "9 ** 9 ** 9"})
    assert not result.ok


def test_calculate_reports_division_by_zero(registry):
    result = registry.call("calculate", {"expression": "1 / 0"})
    assert not result.ok
    assert "zero" in result.content


# --------------------------------------------------------------------------
# File access confinement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    ["../../etc/passwd", "../../../../../../etc/shadow", "/etc/passwd", "..%2F..%2Fetc"],
)
def test_read_note_refuses_paths_outside_the_notes_folder(registry, filename):
    result = registry.call("read_note", {"filename": filename})
    assert not result.ok
    assert "denied" in result.content or "no note named" in result.content


def test_read_note_reads_a_real_note(registry):
    result = registry.call("read_note", {"filename": "cooking.md"})
    assert result.ok
    assert "lentil" in result.content.lower()


def test_missing_note_lists_the_alternatives(registry):
    """A good error message is a prompt: tell the model what it *can* do."""
    result = registry.call("read_note", {"filename": "nope.md"})
    assert not result.ok
    assert "cooking.md" in result.content


def test_list_notes(registry):
    result = registry.call("list_notes", {})
    assert result.ok
    assert "cooking.md" in result.content and "study.md" in result.content


def test_search_notes_finds_a_line(registry):
    result = registry.call("search_notes", {"query": "radiator"})
    assert result.ok
    assert "boiler" in result.content


def test_search_notes_is_case_insensitive(registry):
    assert "boiler" in registry.call("search_notes", {"query": "RADIATOR"}).content


def test_search_notes_rejects_empty_query(registry):
    assert not registry.call("search_notes", {"query": "   "}).ok


# --------------------------------------------------------------------------
# Data tools
# --------------------------------------------------------------------------


def test_spending_by_category_totals(registry):
    import json

    result = registry.call("spending_by_category", {})
    assert result.ok
    data = json.loads(result.content)
    assert data["total_kzt"] == pytest.approx(sum(data["by_category"].values()), abs=1.0)
    assert data["transactions"] > 0


def test_spending_rejects_a_bad_month_format(registry):
    result = registry.call("spending_by_category", {"month": "January"})
    assert not result.ok
    assert "2026-01" in result.content


def test_spending_for_an_empty_month_lists_available_months(registry):
    result = registry.call("spending_by_category", {"month": "1999-01"})
    assert not result.ok
    assert "2026-01" in result.content


def test_largest_expenses_is_sorted_descending(registry):
    import json

    rows = json.loads(registry.call("largest_expenses", {"limit": 5}).content)
    amounts = [r["amount_kzt"] for r in rows]
    assert amounts == sorted(amounts, reverse=True)


def test_largest_expenses_rejects_unknown_category(registry):
    result = registry.call("largest_expenses", {"category": "yachts"})
    assert not result.ok
    assert "groceries" in result.content


def test_expiring_soon_is_deterministic_with_a_fixed_today(registry):
    import json

    rows = json.loads(registry.call("expiring_soon", {"within_days": 3, "today": "2026-03-15"}).content)
    assert rows, "the sample pantry should have items expiring in March 2026"
    assert rows == sorted(rows, key=lambda r: r["days_left"])
    assert all(r["days_left"] <= 3 for r in rows)


def test_days_between(registry):
    assert registry.call("days_between", {"start": "2026-03-01", "end": "2026-03-15"}).content == "14"


def test_days_between_rejects_bad_dates(registry):
    assert not registry.call("days_between", {"start": "1 March", "end": "2026-03-15"}).ok


def test_count_letters_requires_a_single_character(registry):
    assert registry.call("count_letters", {"word": "strawberry", "letter": "r"}).content == "3"
    assert not registry.call("count_letters", {"word": "abc", "letter": "ab"}).ok


def test_read_inbox(registry):
    assert "midterm" in registry.call("read_inbox", {"limit": 3}).content.lower()


# --------------------------------------------------------------------------
# Human-in-the-loop
# --------------------------------------------------------------------------


def test_send_message_requires_confirmation(registry):
    before = len(SENT_MESSAGES)
    result = registry.call("send_message", {"recipient": "a@b.kz", "body": "hello"})
    assert result.ok  # it "succeeded" - by refusing to send
    assert "NOT SENT" in result.content
    assert len(SENT_MESSAGES) == before, "nothing may be sent without confirmation"


def test_send_message_sends_once_confirmed(registry):
    before = len(SENT_MESSAGES)
    result = registry.call(
        "send_message", {"recipient": "a@b.kz", "body": "hello", "confirmed": "yes"}
    )
    assert result.ok and "Sent to" in result.content
    assert len(SENT_MESSAGES) == before + 1


def test_every_tool_has_a_description_and_typed_arguments(registry):
    """A tool with a vague schema is a tool the model will misuse."""
    for schema in registry.schemas():
        fn = schema["function"]
        assert fn["description"] and not fn["description"].startswith("Call the"), fn["name"]
        for arg, spec in fn["parameters"]["properties"].items():
            assert "type" in spec, f"{fn['name']}.{arg}"
            assert "description" in spec, f"{fn['name']}.{arg} has no description"

"""Tolerant parsing of messy model output."""

from __future__ import annotations

from qwen_workshop.parsing import extract_json, first_number, strip_code_fence


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('Here you go:\n```json\n{"a": 2}\n```\nHope that helps!') == {"a": 2}


def test_unlabelled_fence():
    assert extract_json("```\n[1, 2, 3]\n```") == [1, 2, 3]


def test_json_embedded_in_prose():
    assert extract_json('The result is {"a": 3} - anything else?') == {"a": 3}


def test_brace_inside_string_does_not_end_the_scan():
    text = 'Result: {"note": "contains a } brace", "ok": true} done'
    assert extract_json(text) == {"note": "contains a } brace", "ok": True}


def test_escaped_quote_inside_string():
    assert extract_json(r'{"q": "she said \"hi\""}') == {"q": 'she said "hi"'}


def test_nested_objects():
    assert extract_json('blah {"a": {"b": [1, {"c": 2}]}} blah') == {"a": {"b": [1, {"c": 2}]}}


def test_array_at_top_level():
    assert extract_json('Sure: [{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]


def test_no_json_returns_none():
    assert extract_json("I'm sorry, I cannot do that.") is None
    assert extract_json("") is None
    assert extract_json("{ definitely not json") is None


def test_strip_code_fence():
    assert strip_code_fence("```python\nprint(1)\n```") == "print(1)"
    assert strip_code_fence("no fence here") == "no fence here"


def test_first_number():
    assert first_number("The answer is 3 letters.") == 3.0
    assert first_number("about -2.5 degrees") == -2.5
    assert first_number("no digits") is None

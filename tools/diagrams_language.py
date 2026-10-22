#!/usr/bin/env python3
"""Diagram for the Kazakh/Russian material (notebook 22).

Imported by ``make_diagrams.py``. Returns ``(filename, svg, caption)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import Canvas, diagram, wrap  # noqa: E402


@diagram
def language_tax():
    c = Canvas(900, 470)
    c.title("The tax on Kazakh and Russian",
            "Each step multiplies the one before it. Only the first is forced by the script.")

    rows = [
        ("The same sentence", "~60 characters either way", "default",
         "no penalty yet"),
        ("UTF-8 bytes", "1 byte/char Latin · 2 bytes/char Cyrillic", "warn",
         "2x · exact, unavoidable"),
        ("Tokens", "English merges into words; Cyrillic stays near single letters",
         "bad", "2-3x · a choice, not a law"),
    ]
    y = 92
    for label, detail, role, note in rows:
        c.box(28, y, 250, 56, label, role=role)
        c.text(296, y + 24, detail, "small", anchor="start")
        c.text(296, y + 43, note, "small t-warn" if role != "default" else "small",
               anchor="start")
        if y < 200:
            c.arrow(153, y + 56, 153, y + 72)
        y += 76

    c.rule(28, 302, 872, 302)
    c.text(28, 326, "And then the token multiplier hits three things at once:",
           "lbl", anchor="start")

    for x, title, detail in (
        (28, "Context", "a 32k window holds ~131k chars of English but only ~65k of Kazakh"),
        (320, "Cost", "twice the bill for the same content"),
        (612, "Latency", "twice the time to stream the same answer"),
    ):
        c.box(x, 344, 260, 62, title, role="bad")
        # Wrap on word boundaries - slicing the string mid-word leaves
        # "Englis / h" in the caption.
        for line_number, line in enumerate(wrap(detail, 250, 11.5)):
            c.text(x + 130, 424 + line_number * 17, line, "small")

    c.text(28, 464, "Root cause: not the script and not the grammar - the "
                    "quantity of varied Kazakh text on the public internet.",
           "small", anchor="start")

    return "lang-tax.svg", c.render(
        "Three stacked rows showing how the cost compounds: the same sentence "
        "is about sixty characters in either language; UTF-8 makes Cyrillic "
        "two bytes per character, an exact and unavoidable doubling; and "
        "tokenization adds a further two to three times because English "
        "merges into whole words while Cyrillic stays near single letters. "
        "The resulting token multiplier then hits context, cost and latency "
        "simultaneously."
    ), "How the cost compounds for Cyrillic text, and what it hits."

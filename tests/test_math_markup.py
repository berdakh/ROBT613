"""Maths on the course site must use `$$...$$`, never single dollars.

kramdown turns `$$...$$` into \\[...\\] (standalone) or \\(...\\) (inline),
which MathJax 3 renders with its default delimiters. It leaves single
`$...$` completely alone, so it reaches the reader as literal dollar signs.

That failure is silent - the page builds, deploys and looks broken - so it is
worth a test rather than a convention. Verified by writing `$d_k$` into a page
and watching this fail.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A single $...$ that is not part of a $$...$$ pair. Requires a non-space
# first character so prose like "costs $5 to $10" is not flagged.
SINGLE_DOLLAR = re.compile(r"(?<!\$)\$(?!\$)([^\s$][^$\n]*?)\$(?!\$)")
# LaTeX-ish content: a backslash command, a subscript, or a superscript.
LOOKS_LIKE_MATH = re.compile(r"\\[a-zA-Z]+|[_^]\{|[a-zA-Z]_[a-zA-Z0-9]")
FENCE = re.compile(r"^\s*(```|~~~)")


def strip_code(text: str) -> str:
    """Blank fenced code: `$x$` inside a Python string is not maths."""
    out, inside = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out)


def markdown_sources() -> list[Path]:
    files = sorted((REPO_ROOT / "docs").rglob("*.md"))
    files += sorted((REPO_ROOT / "notebook_src").glob("*.py"))
    return files


def test_there_are_sources_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(markdown_sources()) > 30


def test_no_single_dollar_maths():
    offenders: list[str] = []
    for path in markdown_sources():
        for number, line in enumerate(strip_code(path.read_text(encoding="utf-8")).splitlines(), 1):
            for match in SINGLE_DOLLAR.finditer(line):
                if LOOKS_LIKE_MATH.search(match.group(1)):
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT)}:{number}: {match.group(0)}"
                    )
    assert not offenders, (
        "single-dollar maths renders as literal dollar signs - use $$...$$:\n  "
        + "\n  ".join(offenders)
    )

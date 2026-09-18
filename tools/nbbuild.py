#!/usr/bin/env python3
"""Build .ipynb notebooks from jupytext-style "percent" .py sources.

Why this exists
---------------
Notebook JSON is painful to review in a pull request: one changed word shows up
as a rewritten blob. So the *source of truth* for every workshop notebook is a
plain Python file in ``notebook_src/`` using the percent format, and this script
renders the ``notebooks/*.ipynb`` that students actually open.

Percent format recap::

    # %% [markdown]
    # # A heading
    # Some prose.

    # %%
    print("a code cell")

Usage::

    python tools/nbbuild.py            # build every source
    python tools/nbbuild.py --check    # fail if notebooks are stale (CI)
    python tools/nbbuild.py 03_first_generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "notebook_src"
OUT_DIR = REPO_ROOT / "notebooks"

KERNELSPEC = {
    "display_name": "Python 3 (qwen-workshop)",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}


def _as_source_lines(text: str) -> list[str]:
    """Split a cell body into nbformat's list-of-lines-with-newlines shape."""
    text = text.rstrip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _strip_markdown_comments(body: str) -> str:
    """Turn ``# some text`` comment lines back into raw markdown."""
    out = []
    for line in body.split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            # A markdown cell should be fully commented; keep anything odd
            # verbatim rather than silently dropping content.
            out.append(line)
    return "\n".join(out)


def parse_percent(text: str) -> list[dict]:
    """Parse percent-format text into a list of nbformat cell dicts."""
    cells: list[dict] = []
    current_kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_kind is None:
            return
        body = "\n".join(buffer).strip("\n")
        if not body.strip():
            return
        if current_kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _as_source_lines(_strip_markdown_comments(body)),
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": _as_source_lines(body),
                }
            )

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            buffer = []
            current_kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        if current_kind is None:
            # Anything before the first cell marker (module docstring, etc.)
            continue
        buffer.append(line)

    flush()
    return cells


def build_notebook(src: Path) -> dict:
    cells = parse_percent(src.read_text(encoding="utf-8"))
    if not cells:
        raise ValueError(f"{src.name} produced no cells - is it percent-formatted?")
    return {
        "cells": cells,
        "metadata": {"kernelspec": KERNELSPEC, "language_info": LANGUAGE_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="build only sources matching these substrings")
    parser.add_argument("--check", action="store_true", help="verify notebooks are up to date")
    args = parser.parse_args()

    sources = sorted(SRC_DIR.glob("*.py"))
    if args.only:
        sources = [s for s in sources if any(token in s.name for token in args.only)]
    if not sources:
        print("no notebook sources matched", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    stale: list[str] = []
    for src in sources:
        out = OUT_DIR / (src.stem + ".ipynb")
        text = render(build_notebook(src))
        if args.check:
            if not out.exists() or out.read_text(encoding="utf-8") != text:
                stale.append(out.name)
            continue
        out.write_text(text, encoding="utf-8")
        n_cells = len(json.loads(text)["cells"])
        print(f"built {out.relative_to(REPO_ROOT)}  ({n_cells} cells)")

    if args.check:
        if stale:
            print("stale notebooks (run `python tools/nbbuild.py`):", file=sys.stderr)
            for name in stale:
                print(f"  - {name}", file=sys.stderr)
            return 1
        print(f"all {len(sources)} notebooks up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

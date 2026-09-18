#!/usr/bin/env python3
"""Validate the generated notebooks.

Checks, for every notebook in ``notebooks/``:

* it is valid JSON with the nbformat structure Jupyter expects;
* every code cell parses as Python (catching typos that would otherwise only
  surface when a student runs the cell);
* outputs are empty, so the repo stays small and diffs stay readable;
* every relative link in a markdown cell points at a file that exists.

Run it with ``python tools/check_notebooks.py`` or via ``make check``.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

# Lines Jupyter understands but Python does not.
MAGIC = re.compile(r"^\s*[%!]")


def check_notebook(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]

    if nb.get("nbformat") != 4:
        problems.append(f"nbformat is {nb.get('nbformat')}, expected 4")
    if "kernelspec" not in nb.get("metadata", {}):
        problems.append("missing kernelspec metadata")

    for number, cell in enumerate(nb.get("cells", []), 1):
        kind = cell.get("cell_type")
        source = "".join(cell.get("source", []))

        if kind == "code":
            if cell.get("outputs"):
                problems.append(f"cell {number}: has saved outputs (should be cleared)")
            # Skip magics/shell lines, which are not valid Python.
            code = "\n".join("pass" if MAGIC.match(line) else line for line in source.split("\n"))
            try:
                ast.parse(code)
            except SyntaxError as exc:
                problems.append(f"cell {number}: SyntaxError line {exc.lineno}: {exc.msg}")

        elif kind == "markdown":
            for text, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", source):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                resolved = (path.parent / target.split("#")[0]).resolve()
                if not resolved.exists():
                    problems.append(f"cell {number}: broken link [{text}]({target})")

    return problems


def main() -> int:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print("no notebooks found - run python tools/nbbuild.py first", file=sys.stderr)
        return 1

    total_problems = 0
    for path in notebooks:
        problems = check_notebook(path)
        total_problems += len(problems)
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        cells = len(json.loads(path.read_text(encoding='utf-8'))["cells"])
        print(f"{path.name:<44}{cells:>3} cells  {status}")
        for problem in problems:
            print(f"    - {problem}")

    print()
    if total_problems:
        print(f"FAILED: {total_problems} problem(s) across {len(notebooks)} notebooks")
        return 1
    print(f"All {len(notebooks)} notebooks valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

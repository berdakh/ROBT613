#!/usr/bin/env python3
"""Check that every relative link in the markdown files points at a real file.

Covers README.md and everything under docs/. Jekyll serves ``foo.md`` as
``foo.html``, so a link to ``guides/setup.html`` is satisfied by
``guides/setup.md``.

External links (http/https) are not fetched - that would make the check slow
and flaky.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def resolve(source: Path, target: str) -> bool:
    """Is `target`, written inside `source`, a file that exists?"""
    clean = target.split("#")[0]
    if not clean:
        return True  # a pure anchor
    base = source.parent
    candidates = [base / clean]
    if clean.endswith(".html"):
        candidates.append((base / clean).with_suffix(".md"))
    if clean.endswith("/"):
        candidates.append(base / clean / "index.md")
    return any(candidate.exists() for candidate in candidates)


def main() -> int:
    files = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    files += sorted((REPO_ROOT / "data").rglob("*.md"))

    problems: list[str] = []
    checked = 0
    for path in files:
        if not path.exists():
            continue
        for label, target in LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            checked += 1
            if not resolve(path, target):
                problems.append(f"{path.relative_to(REPO_ROOT)}: [{label}]({target})")

    print(f"checked {checked} relative links across {len(files)} files")
    if problems:
        print(f"\n{len(problems)} broken link(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("all relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

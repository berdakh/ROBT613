"""Keep solutions/ in sync with the notebooks.

Exercises get added and renumbered as the material evolves, and an instructor
discovering a missing solution five minutes before a session is exactly the
failure this guards against.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "notebook_src"
SOLUTIONS = REPO_ROOT / "solutions"


def notebook_exercises() -> list[tuple[str, str]]:
    """Return (notebook number, exercise number) for every exercise found."""
    found: list[tuple[str, str]] = []
    for src in sorted(SRC_DIR.glob("*.py")):
        number = src.name[:2]
        for match in re.finditer(r"\*\*Exercise (\d+)\.\*\*", src.read_text(encoding="utf-8")):
            found.append((number, match.group(1)))
    return found


def checkpoint_sections() -> list[str]:
    """Notebook numbers that end with a Checkpoint section."""
    return [
        src.name[:2]
        for src in sorted(SRC_DIR.glob("*.py"))
        if "# ## Checkpoint" in src.read_text(encoding="utf-8")
    ]


def test_solutions_directory_exists():
    assert (SOLUTIONS / "README.md").is_file()
    assert (SOLUTIONS / "exercises.md").is_file()
    assert (SOLUTIONS / "checkpoints.md").is_file()


def test_there_are_exercises_to_solve():
    """Guard against the parser silently matching nothing."""
    assert len(notebook_exercises()) >= 15


@pytest.mark.parametrize("notebook,number", notebook_exercises())
def test_every_exercise_has_a_solution(notebook, number):
    """Each `Exercise N` in notebook NN needs an `NN-N` heading in exercises.md."""
    text = (SOLUTIONS / "exercises.md").read_text(encoding="utf-8")
    assert f"{notebook}-{number}" in text, (
        f"notebook {notebook} exercise {number} has no solution. "
        f"Add a '## {notebook}-{number}' section to solutions/exercises.md."
    )


@pytest.mark.parametrize("notebook", checkpoint_sections())
def test_every_checkpoint_section_is_answered(notebook):
    """Each notebook with a Checkpoint needs a matching section in checkpoints.md."""
    text = (SOLUTIONS / "checkpoints.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {notebook} ", text, re.M), (
        f"notebook {notebook} has checkpoint questions but no answers. "
        f"Add a '## {notebook} · ...' section to solutions/checkpoints.md."
    )


def test_confidence_markers_are_used():
    """Answers must carry a marker, so nobody mistakes an estimate for a fact."""
    text = (SOLUTIONS / "exercises.md").read_text(encoding="utf-8")
    for marker in ("✅", "🔬", "✍️"):
        assert marker in text, f"missing confidence marker {marker}"


def test_readme_explains_the_markers():
    text = (SOLUTIONS / "README.md").read_text(encoding="utf-8")
    for marker in ("✅", "🔬", "✍️"):
        assert marker in text

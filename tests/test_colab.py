"""Colab helpers. These must be safe to call on a laptop, which is most of the point."""

from __future__ import annotations

import sys

import pytest

from qwen_workshop import colab


def test_in_colab_is_false_here():
    assert colab.in_colab() is False


def test_notebook_url_shape():
    url = colab.notebook_url("notebooks/01_llm_foundations.ipynb")
    assert url == (
        "https://colab.research.google.com/github/berdakh/ROBT613/blob/master/"
        "notebooks/01_llm_foundations.ipynb"
    )


def test_notebook_url_tolerates_a_leading_slash():
    assert colab.notebook_url("/notebooks/x.ipynb").endswith("/master/notebooks/x.ipynb")


def test_notebook_url_honours_branch_and_fork():
    url = colab.notebook_url("notebooks/x.ipynb", branch="dev", owner_repo="someone/fork")
    assert "/someone/fork/blob/dev/" in url


def test_setup_outside_colab_finds_the_repo_and_does_not_clone(tmp_path, monkeypatch):
    """The dangerous failure would be cloning onto a student's own machine."""
    called = []
    monkeypatch.setattr(colab, "_run", lambda cmd, quiet: called.append(cmd))

    root = colab.setup()

    assert called == [], "setup() must not run git or pip outside Colab"
    assert (root / "src" / "qwen_workshop").is_dir()


def test_setup_puts_src_on_the_path():
    root = colab.setup()
    assert str(root / "src") in sys.path


def test_find_root_walks_upwards(tmp_path):
    nested = tmp_path / "src" / "qwen_workshop"
    nested.mkdir(parents=True)
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert colab._find_root(deep) == tmp_path.resolve()


def test_find_root_falls_back_to_cwd(tmp_path):
    """No repo anywhere above? Return something rather than raising."""
    assert colab._find_root(tmp_path) == tmp_path.resolve()


def test_start_ollama_refuses_to_install_on_a_real_machine(monkeypatch):
    """Auto-installing software on someone's laptop would be rude."""
    monkeypatch.setattr(colab, "in_colab", lambda: False)
    monkeypatch.setattr("qwen_workshop.client.is_up", lambda *a, **k: False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="https://ollama.com"):
        colab.start_ollama()


def test_start_ollama_is_a_noop_when_already_running(monkeypatch):
    monkeypatch.setattr("qwen_workshop.client.is_up", lambda *a, **k: True)
    assert colab.start_ollama(pull=False) is True

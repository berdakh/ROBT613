"""Make the workshop notebooks work in Google Colab.

A notebook opened from a Colab badge arrives **alone**: Colab fetches that one
``.ipynb`` from GitHub and nothing else. There is no ``src/``, no ``data/``, and
none of the dependencies. Every helper here exists to close that gap.

Nothing in this module does anything outside Colab - :func:`setup` and
:func:`start_ollama` return immediately elsewhere - so the same cell is safe to
run on a laptop.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/berdakh/ROBT613.git"
REPO_DIR = "ROBT613"
DEFAULT_BRANCH = "master"


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def setup(
    *,
    branch: str = DEFAULT_BRANCH,
    install: bool = True,
    quiet: bool = True,
    repo_url: str = REPO_URL,
) -> Path:
    """Clone the workshop repo and install dependencies, if in Colab.

    Outside Colab this only locates the repository root, so notebooks can call
    it unconditionally in their first cell.

    Args:
        branch: Branch to clone.
        install: Whether to pip install ``requirements.txt``.
        quiet: Suppress pip/git chatter.
        repo_url: Override for a fork.

    Returns:
        The repository root, already made the current working directory in
        Colab and added to ``sys.path`` so ``import qwen_workshop`` works.
    """
    if not in_colab():
        root = _find_root()
        _add_src_to_path(root)
        return root

    target = Path("/content") / REPO_DIR
    if not target.exists():
        print(f"Cloning {repo_url} ({branch}) ...")
        _run(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(target)], quiet)
    else:
        print(f"{target} already present - skipping clone.")

    os.chdir(target)

    if install:
        print("Installing requirements (this takes a minute the first time) ...")
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], quiet)

    _add_src_to_path(target)
    print(f"Ready. Working directory: {target}")
    return target


def _find_root(start: Path | None = None) -> Path:
    """Walk upwards looking for the repository root."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "qwen_workshop").is_dir():
            return candidate
    return here


def _add_src_to_path(root: Path) -> None:
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _run(command: list[str], quiet: bool) -> None:
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
    )


def start_ollama(
    model: str = "qwen3:0.6b",
    *,
    timeout: float = 300.0,
    pull: bool = True,
) -> bool:
    """Install and start an Ollama server, then pull a model.

    Notebooks 06, 07, 09, 10 and 12 need a local OpenAI-compatible server.
    Colab has no Ollama, so this installs it, launches ``ollama serve`` in the
    background, and waits until the API answers.

    Expect this to take two to four minutes the first time. On a CPU-only Colab
    runtime, stay with ``qwen3:0.6b``.

    Args:
        model: Tag to pull, e.g. ``"qwen3:4b"``.
        timeout: Seconds to wait for the server and the pull.
        pull: Set False to start the server without downloading anything.

    Returns:
        True if the server is answering.
    """
    import shutil

    from .client import is_up

    if is_up("ollama"):
        print("Ollama is already running.")
        if pull:
            _pull(model, timeout)
        return True

    if shutil.which("ollama") is None:
        if not in_colab():
            raise RuntimeError(
                "Ollama is not installed. On your own machine install it from "
                "https://ollama.com rather than letting a notebook do it."
            )
        print("Installing Ollama ...")
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )

    print("Starting the Ollama server ...")
    # Detached, with output discarded: a live server must outlive this cell.
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_up("ollama"):
            print("Server is up at http://localhost:11434")
            if pull:
                _pull(model, timeout)
            return True
        time.sleep(1.5)

    print(f"Ollama did not answer within {timeout:.0f}s. Try running this cell again.")
    return False


def _pull(model: str, timeout: float) -> None:
    print(f"Pulling {model} (a few hundred MB) ...")
    try:
        subprocess.run(["ollama", "pull", model], check=True, timeout=timeout)
        print(f"{model} ready.")
    except subprocess.TimeoutExpired:
        print(f"Pull of {model} timed out - re-run the cell; downloads resume.")
    except subprocess.CalledProcessError as exc:
        print(f"Pull failed ({exc}). Check the tag at https://ollama.com/library/qwen3")


def notebook_url(notebook: str, *, branch: str = DEFAULT_BRANCH,
                 owner_repo: str = "berdakh/ROBT613") -> str:
    """Build the Colab URL for a notebook in this repo.

    >>> notebook_url("notebooks/01_llm_foundations.ipynb")
    'https://colab.research.google.com/github/berdakh/ROBT613/blob/master/notebooks/01_llm_foundations.ipynb'
    """
    notebook = notebook.lstrip("/")
    return f"https://colab.research.google.com/github/{owner_repo}/blob/{branch}/{notebook}"

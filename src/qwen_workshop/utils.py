"""Small, dependency-free helpers used throughout the notebooks."""

from __future__ import annotations

import os
import random
import textwrap
import time
from contextlib import contextmanager


def human_bytes(num_bytes: float) -> str:
    """Format a byte count the way a human would say it out loud."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < step or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= step
    return f"{num_bytes:.1f} TB"


def wrap_print(text: str, width: int = 88, prefix: str = "") -> None:
    """Print long model output without it running off the side of the screen."""
    for paragraph in str(text).split("\n"):
        if not paragraph.strip():
            print(prefix)
            continue
        for line in textwrap.wrap(paragraph, width=width):
            print(prefix + line)


def seed_everything(seed: int = 613) -> int:
    """Seed Python, NumPy and PyTorch if they are present.

    Note: seeding makes *sampling* reproducible, not deterministic across
    machines - different GPUs and different batch sizes still change results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


class Timer:
    """Context manager that reports wall time, and tokens/second if you ask.

    Example::

        with Timer("generation") as t:
            out = model.generate(**inputs)
        print(t.tokens_per_second(n_tokens=out.shape[-1]))
    """

    def __init__(self, label: str = "block", verbose: bool = True) -> None:
        self.label = label
        self.verbose = verbose
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.verbose:
            print(f"[{self.label}] {self.elapsed:.2f}s")

    def tokens_per_second(self, n_tokens: int) -> float:
        if self.elapsed <= 0:
            return float("nan")
        return n_tokens / self.elapsed


@contextmanager
def section(title: str, char: str = "=", width: int = 72):
    """Print a labelled banner around a block of notebook output."""
    print(char * width)
    print(title)
    print(char * width)
    yield
    print()

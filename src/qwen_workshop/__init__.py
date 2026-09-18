"""Helper library for the ROBT613 Qwen open-weight models workshop.

Everything here is deliberately small and readable: students are expected to
open these files, not just import them. Nothing is hidden behind a framework.

Import the pieces you need, e.g.::

    from qwen_workshop import MODELS, check_environment, load_model, chat
"""

from __future__ import annotations

__version__ = "1.0.0"

from .config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    MODELS,
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    SamplingParams,
    recommend_model,
)
from .env import check_environment, describe_environment, pick_device
from .utils import Timer, human_bytes, seed_everything, wrap_print

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "MODELS",
    "NON_THINKING_SAMPLING",
    "THINKING_SAMPLING",
    "SamplingParams",
    "Timer",
    "check_environment",
    "describe_environment",
    "human_bytes",
    "pick_device",
    "recommend_model",
    "seed_everything",
    "wrap_print",
    "__version__",
]


def __getattr__(name: str):
    """Lazily expose the heavy modules.

    ``load_model`` pulls in torch/transformers, which takes seconds and is not
    installed in every environment (e.g. an Ollama-only setup). Importing
    ``qwen_workshop`` itself must stay instant and dependency-free.
    """
    lazy = {
        "load_model": ("loading", "load_model"),
        "load_tokenizer": ("loading", "load_tokenizer"),
        "chat": ("chat", "chat"),
        "stream_chat": ("chat", "stream_chat"),
        "split_thinking": ("chat", "split_thinking"),
        "get_client": ("client", "get_client"),
        "Agent": ("agent", "Agent"),
        "ToolRegistry": ("tools", "ToolRegistry"),
        "tool": ("tools", "tool"),
    }
    if name in lazy:
        module_name, attr = lazy[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Talking to a local Qwen server through the OpenAI-compatible API.

Once you run Qwen behind vLLM, Ollama, llama.cpp's server or LM Studio, they
all speak the same HTTP dialect. That is the single most useful fact in this
whole workshop: **your application code does not change when you swap the
model or the runtime.** Only `base_url` and `model` change.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default endpoints for the runtimes we cover in notebooks 05-06.
BACKENDS: dict[str, dict] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # ignored, but the OpenAI SDK requires something
        "model": "qwen3:0.6b",
        "start": "ollama serve   (then: ollama pull qwen3:0.6b)",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "model": "Qwen/Qwen3-0.6B",
        "start": "vllm serve Qwen/Qwen3-0.6B --max-model-len 8192",
    },
    "llamacpp": {
        "base_url": "http://localhost:8080/v1",
        "api_key": "no-key",
        "model": "qwen3-0.6b",
        "start": "llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q4_K_M -c 8192",
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "qwen3-0.6b",
        "start": "Start the LM Studio local server from its Developer tab.",
    },
}


@dataclass
class Backend:
    """A resolved local endpoint: everything needed to send a request."""

    name: str
    base_url: str
    api_key: str
    model: str

    def __repr__(self) -> str:
        return f"<Backend {self.name} model={self.model} at {self.base_url}>"


def get_client(backend: str = "ollama", *, model: str | None = None, base_url: str | None = None):
    """Return ``(openai_client, Backend)`` for a local server.

    Args:
        backend: one of ``ollama``, ``vllm``, ``llamacpp``, ``lmstudio``.
        model: override the default model name for that backend.
        base_url: override the URL (e.g. a lab server on another machine).

    Raises:
        ValueError: unknown backend name.
    """
    from openai import OpenAI

    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {', '.join(BACKENDS)}")
    spec = BACKENDS[backend]
    resolved = Backend(
        name=backend,
        base_url=base_url or spec["base_url"],
        api_key=spec["api_key"],
        model=model or spec["model"],
    )
    client = OpenAI(base_url=resolved.base_url, api_key=resolved.api_key)
    return client, resolved


def is_up(backend: str = "ollama", *, base_url: str | None = None, timeout: float = 2.0) -> bool:
    """Cheap liveness check so a notebook can skip a section instead of hanging."""
    import urllib.error
    import urllib.request

    url = (base_url or BACKENDS.get(backend, {}).get("base_url", "")).rstrip("/") + "/models"
    if not url.startswith("http"):
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def require_backend(backend: str = "ollama", *, base_url: str | None = None) -> None:
    """Raise a helpful error if the server is not running yet."""
    if is_up(backend, base_url=base_url):
        return
    spec = BACKENDS.get(backend, {})
    raise RuntimeError(
        f"No server answering at {base_url or spec.get('base_url')}.\n"
        f"Start it with:\n    {spec.get('start', 'see notebook 06')}\n"
        "Then re-run this cell."
    )


def complete(
    client,
    backend: Backend,
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 512,
    enable_thinking: bool | None = None,
    **kwargs,
):
    """One chat completion, with Qwen's thinking switch passed through.

    Servers expose the thinking toggle differently:

    * vLLM  -> ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``
    * Ollama -> ``extra_body={"think": False}``

    so we set both and let the server ignore the one it does not know.
    """
    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    if enable_thinking is not None:
        extra_body.setdefault("chat_template_kwargs", {"enable_thinking": enable_thinking})
        extra_body.setdefault("think", enable_thinking)

    request: dict = {
        "model": backend.model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        **kwargs,
    }
    if tools:
        request["tools"] = tools
    if extra_body:
        request["extra_body"] = extra_body
    return client.chat.completions.create(**request)

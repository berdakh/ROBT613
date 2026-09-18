"""Loading Qwen weights with `transformers`.

This wraps `AutoModelForCausalLM.from_pretrained` with the defaults that save
students the most pain:

* correct dtype per device (bf16 on modern GPUs, fp32 on CPU),
* optional 4-bit quantisation via bitsandbytes,
* a clear error when the download fails for the boring reasons (no network,
  no disk, gated repo, wrong mirror).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import DEFAULT_CHAT_MODEL
from .env import pick_device


@dataclass
class LoadedModel:
    """A model plus its tokenizer, plus the choices we made getting there."""

    model: object
    tokenizer: object
    repo_id: str
    device: str
    dtype: str
    quantized: bool

    def __repr__(self) -> str:  # keeps notebook output tidy
        q = " int4" if self.quantized else ""
        return f"<LoadedModel {self.repo_id} on {self.device} ({self.dtype}{q})>"

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def memory_footprint_gb(self) -> float:
        """How much memory the weights actually occupy, as reported by torch."""
        return self.model.get_memory_footprint() / 1024**3


def _resolve_dtype(device: str, dtype: str | None):
    import torch

    if dtype is not None:
        return getattr(torch, dtype)
    if device == "cuda":
        # bf16 needs Ampere (compute capability >= 8.0); older cards get fp16.
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    # CPU: fp32 is slow but bf16/fp16 matmuls on CPU are slower still on most
    # consumer chips, and fp16 on CPU is often numerically unstable.
    return torch.float32


def load_tokenizer(repo_id: str = DEFAULT_CHAT_MODEL, **kwargs):
    """Load just the tokenizer - a few megabytes, no GPU needed.

    Notebook 01 uses this to explore tokenisation without downloading weights.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo_id, **kwargs)


def load_model(
    repo_id: str = DEFAULT_CHAT_MODEL,
    *,
    device: str | None = None,
    dtype: str | None = None,
    quantize_4bit: bool = False,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> LoadedModel:
    """Download (if needed) and load a Qwen causal LM.

    Args:
        repo_id: Hugging Face repo, e.g. ``"Qwen/Qwen3-0.6B"``.
        device: ``"cuda"``/``"mps"``/``"cpu"``. Auto-detected when omitted.
        dtype: torch dtype name, e.g. ``"bfloat16"``. Auto-selected when omitted.
        quantize_4bit: Load in NF4 via bitsandbytes. CUDA only, cuts VRAM ~4x
            for a small quality cost. See notebook 05.
        attn_implementation: e.g. ``"flash_attention_2"`` if you have it built.
        trust_remote_code: Qwen3 text models are supported natively by recent
            `transformers`, so leave this False unless a model card says otherwise.

    Returns:
        A :class:`LoadedModel`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    torch_dtype = _resolve_dtype(device, dtype)

    load_kwargs: dict = {
        "dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    if quantize_4bit:
        if device != "cuda":
            raise RuntimeError(
                "quantize_4bit=True needs a CUDA GPU (bitsandbytes). "
                "On a CPU-only machine use GGUF + llama.cpp/Ollama instead - see notebook 05."
            )
        import torch
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        # bitsandbytes places the model itself; passing .to(device) later errors.
        load_kwargs["device_map"] = "auto"
        load_kwargs.pop("dtype", None)

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(repo_id, **load_kwargs)
    except Exception as exc:  # noqa: BLE001 - we re-raise with guidance
        raise RuntimeError(_download_hint(repo_id, exc)) from exc

    if not quantize_4bit:
        model = model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        repo_id=repo_id,
        device=device,
        dtype=str(torch_dtype).replace("torch.", ""),
        quantized=quantize_4bit,
    )


def _download_hint(repo_id: str, exc: Exception) -> str:
    """Turn an opaque HF error into something a student can act on."""
    text = str(exc).lower()
    hints = [f"Could not load '{repo_id}'.", f"Original error: {type(exc).__name__}: {exc}", ""]

    if "401" in text or "gated" in text or "authentication" in text:
        hints += [
            "This looks like a gated or private repo. Accept the licence on the",
            "model page, then authenticate:",
            "    huggingface-cli login",
        ]
    elif "connection" in text or "timed out" in text or "network" in text or "resolve" in text:
        hints += [
            "This looks like a network problem. Options:",
            "  1. Retry - HF downloads resume from where they stopped.",
            "  2. Use a mirror:   export HF_ENDPOINT=https://hf-mirror.com",
            "  3. Use ModelScope: modelscope download --model " + repo_id,
            "  4. Work offline from an already-cached copy: export HF_HUB_OFFLINE=1",
        ]
    elif "no space" in text or "disk" in text:
        current = os.environ.get("HF_HOME", "~/.cache/huggingface")
        hints += [
            f"Out of disk. The cache lives at {current}.",
            "Free space, or move the cache to a bigger drive:",
            "    export HF_HOME=/path/with/space/huggingface",
            "Then clean old downloads with:  huggingface-cli delete-cache",
        ]
    elif "out of memory" in text or "cuda" in text:
        hints += [
            "Out of GPU memory. Try, in order:",
            "  1. A smaller model (Qwen3-0.6B or Qwen3-1.7B).",
            "  2. quantize_4bit=True.",
            "  3. device='cpu' - slow, but it always works.",
        ]
    else:
        hints += [
            "Check the model id spelling against huggingface.co, and make sure",
            "`transformers` is recent enough:  pip install -U transformers",
        ]
    return "\n".join(hints)

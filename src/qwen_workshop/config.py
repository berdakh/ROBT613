"""Model catalogue and sampling defaults for the workshop.

The Qwen line-up moves fast. Treat this file as a *snapshot* that was accurate
when the workshop was written, and always confirm against the model card on
Hugging Face before you rely on a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """One entry in the workshop's model catalogue."""

    repo_id: str
    params: str
    kind: str  # "chat" | "embedding" | "reranker" | "vision" | "coder"
    approx_vram_gb_bf16: float
    approx_ram_gb_q4: float
    thinking: bool = False
    notes: str = ""

    @property
    def ollama_tag(self) -> str:
        """Best-guess Ollama tag. Verify with `ollama list` / ollama.com."""
        return self.repo_id.split("/")[-1].lower().replace("qwen3-", "qwen3:")


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
# Sizing rule of thumb used below:
#   bf16 weights  ~= 2 bytes/param      -> 0.6B ~ 1.2 GB
#   int4 weights  ~= 0.55 bytes/param   -> 0.6B ~ 0.4 GB
# then add ~20-30% headroom for the KV cache, activations and framework
# overhead. Notebook 02 derives this properly.

MODELS: dict[str, ModelInfo] = {
    "qwen3-0.6b": ModelInfo(
        repo_id="Qwen/Qwen3-0.6B",
        params="0.6B",
        kind="chat",
        approx_vram_gb_bf16=1.8,
        approx_ram_gb_q4=0.9,
        thinking=True,
        notes="The workshop default. Runs on a CPU-only laptop. Weak at facts, "
        "great for learning the mechanics.",
    ),
    "qwen3-1.7b": ModelInfo(
        repo_id="Qwen/Qwen3-1.7B",
        params="1.7B",
        kind="chat",
        approx_vram_gb_bf16=4.5,
        approx_ram_gb_q4=1.8,
        thinking=True,
        notes="Noticeably better instruction following than 0.6B, still laptop-friendly.",
    ),
    "qwen3-4b": ModelInfo(
        repo_id="Qwen/Qwen3-4B",
        params="4B",
        kind="chat",
        approx_vram_gb_bf16=10.0,
        approx_ram_gb_q4=3.5,
        thinking=True,
        notes="Sweet spot for a free Colab T4 or an 8 GB GPU at int4.",
    ),
    "qwen3-8b": ModelInfo(
        repo_id="Qwen/Qwen3-8B",
        params="8B",
        kind="chat",
        approx_vram_gb_bf16=19.0,
        approx_ram_gb_q4=6.0,
        thinking=True,
        notes="Good tool-calling and RAG quality. Needs int4 to fit in 8 GB VRAM.",
    ),
    "qwen3-14b": ModelInfo(
        repo_id="Qwen/Qwen3-14B",
        params="14B",
        kind="chat",
        approx_vram_gb_bf16=32.0,
        approx_ram_gb_q4=10.0,
        thinking=True,
        notes="Lab-machine territory (24 GB GPU at int4/int8).",
    ),
    "qwen3-30b-a3b": ModelInfo(
        repo_id="Qwen/Qwen3-30B-A3B",
        params="30B total / 3B active (MoE)",
        kind="chat",
        approx_vram_gb_bf16=64.0,
        approx_ram_gb_q4=20.0,
        thinking=True,
        notes="Mixture-of-Experts: memory of a 30B, speed closer to a 3B. "
        "Surprisingly usable on a 32 GB RAM machine with GGUF int4.",
    ),
    "qwen3-embedding-0.6b": ModelInfo(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        params="0.6B",
        kind="embedding",
        approx_vram_gb_bf16=1.8,
        approx_ram_gb_q4=0.9,
        notes="1024-dim embeddings, instruction-aware. Used in notebooks 08-09.",
    ),
    "qwen3-reranker-0.6b": ModelInfo(
        repo_id="Qwen/Qwen3-Reranker-0.6B",
        params="0.6B",
        kind="reranker",
        approx_vram_gb_bf16=1.8,
        approx_ram_gb_q4=0.9,
        notes="Cross-encoder that re-scores retrieved chunks. Optional in notebook 09.",
    ),
    "qwen2.5-coder-1.5b": ModelInfo(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        params="1.5B",
        kind="coder",
        approx_vram_gb_bf16=4.0,
        approx_ram_gb_q4=1.6,
        notes="Small code model, handy for the code-assistant capstone variant.",
    ),
}

DEFAULT_CHAT_MODEL = MODELS["qwen3-0.6b"].repo_id
DEFAULT_EMBEDDING_MODEL = MODELS["qwen3-embedding-0.6b"].repo_id


@dataclass
class SamplingParams:
    """Decoding settings, named the way `transformers` names them.

    ``as_dict()`` is what you splat into ``model.generate(**params.as_dict())``.
    """

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    do_sample: bool = True
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = {k: v for k, v in asdict(self).items() if k != "extra"}
        if not self.do_sample:
            # Greedy decoding: transformers warns if temperature/top_p are set.
            for key in ("temperature", "top_p", "top_k", "min_p"):
                data.pop(key, None)
        data.update(self.extra)
        return data

    def replace(self, **changes) -> SamplingParams:
        data = asdict(self)
        data.update(changes)
        extra = data.pop("extra", {})
        return SamplingParams(**data, extra=extra)


# Qwen3's own model card recommends different settings per mode. Using the
# thinking preset for non-thinking output (or vice versa) is the single most
# common cause of "why is my model rambling / repeating?".
THINKING_SAMPLING = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
NON_THINKING_SAMPLING = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0)

# Greedy decoding is NOT recommended for Qwen3 thinking mode (it tends to
# repeat), but it is useful when you want reproducible output in a demo.
GREEDY_SAMPLING = SamplingParams(do_sample=False, max_new_tokens=512)


def recommend_model(vram_gb: float | None, ram_gb: float, *, quantized: bool = True) -> ModelInfo:
    """Pick the largest catalogue chat model that plausibly fits this machine.

    Args:
        vram_gb: Dedicated GPU memory in GB, or ``None`` for CPU-only.
        ram_gb: System RAM in GB.
        quantized: Whether you intend to run int4/GGUF weights (recommended
            for anything above 2B on student hardware).

    Returns:
        The chosen :class:`ModelInfo`. Always conservative - it is much better
        to start small and scale up than to spend 20 minutes downloading a
        model that then OOMs.
    """
    chat_models = [m for m in MODELS.values() if m.kind == "chat"]
    # Largest first, so the first thing that fits wins.
    chat_models.sort(key=lambda m: m.approx_vram_gb_bf16, reverse=True)

    if vram_gb:
        budget = vram_gb * 0.85  # leave room for the desktop/display
        for model in chat_models:
            need = model.approx_ram_gb_q4 if quantized else model.approx_vram_gb_bf16
            if need <= budget:
                return model
    # CPU fallback. RAM is cheaper than VRAM, but CPU inference is roughly an
    # order of magnitude slower, so we are deliberately stricter here: a model
    # that technically fits but runs at 2 tokens/second ruins a live workshop.
    budget = max(ram_gb - 4.0, 1.0) * 0.35
    for model in chat_models:
        if model.approx_ram_gb_q4 <= budget:
            return model
    return MODELS["qwen3-0.6b"]

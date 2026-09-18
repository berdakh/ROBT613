"""Catalogue sanity and hardware recommendations."""

from __future__ import annotations

from qwen_workshop.config import (
    GREEDY_SAMPLING,
    MODELS,
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    SamplingParams,
    recommend_model,
)


def test_catalogue_entries_are_well_formed():
    for key, info in MODELS.items():
        assert info.repo_id.startswith("Qwen/"), key
        assert info.kind in {"chat", "embedding", "reranker", "vision", "coder"}, key
        assert info.approx_ram_gb_q4 < info.approx_vram_gb_bf16, key


def test_tiny_machines_get_the_tiny_model():
    assert recommend_model(None, 4.0).repo_id == "Qwen/Qwen3-0.6B"


def test_bigger_machines_get_bigger_models():
    small = recommend_model(8.0, 16.0)
    large = recommend_model(24.0, 64.0)
    assert large.approx_vram_gb_bf16 > small.approx_vram_gb_bf16


def test_recommendation_is_always_a_chat_model():
    for vram, ram in [(None, 2.0), (None, 64.0), (4.0, 8.0), (80.0, 256.0)]:
        assert recommend_model(vram, ram).kind == "chat"


def test_cpu_only_is_more_conservative_than_gpu():
    cpu = recommend_model(None, 32.0)
    gpu = recommend_model(32.0, 32.0)
    assert cpu.approx_vram_gb_bf16 <= gpu.approx_vram_gb_bf16


def test_greedy_sampling_omits_temperature():
    """transformers warns loudly if you pass temperature with do_sample=False."""
    params = GREEDY_SAMPLING.as_dict()
    assert params["do_sample"] is False
    assert "temperature" not in params
    assert "top_p" not in params


def test_sampling_presets_differ_per_mode():
    assert THINKING_SAMPLING.temperature != NON_THINKING_SAMPLING.temperature
    assert THINKING_SAMPLING.top_p != NON_THINKING_SAMPLING.top_p


def test_replace_returns_a_new_object():
    base = SamplingParams(temperature=0.7)
    changed = base.replace(temperature=0.1, max_new_tokens=32)
    assert base.temperature == 0.7  # unchanged
    assert changed.temperature == 0.1
    assert changed.as_dict()["max_new_tokens"] == 32


def test_extra_fields_pass_through():
    params = SamplingParams(extra={"presence_penalty": 1.5})
    assert params.as_dict()["presence_penalty"] == 1.5

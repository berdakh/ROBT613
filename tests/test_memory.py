"""Memory strategies: what each one keeps, and what it silently loses."""

from __future__ import annotations

import pytest

from qwen_workshop.memory import (
    ConversationBuffer,
    FactStore,
    Scratchpad,
    SummarisingMemory,
    WindowedMemory,
    estimate_tokens,
)


class CountingTokenizer:
    """One token per whitespace-separated word. Deterministic, unlike a real one."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())


def test_estimate_tokens_uses_tokenizer_when_given():
    assert estimate_tokens("one two three four", CountingTokenizer()) == 4
    assert estimate_tokens("x" * 40) == 10  # ~4 chars per token


def test_buffer_keeps_everything_and_puts_system_first():
    buffer = ConversationBuffer(system="S")
    buffer.add_user("a").add_assistant("b").add_user("c")
    rendered = buffer.render()
    assert rendered[0]["role"] == "system"
    assert [m["content"] for m in rendered[1:]] == ["a", "b", "c"]
    assert len(buffer) == 3


def test_window_drops_oldest_and_counts_the_loss():
    window = WindowedMemory(budget=40)
    for i in range(10):
        window.add_user(f"message number {i} with some padding text")
    rendered = window.render()
    assert window.dropped > 0
    assert len(rendered) + window.dropped == 10
    # What survives is the newest, which is the whole point and the whole risk.
    assert "9" in rendered[-1]["content"]


def test_keep_first_pins_the_task_statement():
    window = WindowedMemory(budget=40, keep_first=1)
    window.add_user("TASK: the thing the user actually asked for")
    for i in range(10):
        window.add_assistant(f"filler reply number {i} with padding")
    contents = [m["content"] for m in window.render()]
    assert any("TASK:" in c for c in contents)


def test_window_budget_is_respected():
    window = WindowedMemory(budget=60, system="sys")
    for i in range(30):
        window.add_user(f"turn {i} padded out a bit")
    assert window.token_count() <= 60 or len(window.render()) <= 2


def test_summarising_compacts_once_over_budget():
    memory = SummarisingMemory(budget=50, keep_recent=2,
                               summarise=lambda msgs: f"[{len(msgs)} turns summarised]")
    for i in range(8):
        memory.add_user(f"question {i} with a reasonable amount of padding")
        memory.compact()
    assert memory.compactions >= 1
    assert "summarised" in memory.summary
    assert any("Earlier conversation" in str(m["content"]) for m in memory.render())


def test_summarising_does_nothing_without_a_summariser():
    memory = SummarisingMemory(budget=1, keep_recent=1)
    memory.add_user("x" * 400)
    assert memory.compact() is False


def test_summarising_will_not_eat_the_recent_turns():
    """Compacting below keep_recent would delete what the next reply needs."""
    memory = SummarisingMemory(budget=1, keep_recent=4, summarise=lambda m: "s")
    memory.add_user("a").add_assistant("b")
    assert memory.compact() is False
    assert len(memory) == 2


def test_fact_store_round_trips_and_normalises_keys(tmp_path):
    store = FactStore()
    store.remember("Home City", "Almaty")
    assert store.recall("home city") == "Almaty"
    assert store.recall("HOME CITY") == "Almaty"
    path = tmp_path / "facts.json"
    store.save(path)
    assert FactStore.load(path).facts == store.facts


def test_fact_store_load_missing_file_is_empty(tmp_path):
    assert len(FactStore.load(tmp_path / "nope.json")) == 0


def test_fact_store_update_does_not_duplicate():
    store = FactStore()
    store.remember("city", "Almaty")
    assert "updated" in store.remember("city", "Astana")
    assert len(store) == 1
    assert store.recall("city") == "Astana"


def test_fact_store_is_bounded():
    """An unbounded memory is a context-window bug with a delay fuse."""
    store = FactStore(max_facts=2)
    store.remember("a", "1")
    store.remember("b", "2")
    with pytest.raises(ValueError, match="full"):
        store.remember("c", "3")
    # Updating an existing key must still work when full.
    assert store.remember("a", "9")


def test_fact_store_rejects_empty_key():
    with pytest.raises(ValueError):
        FactStore().remember("   ", "value")


def test_fact_store_search_covers_keys_and_values():
    store = FactStore()
    store.remember("diet", "vegetarian")
    store.remember("city", "Almaty")
    assert set(store.search("veg")) == {"diet"}
    assert set(store.search("cit")) == {"city"}


def test_empty_fact_store_has_no_system_message():
    assert FactStore().as_system_message() is None


def test_scratchpad_renders_numbered_notes():
    pad = Scratchpad().note("no eggs").note("calculate failed")
    assert "1. no eggs" in pad.render()
    assert "2. calculate failed" in pad.render()
    assert len(pad.clear()) == 0
    assert "nothing yet" in pad.render()


def test_summary_folds_instead_of_growing():
    """Concatenating summaries makes compaction cost more context than it saves."""
    memory = SummarisingMemory(
        budget=40, keep_recent=2,
        # A folding summariser: fixed length regardless of how much it is given.
        summarise=lambda msgs: "FOLDED",
    )
    for i in range(20):
        memory.add_user(f"turn {i} with a decent amount of padding text here")
        memory.compact()
    assert memory.compactions > 3
    assert memory.summary == "FOLDED"  # not "FOLDED\nFOLDED\nFOLDED..."


def test_previous_summary_is_handed_to_the_summariser():
    seen = []
    memory = SummarisingMemory(budget=30, keep_recent=1,
                               summarise=lambda msgs: seen.append(msgs) or f"s{len(seen)}")
    for i in range(6):
        memory.add_user(f"turn {i} padded out so the budget is exceeded quickly")
        memory.compact()
    assert len(seen) >= 2
    # From the second compaction on, the prior summary must be in the input.
    assert any("Summary of the conversation so far" in str(m.get("content"))
               for m in seen[-1])

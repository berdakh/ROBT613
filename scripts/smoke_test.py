#!/usr/bin/env python3
"""End-to-end check: load a model, generate text, embed a sentence.

If this prints generated text, your setup works.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --skip-embedding
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def check_generation(model_id: str) -> bool:
    from qwen_workshop.chat import chat
    from qwen_workshop.config import SamplingParams
    from qwen_workshop.loading import load_model

    print(f"\n[1/3] Loading {model_id} ...")
    qwen = load_model(model_id)
    print(f"      {qwen}")
    print(f"      {qwen.n_params / 1e9:.2f}B parameters, "
          f"{qwen.memory_footprint_gb():.2f} GB in memory")

    print("\n[2/3] Generating ...")
    result = chat(
        qwen,
        "Reply with exactly one short sentence confirming you are running locally.",
        sampling=SamplingParams(temperature=0.3, max_new_tokens=60),
    )
    print(f"      model says: {result.text.strip()[:200]}")
    print(f"      {result.completion_tokens} tokens in {result.seconds:.1f}s "
          f"= {result.tokens_per_second:.1f} tok/s")
    return bool(result.text.strip())


def check_embedding() -> bool:
    import numpy as np

    from qwen_workshop.rag import Embedder

    print("\n[3/3] Embedding ...")
    embedder = Embedder("Qwen/Qwen3-Embedding-0.6B")
    vectors = embedder.encode_documents(
        ["The cat sat on the mat.", "A feline rested on the rug.", "Interest rates rose."]
    )
    similar = float(np.dot(vectors[0], vectors[1]))
    different = float(np.dot(vectors[0], vectors[2]))
    print(f"      dimensions: {embedder.dim}")
    print(f"      similar sentences  : {similar:.3f}")
    print(f"      unrelated sentences: {different:.3f}")
    if similar <= different:
        print("      WARNING: similar sentences did not score higher. Something is wrong.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--skip-embedding", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print(" ROBT613 workshop - smoke test")
    print("=" * 64)

    try:
        if not check_generation(args.model):
            print("\nFAILED: the model produced no output.")
            return 1
        if not args.skip_embedding and not check_embedding():
            print("\nFAILED: embeddings look wrong.")
            return 1
    except ImportError as exc:
        print(f"\nMissing dependency: {exc}")
        print("\n    pip install -r requirements.txt\n")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        print("\nSee docs/guides/troubleshooting.html")
        return 1

    print("\n" + "=" * 64)
    print(" ALL CHECKS PASSED - you are ready for day 1")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

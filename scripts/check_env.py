#!/usr/bin/env python3
"""Report this machine's hardware, packages and model cache.

Run this first, and whenever something breaks:

    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_workshop.config import recommend_model  # noqa: E402
from qwen_workshop.env import check_environment, describe_environment  # noqa: E402

CORE_PACKAGES = ["torch", "transformers", "huggingface-hub"]


def main() -> int:
    report = check_environment()
    print(describe_environment(report))

    print()
    print("=" * 64)
    print(" Recommendation")
    print("=" * 64)
    choice = recommend_model(report.vram_gb, report.ram_gb, quantized=True)
    print(f"Start with : {choice.repo_id}  ({choice.params})")
    print(f"Memory     : ~{choice.approx_ram_gb_q4:.1f} GB at int4")
    print(f"Why        : {choice.notes}")

    if report.device == "cpu":
        print("\nCPU-only: expect 8-20 tokens/second on the 0.6B model.")
        print("Everything in the workshop works, but use the smallest models,")
        print("or open the notebooks in Google Colab with a free T4 GPU.")

    missing_core = [p for p in CORE_PACKAGES if report.packages.get(p) is None]
    print()
    print("=" * 64)
    if missing_core:
        print(" NOT READY")
        print("=" * 64)
        print("Missing core packages:", ", ".join(missing_core))
        print("\n    pip install -r requirements.txt\n")
        return 1

    print(" READY")
    print("=" * 64)
    print("Next steps:")
    print("  1. python scripts/download_models.py --set core")
    print("  2. jupyter lab notebooks/01_llm_foundations.ipynb")
    if not report.binaries.get("ollama"):
        print("\nOllama is not installed. You will need it from day 2:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

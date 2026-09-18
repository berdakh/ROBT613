#!/usr/bin/env python3
"""Pre-download the models the workshop uses.

Do this on good wifi, ideally the night before.

    python scripts/download_models.py --list
    python scripts/download_models.py --set core
    python scripts/download_models.py --set full
    python scripts/download_models.py --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SETS: dict[str, list[tuple[str, str, str]]] = {
    # (repo_id, approximate size, what needs it)
    "tokenizer": [
        ("Qwen/Qwen3-0.6B", "tokenizer only, ~10 MB", "notebook 01"),
    ],
    "core": [
        ("Qwen/Qwen3-0.6B", "~1.5 GB", "notebooks 03, 04, 11"),
        ("Qwen/Qwen3-Embedding-0.6B", "~1.2 GB", "notebooks 08, 09"),
    ],
    "full": [
        ("Qwen/Qwen3-0.6B", "~1.5 GB", "notebooks 03, 04, 11"),
        ("Qwen/Qwen3-1.7B", "~3.4 GB", "optional, better quality"),
        ("Qwen/Qwen3-Embedding-0.6B", "~1.2 GB", "notebooks 08, 09"),
        ("Qwen/Qwen3-Reranker-0.6B", "~1.2 GB", "notebook 09, optional"),
    ],
}


def download(repo_id: str, tokenizer_only: bool = False) -> bool:
    from huggingface_hub import snapshot_download

    patterns = (
        ["*.json", "*.txt", "*.model"]
        if tokenizer_only
        else ["*.json", "*.txt", "*.model", "*.safetensors"]
    )
    print(f"\n-> {repo_id}")
    try:
        path = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
        print(f"   done: {path}")
        return True
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {type(exc).__name__}: {exc}")
        print("   Try:  export HF_ENDPOINT=https://hf-mirror.com")
        print("   Or:   modelscope download --model " + repo_id)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="set_name", choices=sorted(SETS), default="core")
    parser.add_argument("--model", help="download one specific repo id instead")
    parser.add_argument("--list", action="store_true", help="show the sets and exit")
    args = parser.parse_args()

    if args.list:
        for name, entries in SETS.items():
            print(f"\n{name}:")
            for repo_id, size, used_by in entries:
                print(f"  {repo_id:<32}{size:<24}{used_by}")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub is not installed.\n\n    pip install -r requirements.txt\n")
        return 1

    if args.model:
        return 0 if download(args.model) else 1

    entries = SETS[args.set_name]
    print(f"Downloading the '{args.set_name}' set ({len(entries)} model(s)):")
    for repo_id, size, used_by in entries:
        print(f"  {repo_id:<32}{size:<24}{used_by}")

    ok = all(download(repo_id, tokenizer_only=(args.set_name == "tokenizer"))
             for repo_id, _size, _used in entries)

    print()
    if ok:
        print("All downloads complete. You can now work offline with:")
        print("    export HF_HUB_OFFLINE=1")
        return 0
    print("Some downloads failed - see the messages above.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Downloads resume where they stopped - just run it again.")
        raise SystemExit(130) from None

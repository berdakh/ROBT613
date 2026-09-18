"""Environment and hardware inspection.

Notebook 02 leans on this heavily. It is written so that it degrades
gracefully: a machine with no torch, no GPU and no internet still gets a
useful report instead of a traceback.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field

# Packages we care about, in the order we want them reported.
WATCHED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "sentence-transformers",
    "huggingface-hub",
    "bitsandbytes",
    "peft",
    "datasets",
    "trl",
    "vllm",
    "llama-cpp-python",
    "openai",
    "gradio",
    "faiss-cpu",
    "numpy",
    "pydantic",
]


@dataclass
class EnvReport:
    python_version: str
    platform: str
    ram_gb: float
    cpu_count: int
    device: str
    gpu_name: str | None = None
    vram_gb: float | None = None
    torch_version: str | None = None
    packages: dict[str, str | None] = field(default_factory=dict)
    binaries: dict[str, str | None] = field(default_factory=dict)
    hf_home: str | None = None
    hf_cache_gb: float | None = None

    @property
    def missing_packages(self) -> list[str]:
        return [name for name, version in self.packages.items() if version is None]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _total_ram_gb() -> float:
    """System RAM in GB, without requiring psutil."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        pass
    try:  # Windows
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / 1024**3
    except Exception:
        return 0.0


def pick_device() -> str:
    """Return the best available torch device string: cuda / mps / cpu.

    ``mps`` is Apple Silicon. It works for inference but a few kernels still
    fall back to CPU, so expect it to be slower than an equivalent NVIDIA GPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _hf_cache_size_gb() -> tuple[str | None, float | None]:
    """Where Hugging Face caches models, and how much space it is using."""
    home = os.environ.get("HF_HOME")
    if home:
        cache = os.path.join(home, "hub")
    else:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if not os.path.isdir(cache):
        return cache, 0.0
    total = 0
    for root, _dirs, files in os.walk(cache):
        for name in files:
            path = os.path.join(root, name)
            try:
                # follow_symlinks=False: the blob is counted once, not twice
                # (the snapshot dir is full of symlinks into blobs/).
                total += os.stat(path, follow_symlinks=False).st_size
            except OSError:
                continue
    return cache, total / 1024**3


def check_environment() -> EnvReport:
    """Collect everything notebook 02 needs to tell a student what to run."""
    device = pick_device()
    gpu_name: str | None = None
    vram_gb: float | None = None
    torch_version: str | None = None

    try:
        import torch

        torch_version = torch.__version__
        if device == "cuda":
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_gb = props.total_memory / 1024**3
        elif device == "mps":
            gpu_name = "Apple Silicon (MPS) - unified memory"
            # MPS shares system RAM; treat ~60% of it as usable for weights.
            vram_gb = _total_ram_gb() * 0.6
    except ImportError:
        pass

    cache_dir, cache_gb = _hf_cache_size_gb()

    return EnvReport(
        python_version=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        ram_gb=_total_ram_gb(),
        cpu_count=os.cpu_count() or 1,
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        torch_version=torch_version,
        packages={name: _package_version(name) for name in WATCHED_PACKAGES},
        binaries={name: shutil.which(name) for name in ("ollama", "llama-cli", "git", "git-lfs")},
        hf_home=cache_dir,
        hf_cache_gb=cache_gb,
    )


def describe_environment(report: EnvReport | None = None) -> str:
    """Render an :class:`EnvReport` as a readable block of text."""
    report = report or check_environment()
    lines = [
        "=" * 64,
        " ROBT613 workshop - environment report",
        "=" * 64,
        f"Python        : {report.python_version}",
        f"Platform      : {report.platform}",
        f"CPU cores     : {report.cpu_count}",
        f"System RAM    : {report.ram_gb:.1f} GB",
        f"Torch device  : {report.device}" + (f" (torch {report.torch_version})" if report.torch_version else " (torch not installed)"),
    ]
    if report.gpu_name:
        lines.append(f"Accelerator   : {report.gpu_name}")
    if report.vram_gb:
        lines.append(f"Usable VRAM   : {report.vram_gb:.1f} GB")
    lines.append(f"HF cache      : {report.hf_home} ({report.hf_cache_gb:.1f} GB used)")

    lines.append("")
    lines.append("Packages:")
    for name, version in report.packages.items():
        mark = "OK  " if version else "--  "
        lines.append(f"  {mark}{name:<22}{version or 'not installed'}")

    lines.append("")
    lines.append("Command-line tools:")
    for name, path in report.binaries.items():
        mark = "OK  " if path else "--  "
        lines.append(f"  {mark}{name:<22}{path or 'not found on PATH'}")

    if report.missing_packages:
        lines.append("")
        lines.append("Missing packages are fine if you are not doing that section.")
        lines.append("Install the core set with:  pip install -r requirements.txt")
    return "\n".join(lines)

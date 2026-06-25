from __future__ import annotations

import platform
import sys

import torch


def print_environment_info() -> None:
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"HIP/ROCm build: {torch.version.hip}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")


def resolve_device(require_gpu: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_gpu:
        raise RuntimeError(
            "GPU training was requested, but torch.cuda.is_available() is False."
        )

    return torch.device("cpu")

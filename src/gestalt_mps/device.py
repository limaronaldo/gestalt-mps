"""Device selection and local runtime diagnostics."""

import os
import platform
import subprocess
from contextlib import nullcontext

import torch


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _auto_device(system: str, mps_available: bool, cuda_available: bool) -> str:
    if system == "Darwin" and mps_available:
        return "mps"
    if cuda_available:
        return "cuda"
    return "cpu"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = _auto_device(
            platform.system(), torch.backends.mps.is_available(), torch.cuda.is_available()
        )
    if requested not in ("mps", "cuda", "cpu"):
        raise ValueError(f"Unknown device {requested!r}. Choose auto, mps, cuda or cpu.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Check your macOS and PyTorch build or select --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your GPU, driver and PyTorch build or select --device cpu."
        )
    return torch.device(requested)


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        requested = "fp32" if device.type == "cpu" else "bf16"
    if requested not in _DTYPES:
        raise ValueError(f"Unknown dtype {requested!r}. Choose auto, bf16, fp16 or fp32.")
    if requested == "bf16":
        if device.type == "mps" and not torch.backends.mps.is_macos_or_newer(14, 0):
            raise RuntimeError("MPS BF16 requires macOS 14 or newer. Select --dtype fp16 or --dtype fp32.")
        if device.type == "cuda":
            selected_device = torch.cuda.device(device) if device.index is not None else nullcontext()
            with selected_device:
                supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            if not supports_bf16:
                raise RuntimeError("CUDA BF16 requires native GPU support. Select --dtype fp16 or --dtype fp32.")
    return _DTYPES[requested]


def _command_output(*command: str) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _physical_memory_bytes(system: str) -> int | None:
    if system == "Darwin":
        value = _command_output("/usr/sbin/sysctl", "-n", "hw.memsize")
        return int(value) if value and value.isdecimal() else None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _memory_pressure(system: str) -> str | None:
    if system != "Darwin":
        return None
    output = _command_output("/usr/bin/memory_pressure", "-Q")
    for line in (output or "").splitlines():
        if line.startswith("System-wide memory free percentage:"):
            return line
    return None


def device_status() -> dict[str, str | bool | int | None]:
    system = platform.system()
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    try:
        recommended_memory = torch.mps.recommended_max_memory() if mps_available else None
    except RuntimeError:
        recommended_memory = None
    return {
        "platform": f"{system} {platform.machine()}",
        "macos_version": platform.mac_ver()[0] or None,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "auto_device": _auto_device(system, mps_available, cuda_available),
        "physical_memory_bytes": _physical_memory_bytes(system),
        "mps_recommended_max_memory_bytes": recommended_memory,
        "memory_pressure": _memory_pressure(system),
    }

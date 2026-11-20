import subprocess

import pytest
import torch

from gestalt_mps import device


def test_device_auto_priority_and_explicit_error(monkeypatch):
    monkeypatch.setattr(device.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert device.resolve_device("auto") == torch.device("mps")
    assert device.resolve_device("cpu") == torch.device("cpu")

    monkeypatch.setattr(device.platform, "system", lambda: "Linux")
    assert device.resolve_device("auto") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert device.resolve_device("auto") == torch.device("cpu")
    for requested in ("mps", "cuda"):
        with pytest.raises(RuntimeError, match=rf"(?i){requested}.*unavailable"):
            device.resolve_device(requested)
    with pytest.raises(ValueError, match="tpu"):
        device.resolve_device("tpu")


def test_resolve_dtype_does_not_silently_change_mps_precision(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *_: True)
    mps = torch.device("mps")
    assert device.resolve_dtype("auto", mps) == torch.bfloat16
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *_: False)
    with pytest.raises(RuntimeError, match=r"MPS.*BF16.*--dtype"):
        device.resolve_dtype("auto", mps)
    assert device.resolve_dtype("fp16", mps) == torch.float16
    assert device.resolve_dtype("fp32", mps) == torch.float32


def test_cuda_bf16_requires_native_support(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *, including_emulation: False)
    with pytest.raises(RuntimeError, match=r"CUDA.*BF16.*--dtype"):
        device.resolve_dtype("auto", cuda)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *, including_emulation: not including_emulation)
    assert device.resolve_dtype("auto", cuda) == torch.bfloat16


def test_cpu_auto_and_explicit_dtype():
    cpu = torch.device("cpu")
    assert device.resolve_dtype("auto", cpu) == torch.float32
    assert device.resolve_dtype("bf16", cpu) == torch.bfloat16
    with pytest.raises(ValueError, match="fp64"):
        device.resolve_dtype("fp64", cpu)


def test_device_status_separates_host_and_mps_memory(monkeypatch):
    monkeypatch.setattr(device.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 6000)

    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, {
            "/usr/sbin/sysctl": "9000\n",
            "/usr/bin/memory_pressure": "The system has 9000.\nSystem-wide memory free percentage: 82%\n",
        }[command[0]])

    monkeypatch.setattr(device.subprocess, "run", run)
    status = device.device_status()
    assert status["mps_built"]
    assert status["mps_available"]
    assert status["auto_device"] == "mps"
    assert status["physical_memory_bytes"] == 9000
    assert status["mps_recommended_max_memory_bytes"] == 6000
    assert status["memory_pressure"] == "System-wide memory free percentage: 82%"

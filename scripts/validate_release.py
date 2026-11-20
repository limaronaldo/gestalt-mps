#!/usr/bin/env python
"""Release validation runner for the installed ``gestalt`` CLI.

Each CLI invocation runs in its own child process, one after another.  The
child calls the very function behind the console script (``gestalt_mps.cli:main``
by default), so the validator exercises the same code as ``gestalt ...``.
Inside the child, a sampler thread measures the process peak RSS and, once
PyTorch is loaded and MPS is available, ``torch.mps.driver_allocated_memory()``.
The parent keeps stdout and stderr separate (sanitized) and writes a JSON
receipt.

Phase hooks (optional, no pipeline change required today)
---------------------------------------------------------
The child installs a tiny module named ``gestalt_validation_phases``.  Code
under test may mark phase boundaries without a hard dependency::

    try:
        import gestalt_validation_phases as _phases
    except ImportError:
        _phases = None
    if _phases: _phases.mark("load")      # also "forward", "cfg", "swap", ...

``mark(name)`` starts phase ``name``: it synchronizes MPS (when torch MPS is
in use), takes one driver sample that closes the previous phase and switches the
phase label used by the periodic sampler.  Without marks the whole run is one
phase called ``start`` and the receipt says ``phase_hook.marks == 0``.

MPS metrics are never reported as measured without samples: if no sample was
obtained the receipt has ``status: "not_measured"`` and ``null`` values.

Usage::

    python scripts/validate_release.py --profile smoke --output-dir RUN
    python scripts/validate_release.py --profile default --device mps --dtype bf16 \\
        --output-dir RUN --keep-going
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

SCHEMA = "gestalt-validate-release/1"
DEFAULT_ENTRYPOINT = "gestalt_mps.cli:main"
PHASE_MODULE = "gestalt_validation_phases"
PHASE_PROTOCOL = (
    f"import {PHASE_MODULE}; {PHASE_MODULE}.mark('<phase>')  # guarded by ImportError"
)
FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"
PROFILES = ("smoke", "default")
IBQ_VOCAB = 16384

# Revisions pinned by the implementation plan (recorded, not verified here).
EXPECTED_REVISIONS = {
    "gestalt_code": "a385cfc26195480a6cd8f1ad4ef6436ce399c8e0",
    "gestalt_weights": "42a8b5d94f80352515603a5b4bab9d2742b6ed80",
    "seed_voken_code": "c1544efff959ba18d453089071307a7115698551",
    "ibq_weights": "ef2200997f97118050c5b17dde48e8b78690f866",
}
VERSION_PACKAGES = (
    "gestalt-mps",
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "pillow",
    "einops",
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY|_KEY$", re.I)
_HF_TOKEN = re.compile(r"hf_[A-Za-z0-9]{20,}")
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def sanitize_text(text: str) -> str:
    """Hide secrets (env values, HF tokens, bearer tokens) and the home dir."""
    secrets = sorted(
        {v for k, v in os.environ.items() if _SECRET_NAME.search(k) and len(v) >= 6},
        key=len,
        reverse=True,
    )
    for value in secrets:
        text = text.replace(value, "***")
    text = _HF_TOKEN.sub("hf_***", text)
    text = _BEARER.sub(r"\1***", text)
    home = str(Path.home())
    if len(home) > 1:
        text = text.replace(home, "~")
    return text


def peak_rss_bytes() -> int:
    """Peak RSS of this process (``ru_maxrss``; bytes on macOS, KiB elsewhere)."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def _write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# --------------------------------------------------------------------------
# host metrics (swap, memory pressure)
# --------------------------------------------------------------------------

_SWAP = re.compile(
    r"total\s*=\s*([\d.]+)([MG])\s+used\s*=\s*([\d.]+)([MG])\s+free\s*=\s*([\d.]+)([MG])"
)


def parse_swapusage(text: str) -> dict[str, int] | None:
    match = _SWAP.search(text or "")
    if not match:
        return None
    scale = {"M": 1024**2, "G": 1024**3}
    g = match.groups()
    total, used, free = (int(float(g[i]) * scale[g[i + 1]]) for i in (0, 2, 4))
    return {"total_bytes": total, "used_bytes": used, "free_bytes": free}


def parse_memory_free_percent(text: str) -> int | None:
    match = re.search(r"memory free percentage:\s*(\d+)%", text or "")
    return int(match.group(1)) if match else None


def host_snapshot() -> dict[str, Any]:
    """Best-effort swap and memory-pressure reading (macOS); ``None`` elsewhere."""
    if sys.platform != "darwin":
        return {"swap": None, "memory_free_percent": None}
    swap = _run_text(["sysctl", "-n", "vm.swapusage"])
    pressure = _run_text(["memory_pressure", "-Q"])
    return {
        "swap": parse_swapusage(swap) if swap else None,
        "memory_free_percent": parse_memory_free_percent(pressure) if pressure else None,
    }


# --------------------------------------------------------------------------
# environment / versions
# --------------------------------------------------------------------------


def entrypoint_info(used: str) -> dict[str, Any]:
    declared = None
    try:
        for ep in importlib.metadata.entry_points(group="console_scripts"):
            if ep.name == "gestalt":
                declared = ep.value
                break
    except Exception:  # metadata lookup must never break a run
        declared = None
    return {"used": used, "declared": declared, "matches_declared": declared == used}


def collect_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    rev = _run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _run_text(["git", "-C", str(root), "status", "--porcelain"])
    return {
        "head": rev.strip() if rev else None,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def collect_environment(entrypoint: str) -> dict[str, Any]:
    mem = _run_text(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "physical_memory_bytes": int(mem) if mem and mem.strip().isdigit() else None,
        "versions": collect_versions(),
        "entrypoint": entrypoint_info(entrypoint),
        "candidate_git": _git_state(),
        "expected_revisions": dict(EXPECTED_REVISIONS),
        "mps_fallback_disabled": True,
    }


# --------------------------------------------------------------------------
# child: sampling inside the process that runs the CLI
# --------------------------------------------------------------------------


def _load_callable(spec: str) -> Callable[..., Any]:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), attr)


class ChildRecorder:
    """Collects peak RSS, MPS driver samples and phase marks inside the child."""

    def __init__(self, metrics_file: Path, interval: float, probe: Callable[[], Any] | None):
        self.metrics_file = metrics_file
        self.interval = interval
        self.probe = probe
        self.probe_error: str | None = None
        self.started = time.monotonic()
        self.phase = "start"
        self.phases: list[dict[str, Any]] = []
        self.rss_samples = 0
        self.mps_samples = 0
        self.mps_max: int | None = None
        self.mps_by_phase: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_flush = 0.0
        self._synced = False

    # -- sampling ---------------------------------------------------------
    def _torch_mps(self) -> Any:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        try:
            return torch if torch.backends.mps.is_available() else None
        except Exception:
            return None

    def _read_driver(self, sync: bool) -> int | None:
        try:
            if self.probe is not None:
                value = self.probe()
            else:
                torch = self._torch_mps()
                if torch is None:
                    return None
                if sync:
                    torch.mps.synchronize()
                    self._synced = True
                value = torch.mps.driver_allocated_memory()
        except Exception as exc:  # record, never crash the run under test
            self.probe_error = f"{type(exc).__name__}: {exc}"
            return None
        return None if value is None else int(value)

    def sample(self, *, sync: bool = False, phase: str | None = None) -> None:
        with self._lock:
            self.rss_samples += 1
            value = self._read_driver(sync)
            if value is not None:
                self.mps_samples += 1
                self.mps_max = value if self.mps_max is None else max(self.mps_max, value)
                entry = self.mps_by_phase.setdefault(
                    phase or self.phase, {"samples": 0, "max_driver_allocated_bytes": value}
                )
                entry["samples"] += 1
                entry["max_driver_allocated_bytes"] = max(
                    entry["max_driver_allocated_bytes"], value
                )
            if time.monotonic() - self._last_flush > 1.0:
                self.flush()

    def mark(self, name: str) -> None:
        """Start phase ``name``; closes the previous phase with a synced sample."""
        with self._lock:
            self.sample(sync=True)
            self.phase = str(name)
            self.phases.append(
                {
                    "name": self.phase,
                    "t_s": round(time.monotonic() - self.started, 6),
                    "peak_rss_bytes": peak_rss_bytes(),
                }
            )
            self.flush()

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        self.sample()

        def loop() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=loop, name="validate-sampler", daemon=True)
        self._thread.start()

    def stop(self, exit_code: int) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample(sync=True)
        self.flush(exit_code=exit_code)

    # -- output -----------------------------------------------------------
    def _mps_reason(self) -> str:
        if self.probe_error:
            return f"driver probe failed: {self.probe_error}"
        if self.probe is not None:
            return "probe returned no value (MPS unavailable)"
        if "torch" not in sys.modules:
            return "torch was never imported in the child"
        if self._torch_mps() is None:
            return "torch.backends.mps is not available in the child"
        return "no successful driver sample"

    def snapshot(self, exit_code: int | None = None) -> dict[str, Any]:
        torch = sys.modules.get("torch")
        built = available = None
        if torch is not None:
            try:
                built = bool(torch.backends.mps.is_built())
                available = bool(torch.backends.mps.is_available())
            except Exception:
                pass
        if self.mps_samples:
            mps: dict[str, Any] = {
                "status": "measured",
                "samples": self.mps_samples,
                "max_driver_allocated_bytes": self.mps_max,
                "phases": {k: dict(v) for k, v in self.mps_by_phase.items()},
                "synchronized_at_marks": self._synced,
            }
        else:
            mps = {
                "status": "not_measured",
                "reason": self._mps_reason(),
                "samples": 0,
                "max_driver_allocated_bytes": None,
                "phases": {},
            }
        return {
            "exit_code": exit_code,
            "elapsed_s": round(time.monotonic() - self.started, 6),
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_source": "ru_maxrss",
            "rss_samples": self.rss_samples,
            "sample_interval_s": self.interval,
            "mps": mps,
            "torch_mps": {"torch_imported": torch is not None, "built": built, "available": available},
            "phases": list(self.phases),
            "phase_hook": {"protocol": PHASE_PROTOCOL, "marks": len(self.phases)},
        }

    def flush(self, exit_code: int | None = None) -> None:
        self._last_flush = time.monotonic()
        try:
            _write_json_atomic(self.metrics_file, self.snapshot(exit_code))
        except OSError:
            pass


def _exit_code_of(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    sys.stderr.write(f"{value}\n")
    return 1


def child_main(argv: Sequence[str]) -> int:
    if "--" in argv:
        split = list(argv).index("--")
        own, cli_argv = list(argv[:split]), list(argv[split + 1 :])
    else:
        own, cli_argv = list(argv), []
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--metrics-file", required=True, type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--mps-probe", default=None)
    ns = parser.parse_args(own)

    probe = None
    probe_error = None
    if ns.mps_probe:
        try:
            probe = _load_callable(ns.mps_probe)
        except Exception as exc:
            probe_error = f"{type(exc).__name__}: {exc}"
    recorder = ChildRecorder(ns.metrics_file, ns.sample_interval, probe)
    recorder.probe_error = probe_error
    hooks = types.ModuleType(PHASE_MODULE)
    hooks.mark = recorder.mark  # type: ignore[attr-defined]
    sys.modules[PHASE_MODULE] = hooks
    recorder.start()
    code = 1
    set_observer = None
    try:
        entry = _load_callable(ns.entrypoint)
        if ns.entrypoint == DEFAULT_ENTRYPOINT:
            from gestalt_mps.pipeline import set_phase_observer

            set_observer = set_phase_observer
            set_observer(recorder.mark)
        code = _exit_code_of(entry(cli_argv))
    except SystemExit as exc:
        code = _exit_code_of(exc.code)
    except BaseException:  # keep the traceback on stderr and a nonzero code
        traceback.print_exc()
        code = 1
    finally:
        if set_observer is not None:
            set_observer(None)
        recorder.stop(code)
        sys.stdout.flush()
        sys.stderr.flush()
    return code


# --------------------------------------------------------------------------
# parent: one invocation
# --------------------------------------------------------------------------


def _not_measured_metrics(reason: str) -> dict[str, Any]:
    return {
        "peak_rss_bytes": None,
        "peak_rss_source": None,
        "rss_samples": 0,
        "mps": {
            "status": "not_measured",
            "reason": reason,
            "samples": 0,
            "max_driver_allocated_bytes": None,
            "phases": {},
        },
        "phases": [],
        "phase_hook": {"protocol": PHASE_PROTOCOL, "marks": 0},
    }


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def run_invocation(
    name: str,
    argv: Sequence[str],
    *,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    output_dir: Path,
    timeout_s: float = 3600.0,
    sample_interval: float = 0.25,
    mps_probe: str | None = None,
    expected_exit: int = 0,
) -> dict[str, Any]:
    """Run one CLI invocation in a fresh child process and return its record."""
    output_dir = Path(output_dir)
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    metrics_file = logs / f"{name}.metrics.json"
    metrics_file.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--entrypoint",
        entrypoint,
        "--metrics-file",
        str(metrics_file),
        "--sample-interval",
        str(sample_interval),
    ]
    if mps_probe:
        cmd += ["--mps-probe", mps_probe]
    cmd += ["--", *argv]

    env = os.environ.copy()
    parent_fallback = env.pop(FALLBACK_ENV, None)  # fallback must stay off
    started_utc, t0 = utc_now(), time.monotonic()
    timed_out = False
    try:
        done = subprocess.run(
            cmd,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        exit_code: int | None = done.returncode
        out_raw, err_raw = done.stdout, done.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out, exit_code = True, None
        out_raw, err_raw = exc.stdout, exc.stderr
    duration = time.monotonic() - t0
    finished_utc = utc_now()

    stdout_text = sanitize_text(_decode(out_raw))
    stderr_text = sanitize_text(_decode(err_raw))
    stdout_path = logs / f"{name}.stdout.txt"
    stderr_path = logs / f"{name}.stderr.txt"
    stdout_path.write_text(stdout_text)
    stderr_path.write_text(stderr_text)

    try:
        metrics = json.loads(metrics_file.read_text())
        metrics.pop("exit_code", None)
    except (OSError, ValueError):
        metrics = _not_measured_metrics("child produced no metrics file")

    return {
        "name": name,
        "argv": [sanitize_text(a) for a in argv],
        "entrypoint": entrypoint,
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "duration_s": round(duration, 6),
        "exit_code": exit_code,
        "expected_exit": expected_exit,
        "timed_out": timed_out,
        "passed": (not timed_out) and exit_code == expected_exit,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_bytes": len(stdout_text.encode()),
        "stderr_bytes": len(stderr_text.encode()),
        "stdout_sha256": _sha256_bytes(stdout_text.encode()),
        "stderr_sha256": _sha256_bytes(stderr_text.encode()),
        "mps_fallback": {"parent_value": parent_fallback, "child_value": None},
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# fixtures, manifests, checks
# --------------------------------------------------------------------------

FIXTURE_RED = "fixture_red_square.png"
FIXTURE_BLUE = "fixture_blue_stripes.png"


def write_fixture_images(directory: Path) -> dict[str, Path]:
    """Write small, original, deterministic, non-square test images."""
    from PIL import Image, ImageDraw

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    red = Image.new("RGB", (640, 480), (245, 245, 245))
    draw = ImageDraw.Draw(red)
    draw.rectangle((200, 140, 440, 340), fill=(220, 20, 30))
    draw.rectangle((0, 400, 640, 480), fill=(60, 140, 60))

    blue = Image.new("RGB", (384, 576), (250, 250, 250))
    draw = ImageDraw.Draw(blue)
    for i, top in enumerate(range(48, 528, 96)):
        if i % 2 == 0:
            draw.rectangle((0, top, 384, top + 48), fill=(20, 60, 200))

    paths = {FIXTURE_RED: directory / FIXTURE_RED, FIXTURE_BLUE: directory / FIXTURE_BLUE}
    red.save(paths[FIXTURE_RED])
    blue.save(paths[FIXTURE_BLUE])
    return paths


def _tail(artifacts_dir: Path | None, device: str | None, dtype: str | None, *, model: bool = True):
    args: list[str] = []
    if artifacts_dir is not None:
        args += ["--dir", str(artifacts_dir)]
    if model:
        if device:
            args += ["--device", device]
        if dtype:
            args += ["--dtype", dtype]
    return args


def _png(path: Path, w: int, h: int) -> dict[str, Any]:
    return {"kind": "png_size", "path": str(path), "size": [w, h]}


def _ids(path: Path, n: int) -> dict[str, Any]:
    return {"kind": "npy_ids", "path": str(path), "length": n, "low": 0, "high": IBQ_VOCAB}


def build_manifest(
    profile: str,
    *,
    work_dir: Path,
    artifacts_dir: Path | None,
    device: str | None,
    dtype: str | None,
) -> dict[str, Any]:
    """Return the ordered steps and cross-step comparisons for a profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILES}")
    work = Path(work_dir)
    model_tail = _tail(artifacts_dir, device, dtype)
    steps: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    def step(name, argv, *, requires=(), checks=()):
        steps.append(
            {"name": name, "argv": list(argv), "requires": list(requires), "checks": list(checks)}
        )

    nonempty = {"kind": "stdout_nonempty"}
    step("doctor", ["doctor", *_tail(artifacts_dir, None, None, model=False)], checks=[nonempty])

    if profile == "smoke":
        png, npy = work / "smoke_gen.png", work / "smoke_gen.npy"
        step(
            "generate-smoke",
            ["generate", "a red circle on a white background", "-o", str(png),
             "--steps", "4", "--cfg-scale", "4", "--lat-h", "16", "--lat-w", "16",
             "--seed", "1234", "--save-tokens", str(npy), *model_tail],
            checks=[_png(png, 256, 256), _ids(npy, 256)],
        )  # fmt: skip
        step(
            "ask-smoke",
            ["ask", str(npy), "Describe this image.", "--lat-h", "16", "--lat-w", "16", *model_tail],
            requires=["generate-smoke"],
            checks=[nonempty],
        )
        step(
            "text-smoke",
            ["text", "Say hello in one short sentence.", *model_tail],
            checks=[nonempty],
        )
        return {"profile": profile, "steps": steps, "comparisons": comparisons, "fixtures": {}}

    fixtures = {n: work / "fixtures" / n for n in (FIXTURE_RED, FIXTURE_BLUE)}
    seed, prompt = "1234", "a red circle on a white background"

    def generate(label: str, *, low_memory: bool = False):
        png, npy = work / f"gen-{label}.png", work / f"gen-{label}.npy"
        argv = ["generate", prompt, "-o", str(png), "--steps", "64", "--cfg-scale", "4",
                "--lat-h", "32", "--lat-w", "32", "--seed", seed, "--save-tokens", str(npy)]  # fmt: skip
        if low_memory:
            argv.append("--low-memory")
        step(f"generate-{label}", argv + model_tail,
             checks=[_png(png, 512, 512), _ids(npy, 1024)])  # fmt: skip
        return png, npy

    def ask(label: str, source: Path, question: str, *, expect=None, low_memory=False, requires=()):
        argv = ["ask", str(source), question, "--lat-h", "32", "--lat-w", "32"]
        if low_memory:
            argv.append("--low-memory")
        checks: list[dict[str, Any]] = [nonempty]
        if expect:  # answer quality is advisory: it does not fail the invocation
            checks.append({"kind": "stdout_contains", "any_of": [expect], "required": False})
        step(f"ask-{label}", argv + model_tail, requires=requires, checks=checks)

    png_a, npy_a = generate("normal-a")
    png_b, npy_b = generate("normal-b")
    _, npy_lm = generate("low-memory", low_memory=True)
    ask("red-square", fixtures[FIXTURE_RED], "What color is the square?", expect="red")
    ask("blue-stripes", fixtures[FIXTURE_BLUE], "What color are the stripes?", expect="blue")
    ask("generated-tokens", npy_a, "Describe this image.", requires=["generate-normal-a"])
    ask("red-square-low-memory", fixtures[FIXTURE_RED], "What color is the square?",
        expect="red", low_memory=True)  # fmt: skip
    step("text-hello", ["text", "Say hello in one short sentence.", *model_tail], checks=[nonempty])
    step(
        "text-capital",
        ["text", "What is the capital of France? Answer in one sentence.", *model_tail],
        checks=[nonempty, {"kind": "stdout_contains", "any_of": ["Paris"], "required": False}],
    )

    comparisons += [
        {"name": "repeat-same-seed-tokens", "kind": "same_file_sha256", "a": str(npy_a),
         "b": str(npy_b), "requires": ["generate-normal-a", "generate-normal-b"], "required": True},
        {"name": "repeat-same-seed-png", "kind": "same_file_sha256", "a": str(png_a),
         "b": str(png_b), "requires": ["generate-normal-a", "generate-normal-b"], "required": True},
        {"name": "low-memory-matches-normal-tokens", "kind": "same_file_sha256", "a": str(npy_a),
         "b": str(npy_lm), "requires": ["generate-normal-a", "generate-low-memory"],
         "required": False},
        {"name": "low-memory-ask-same-answer", "kind": "same_stdout", "a": "ask-red-square",
         "b": "ask-red-square-low-memory",
         "requires": ["ask-red-square", "ask-red-square-low-memory"], "required": False},
    ]  # fmt: skip
    return {
        "profile": profile,
        "steps": steps,
        "comparisons": comparisons,
        "fixtures": {n: str(p) for n, p in fixtures.items()},
    }


def _check_png(check: dict[str, Any]) -> tuple[bool, str]:
    try:
        from PIL import Image
    except ImportError:
        return False, "Pillow is not installed"
    try:
        with Image.open(check["path"]) as img:
            img.load()
            mode, size = img.mode, list(img.size)
    except Exception as exc:
        return False, f"cannot read PNG: {type(exc).__name__}: {exc}"
    want = list(check["size"])
    return (mode == "RGB" and size == want), f"mode={mode} size={size} expected RGB {want}"


def _check_npy(check: dict[str, Any]) -> tuple[bool, str]:
    try:
        import numpy as np
    except ImportError:
        return False, "numpy is not installed"
    try:
        arr = np.load(check["path"], allow_pickle=False)
    except Exception as exc:
        return False, f"cannot read npy: {type(exc).__name__}: {exc}"
    if arr.dtype.kind not in "iu":
        return False, f"dtype {arr.dtype} is not integer"
    if arr.size != check["length"]:
        return False, f"length {arr.size} != {check['length']}"
    lo, hi = int(arr.min()), int(arr.max())
    ok = lo >= check["low"] and hi < check["high"]
    return ok, f"min={lo} max={hi} allowed=[{check['low']}, {check['high']})"


def evaluate_checks(checks: Sequence[dict[str, Any]], *, stdout_text: str) -> list[dict[str, Any]]:
    results = []
    for check in checks:
        kind = check["kind"]
        if kind == "stdout_nonempty":
            ok, detail = bool(stdout_text.strip()), f"{len(stdout_text.strip())} chars"
        elif kind == "stdout_contains":
            low = stdout_text.lower()
            ok = any(w.lower() in low for w in check["any_of"])
            detail = f"any of {check['any_of']}"
        elif kind == "png_size":
            ok, detail = _check_png(check)
        elif kind == "npy_ids":
            ok, detail = _check_npy(check)
        else:
            ok, detail = False, f"unknown check kind {kind!r}"
        results.append(
            {"kind": kind, "ok": ok, "required": check.get("required", True), "detail": detail}
        )
    return results


# --------------------------------------------------------------------------
# whole run
# --------------------------------------------------------------------------


def _skipped(step: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "name": step["name"],
        "argv": [sanitize_text(a) for a in step["argv"]],
        "status": "skipped",
        "passed": False,
        "exit_code": None,
        "skip_reason": reason,
        "checks": [],
    }


def _compare(comp: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {"name": comp["name"], "kind": comp["kind"], "required": comp["required"]}
    unmet = [n for n in comp["requires"] if records.get(n, {}).get("status") != "passed"]
    if unmet:
        return {**out, "ok": False, "status": "skipped", "detail": f"steps not passed: {unmet}"}
    if comp["kind"] == "same_file_sha256":
        try:
            a, b = _sha256_file(Path(comp["a"])), _sha256_file(Path(comp["b"]))
        except OSError as exc:
            return {**out, "ok": False, "status": "error", "detail": str(exc)}
        return {**out, "ok": a == b, "status": "compared", "detail": f"{a[:12]} vs {b[:12]}"}
    if comp["kind"] == "same_stdout":
        a = records[comp["a"]]["stdout_sha256"]
        b = records[comp["b"]]["stdout_sha256"]
        return {**out, "ok": a == b, "status": "compared", "detail": f"{a[:12]} vs {b[:12]}"}
    return {**out, "ok": False, "status": "error", "detail": "unknown comparison kind"}


def run_profile(
    profile: str,
    *,
    output_dir: Path,
    artifacts_dir: Path | None = None,
    device: str | None = None,
    dtype: str | None = None,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    mps_probe: str | None = None,
    sample_interval: float = 0.25,
    timeout_s: float = 3600.0,
    keep_going: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    work = output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        profile, work_dir=work, artifacts_dir=artifacts_dir, device=device, dtype=dtype
    )
    fixtures: dict[str, Any] = {}
    if manifest["fixtures"]:
        paths = write_fixture_images(work / "fixtures")
        try:
            import PIL

            pillow = PIL.__version__
        except Exception:
            pillow = None
        fixtures = {
            "pillow_version": pillow,
            "files": {n: {"bytes": p.stat().st_size, "sha256": _sha256_file(p)} for n, p in paths.items()},
        }

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "profile": profile,
        "complete": False,
        "started_utc": utc_now(),
        "options": {
            "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
            "device": device,
            "dtype": dtype,
            "sample_interval_s": sample_interval,
            "timeout_s": timeout_s,
            "keep_going": keep_going,
            "mps_probe": mps_probe,
        },
        "environment": collect_environment(entrypoint),
        "fixtures": fixtures,
        "host": {"before": host_snapshot()},
        "steps": [],
        "comparisons": [],
    }
    partial = output_dir / "receipt.partial.json"
    t0 = time.monotonic()
    records: dict[str, dict[str, Any]] = {}
    failed_earlier = False

    for step in manifest["steps"]:
        unmet = [n for n in step["requires"] if records.get(n, {}).get("status") != "passed"]
        if unmet:
            rec = _skipped(step, f"prerequisite steps not passed: {unmet}")
        elif failed_earlier and not keep_going:
            rec = _skipped(step, "an earlier step failed (use --keep-going to continue)")
        else:
            if log:
                log(f"[{step['name']}] gestalt {' '.join(step['argv'][:2])} ...")
            rec = run_invocation(
                step["name"],
                step["argv"],
                entrypoint=entrypoint,
                output_dir=output_dir,
                timeout_s=timeout_s,
                sample_interval=sample_interval,
                mps_probe=mps_probe,
            )
            stdout_text = Path(rec["stdout_path"]).read_text()
            rec["checks"] = evaluate_checks(step["checks"], stdout_text=stdout_text)
            checks_ok = all(c["ok"] for c in rec["checks"] if c["required"])
            rec["passed"] = bool(rec["passed"] and checks_ok)
            rec["status"] = "passed" if rec["passed"] else "failed"
            rec["host_after"] = host_snapshot()
            if not rec["passed"]:
                failed_earlier = True
        records[step["name"]] = rec
        receipt["steps"].append(rec)
        if log:
            log(f"[{step['name']}] {rec['status']}")
        _write_json_atomic(partial, receipt)

    receipt["comparisons"] = [_compare(c, records) for c in manifest["comparisons"]]
    receipt["host"]["after"] = host_snapshot()
    statuses = [s["status"] for s in receipt["steps"]]
    rss = [
        s["metrics"]["peak_rss_bytes"]
        for s in receipt["steps"]
        if s.get("metrics") and s["metrics"].get("peak_rss_bytes")
    ]
    receipt["summary"] = {
        "passed": statuses.count("passed"),
        "failed": statuses.count("failed"),
        "skipped": statuses.count("skipped"),
        "max_peak_rss_bytes": max(rss) if rss else None,
    }
    receipt["overall_passed"] = (
        statuses.count("passed") == len(statuses)
        and all(c["ok"] for c in receipt["comparisons"] if c["required"])
    )
    receipt["finished_utc"] = utc_now()
    receipt["duration_s"] = round(time.monotonic() - t0, 6)
    receipt["complete"] = True
    return receipt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate_release.py",
        description="Run the installed gestalt CLI sequentially and write a JSON receipt.",
    )
    p.add_argument("--profile", choices=PROFILES, default="smoke")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--dir", dest="artifacts_dir", type=Path, default=None,
                   help="artifact directory passed as --dir to every command")  # fmt: skip
    p.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=None)
    p.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default=None)
    p.add_argument("--entrypoint", default=DEFAULT_ENTRYPOINT,
                   help="module:function run in each child (default: the console script)")  # fmt: skip
    p.add_argument("--mps-probe", default=None, help=argparse.SUPPRESS)
    p.add_argument("--sample-interval", type=float, default=0.25)
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per invocation")
    p.add_argument("--keep-going", action="store_true", help="run later steps after a failure")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--child":
        return child_main(argv[1:])
    ns = build_parser().parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ns.output_dir or Path(f"validation-{ns.profile}-{stamp}")
    receipt_path = out / "receipt.json"
    if receipt_path.exists():
        print(f"error: {receipt_path} already exists; choose a new --output-dir", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    receipt = run_profile(
        ns.profile,
        output_dir=out,
        artifacts_dir=ns.artifacts_dir,
        device=ns.device,
        dtype=ns.dtype,
        entrypoint=ns.entrypoint,
        mps_probe=ns.mps_probe,
        sample_interval=ns.sample_interval,
        timeout_s=ns.timeout,
        keep_going=ns.keep_going,
        log=lambda m: print(m, file=sys.stderr),
    )
    _write_json_atomic(receipt_path, receipt)
    (out / "receipt.partial.json").unlink(missing_ok=True)
    print(receipt_path)
    return 0 if receipt["overall_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

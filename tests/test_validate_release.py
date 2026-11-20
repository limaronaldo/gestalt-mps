"""Tests for scripts/validate_release.py.

Nothing here loads weights or imports the real pipeline: the runner is pointed
at a tiny fake entrypoint written into ``tmp_path`` and executed in real child
processes, so the subprocess, sampling and receipt logic is exercised for real.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_release.py"

FAKE_CLI = textwrap.dedent(
    '''
    import os, sys, time
    from pathlib import Path

    _COUNTER = [0]


    def _mark(name):
        if os.environ.get("FAKE_NO_MARK"):
            return
        try:
            import gestalt_validation_phases as phases
        except ImportError:
            return
        phases.mark(name)


    def probe():
        mode = os.environ.get("FAKE_PROBE_MODE", "grow")
        if mode == "none":
            return None
        if mode == "zero":
            return 0
        _COUNTER[0] += 100
        return _COUNTER[0]


    def _opt(argv, flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default


    def _write_outputs(argv):
        import numpy as np
        from PIL import Image

        lat_h = int(_opt(argv, "--lat-h", "32"))
        lat_w = int(_opt(argv, "--lat-w", "32"))
        seed = int(_opt(argv, "--seed", "0"))
        size = (16 * lat_w, 16 * lat_h)
        if os.environ.get("FAKE_BAD_PNG"):
            size = (8, 8)
        Image.new("RGB", size, (seed % 256, 10, 20)).save(_opt(argv, "-o"))
        tokens_path = _opt(argv, "--save-tokens")
        if tokens_path:
            ids = (np.arange(lat_h * lat_w, dtype=np.int64) * 7 + seed) % 16384
            np.save(tokens_path, ids.astype(np.int32), allow_pickle=False)


    def main(argv=None):
        argv = list(argv or [])
        cmd = argv[0] if argv else ""
        _mark("load")
        time.sleep(0.15)
        _mark("forward")
        if cmd == "boom":
            sys.stderr.write("boom error\\n")
            return 3
        if cmd == "raise":
            raise RuntimeError("kaput")
        if cmd == "exit5":
            raise SystemExit(5)
        if cmd == "sleep":
            time.sleep(60)
        if cmd == "env":
            print(repr(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")))
            print("token=" + os.environ.get("HF_TOKEN", ""))
            print("home=" + str(Path.home()))
            return 0
        if cmd == "generate":
            if os.environ.get("FAKE_FAIL_GENERATE"):
                sys.stderr.write("generate failed\\n")
                return 1
            _write_outputs(argv)
            sys.stderr.write("saved\\n")
            return 0
        if cmd == "ask":
            if not Path(argv[1]).exists():
                sys.stderr.write("missing input\\n")
                return 2
            print("The picture shows " + Path(argv[1]).stem.replace("_", " ") + ".")
            return 0
        if cmd == "text":
            print("Hello there.")
            return 0
        if cmd == "doctor":
            print("MPS available: False")
            return 0
        print("answer:" + " ".join(argv))
        sys.stderr.write("status line\\n")
        return 0
    '''
)


@pytest.fixture(scope="module")
def vr():
    spec = importlib.util.spec_from_file_location("validate_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_release"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake(tmp_path, monkeypatch):
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "fake_cli.py").write_text(FAKE_CLI)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(pkg), existing])))
    monkeypatch.delenv("FAKE_PROBE_MODE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    return "fake_cli:main"


@pytest.fixture
def no_host(vr, monkeypatch):
    monkeypatch.setattr(vr, "host_snapshot", lambda: {"swap": None, "memory_free_percent": None})


def invoke(vr, tmp_path, argv, entrypoint, **kw):
    kw.setdefault("sample_interval", 0.02)
    kw.setdefault("timeout_s", 60)
    return vr.run_invocation(
        "step", argv, entrypoint=entrypoint, output_dir=tmp_path / "out", **kw
    )


# --- one invocation -------------------------------------------------------


def test_invocation_keeps_stdout_and_stderr_separate(vr, fake, tmp_path):
    rec = invoke(vr, tmp_path, ["hello", "world"], fake)
    assert rec["exit_code"] == 0 and rec["passed"] is True
    out = Path(rec["stdout_path"]).read_text()
    err = Path(rec["stderr_path"]).read_text()
    assert out == "answer:hello world\n"
    assert err == "status line\n"
    assert rec["stdout_sha256"] == hashlib.sha256(out.encode()).hexdigest()
    assert rec["stderr_bytes"] == len(err.encode())
    assert rec["argv"] == ["hello", "world"]


@pytest.mark.parametrize(
    "argv, code",
    [(["boom"], 3), (["exit5"], 5), (["raise"], 1)],
)
def test_nonzero_exit_is_recorded_and_fails_step(vr, fake, tmp_path, argv, code):
    rec = invoke(vr, tmp_path, argv, fake)
    assert rec["exit_code"] == code
    assert rec["passed"] is False and rec["timed_out"] is False
    if argv == ["raise"]:
        assert "RuntimeError: kaput" in Path(rec["stderr_path"]).read_text()


def test_expected_nonzero_exit_can_pass(vr, fake, tmp_path):
    rec = invoke(vr, tmp_path, ["boom"], fake, expected_exit=3)
    assert rec["passed"] is True


def test_timeout_kills_child_and_keeps_partial_record(vr, fake, tmp_path):
    rec = invoke(vr, tmp_path, ["sleep"], fake, timeout_s=2)
    assert rec["timed_out"] is True and rec["exit_code"] is None
    assert rec["passed"] is False
    assert Path(rec["stdout_path"]).exists() and Path(rec["stderr_path"]).exists()


def test_peak_rss_measured_inside_child(vr, fake, tmp_path):
    m = invoke(vr, tmp_path, ["ok"], fake)["metrics"]
    assert isinstance(m["peak_rss_bytes"], int) and m["peak_rss_bytes"] > 1_000_000
    assert m["rss_samples"] >= 1
    assert m["peak_rss_source"] == "ru_maxrss"


def test_mps_not_reported_as_measured_without_samples(vr, fake, tmp_path):
    """No torch in the fake child and no probe: nothing may be reported as 0."""
    mps = invoke(vr, tmp_path, ["ok"], fake)["metrics"]["mps"]
    assert mps["status"] == "not_measured"
    assert mps["samples"] == 0
    assert mps["max_driver_allocated_bytes"] is None
    assert mps["reason"]


def test_mps_probe_returning_none_is_not_measured(vr, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE_MODE", "none")
    mps = invoke(vr, tmp_path, ["ok"], fake, mps_probe="fake_cli:probe")["metrics"]["mps"]
    assert mps["status"] == "not_measured"
    assert mps["max_driver_allocated_bytes"] is None
    assert mps["samples"] == 0


def test_mps_sampled_zero_is_measured_zero(vr, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PROBE_MODE", "zero")
    mps = invoke(vr, tmp_path, ["ok"], fake, mps_probe="fake_cli:probe")["metrics"]["mps"]
    assert mps["status"] == "measured"
    assert mps["samples"] >= 2
    assert mps["max_driver_allocated_bytes"] == 0


def test_mps_driver_sampling_and_phase_marks(vr, fake, tmp_path):
    m = invoke(vr, tmp_path, ["ok"], fake, mps_probe="fake_cli:probe")["metrics"]
    mps = m["mps"]
    assert mps["status"] == "measured"
    assert mps["samples"] >= 2
    assert mps["max_driver_allocated_bytes"] >= 200
    assert [p["name"] for p in m["phases"]] == ["load", "forward"]
    assert m["phase_hook"]["marks"] == 2
    by_phase = mps["phases"]
    assert set(by_phase) >= {"load", "forward"}
    assert mps["max_driver_allocated_bytes"] == max(
        p["max_driver_allocated_bytes"] for p in by_phase.values()
    )
    for p in m["phases"]:
        assert p["peak_rss_bytes"] > 0


def test_run_without_phase_marks_reports_zero_marks(vr, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_NO_MARK", "1")
    m = invoke(vr, tmp_path, ["ok"], fake, mps_probe="fake_cli:probe")["metrics"]
    assert m["phases"] == [] and m["phase_hook"]["marks"] == 0
    assert m["phase_hook"]["protocol"].startswith("import gestalt_validation_phases")
    assert set(m["mps"]["phases"]) == {"start"}  # everything lands in one phase


def test_mps_fallback_env_is_removed_and_logged(vr, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    rec = invoke(vr, tmp_path, ["env"], fake)
    assert Path(rec["stdout_path"]).read_text().splitlines()[0] == "None"
    assert rec["mps_fallback"] == {"parent_value": "1", "child_value": None}


def test_secrets_and_home_are_sanitized(vr, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "A1b2C3d4E5f6G7h8I9j0K1l2")
    rec = invoke(vr, tmp_path, ["env"], fake)
    text = Path(rec["stdout_path"]).read_text()
    assert "hf_A1b2" not in text and "A1b2C3d4E5f6G7h8I9j0K1l2" not in text
    assert "token=***" in text
    assert str(Path.home()) not in text and "home=~" in text


def test_sanitize_text_patterns(vr, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "supersecretvalue")
    out = vr.sanitize_text(
        "Authorization: Bearer abc.DEF-123 key supersecretvalue "
        "hf_abcdefghijklmnopqrstuvwxyz0123 ok"
    )
    assert "abc.DEF-123" not in out
    assert "supersecretvalue" not in out
    assert "hf_abcdefghijkl" not in out
    assert out.endswith(" ok")


# --- host metrics ---------------------------------------------------------


def test_parse_swapusage_and_memory_pressure(vr):
    swap = vr.parse_swapusage("total = 2048.00M  used = 1024.50M  free = 1023.50M  (encrypted)")
    assert swap == {
        "total_bytes": 2048 * 1024**2,
        "used_bytes": int(1024.5 * 1024**2),
        "free_bytes": int(1023.5 * 1024**2),
    }
    assert vr.parse_swapusage("garbage") is None
    assert vr.parse_memory_free_percent("System-wide memory free percentage: 42%\n") == 42
    assert vr.parse_memory_free_percent("nope") is None


# --- manifests ------------------------------------------------------------


def _by_cmd(steps, cmd):
    return [s for s in steps if s["argv"][0] == cmd]


def test_smoke_manifest_is_small_and_ordered(vr, tmp_path):
    m = vr.build_manifest("smoke", work_dir=tmp_path, artifacts_dir=None, device=None, dtype=None)
    steps = m["steps"]
    assert [s["argv"][0] for s in steps] == ["doctor", "generate", "ask", "text"]
    gen = steps[1]["argv"]
    assert gen[gen.index("--lat-h") + 1] == "16" and gen[gen.index("--lat-w") + 1] == "16"
    assert int(gen[gen.index("--steps") + 1]) <= 8
    assert "--save-tokens" in gen and "--low-memory" not in gen
    assert steps[2]["requires"] == [steps[1]["name"]]
    names = [s["name"] for s in steps]
    assert len(names) == len(set(names))


def test_default_manifest_uses_upstream_defaults_and_low_memory(vr, tmp_path):
    m = vr.build_manifest("default", work_dir=tmp_path, artifacts_dir=None, device=None, dtype=None)
    steps = m["steps"]
    gens = _by_cmd(steps, "generate")
    assert len(gens) == 3
    for g in gens:
        a = g["argv"]
        assert a[a.index("--lat-h") + 1] == "32" and a[a.index("--lat-w") + 1] == "32"
        assert a[a.index("--steps") + 1] == "64" and a[a.index("--cfg-scale") + 1] == "4"
    assert sum("--low-memory" in g["argv"] for g in gens) == 1
    seeds = {g["argv"][g["argv"].index("--seed") + 1] for g in gens}
    assert len(seeds) == 1  # same seed for repetition and low-memory comparison
    asks = _by_cmd(steps, "ask")
    assert len(asks) >= 3
    assert any(c["kind"] == "same_file_sha256" for c in m["comparisons"])
    assert len(_by_cmd(steps, "text")) >= 2


def test_manifest_forwards_dir_device_dtype(vr, tmp_path):
    m = vr.build_manifest(
        "smoke", work_dir=tmp_path, artifacts_dir=Path("/models"), device="mps", dtype="bf16"
    )
    for s in m["steps"]:
        a = s["argv"]
        assert a[a.index("--dir") + 1] == "/models"
        if a[0] == "doctor":
            assert "--device" not in a and "--dtype" not in a
        else:
            assert a[a.index("--device") + 1] == "mps"
            assert a[a.index("--dtype") + 1] == "bf16"


def test_unknown_profile_rejected(vr, tmp_path):
    with pytest.raises(ValueError):
        vr.build_manifest("nightly", work_dir=tmp_path, artifacts_dir=None, device=None, dtype=None)


def test_fixture_images_are_deterministic_and_not_square(vr, tmp_path):
    pytest.importorskip("PIL")
    a = vr.write_fixture_images(tmp_path / "a")
    b = vr.write_fixture_images(tmp_path / "b")
    assert len(a) >= 2
    for name in a:
        assert a[name].read_bytes() == b[name].read_bytes()
    from PIL import Image

    sizes = {Image.open(p).size for p in a.values()}
    assert any(w != h for w, h in sizes)
    assert len({p.read_bytes() for p in a.values()}) == len(a)


# --- checks ---------------------------------------------------------------


def test_checks_png_and_npy(vr, tmp_path):
    np = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")
    png = tmp_path / "x.png"
    pil.new("RGB", (64, 32)).save(png)
    good = tmp_path / "ids.npy"
    np.save(good, np.arange(8, dtype=np.int32), allow_pickle=False)
    bad = tmp_path / "bad.npy"
    np.save(bad, np.array([0, 16384], dtype=np.int32), allow_pickle=False)
    checks = [
        {"kind": "png_size", "path": str(png), "size": [64, 32]},
        {"kind": "png_size", "path": str(png), "size": [32, 32]},
        {"kind": "npy_ids", "path": str(good), "length": 8, "low": 0, "high": 16384},
        {"kind": "npy_ids", "path": str(bad), "length": 2, "low": 0, "high": 16384},
        {"kind": "npy_ids", "path": str(tmp_path / "missing.npy"), "length": 1, "low": 0, "high": 2},
        {"kind": "stdout_nonempty"},
        {"kind": "stdout_contains", "any_of": ["RED"], "required": False},
    ]
    res = vr.evaluate_checks(checks, stdout_text="a red square\n")
    assert [r["ok"] for r in res] == [True, False, True, False, False, True, True]
    assert res[-1]["required"] is False


def test_checks_stdout_failures(vr):
    res = vr.evaluate_checks(
        [{"kind": "stdout_nonempty"}, {"kind": "stdout_contains", "any_of": ["blue"], "required": False}],
        stdout_text="  \n",
    )
    assert [r["ok"] for r in res] == [False, False]


# --- whole runs -----------------------------------------------------------


def run_main(vr, tmp_path, fake, *extra):
    out = tmp_path / "run"
    rc = vr.main(
        ["--output-dir", str(out), "--entrypoint", fake, "--sample-interval", "0.02", *extra]
    )
    return rc, out


def test_smoke_run_writes_json_receipt(vr, fake, no_host, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    rc, out = run_main(vr, tmp_path, fake, "--profile", "smoke", "--device", "cpu")
    assert rc == 0
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["profile"] == "smoke" and receipt["overall_passed"] is True
    assert [s["argv"][0] for s in receipt["steps"]] == ["doctor", "generate", "ask", "text"]
    assert all(s["passed"] and s["status"] == "passed" for s in receipt["steps"])
    env = receipt["environment"]
    assert env["python"].count(".") >= 2 and "platform" in env
    assert "versions" in env and "torch" in env["versions"]
    assert env["entrypoint"]["used"] == fake
    assert receipt["host"]["before"] == {"swap": None, "memory_free_percent": None}
    # sequential: each invocation finished before the next one started
    for prev, nxt in zip(receipt["steps"], receipt["steps"][1:]):
        assert prev["finished_utc"] <= nxt["started_utc"]
    assert receipt["steps"][1]["checks"] and all(c["ok"] for c in receipt["steps"][1]["checks"])
    assert receipt["steps"][1]["metrics"]["mps"]["status"] == "not_measured"
    assert receipt["schema"] == "gestalt-validate-release/1"


def test_default_run_compares_repeat_and_low_memory(vr, fake, no_host, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    rc, out = run_main(vr, tmp_path, fake, "--profile", "default")
    receipt = json.loads((out / "receipt.json").read_text())
    assert rc == 0, receipt
    comps = {c["name"]: c for c in receipt["comparisons"]}
    assert comps and all(c["ok"] for c in comps.values())
    asks = [s for s in receipt["steps"] if s["argv"][0] == "ask"]
    red = [s for s in asks if "red" in " ".join(s["argv"])]
    assert red and all(c["ok"] for s in red for c in s["checks"])


def test_failed_prerequisite_skips_dependents_and_fails_run(vr, fake, no_host, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setenv("FAKE_FAIL_GENERATE", "1")
    rc, out = run_main(vr, tmp_path, fake, "--profile", "smoke", "--keep-going")
    receipt = json.loads((out / "receipt.json").read_text())
    assert rc == 1 and receipt["overall_passed"] is False
    status = {s["argv"][0]: s["status"] for s in receipt["steps"]}
    assert status == {"doctor": "passed", "generate": "failed", "ask": "skipped", "text": "passed"}
    skipped = [s for s in receipt["steps"] if s["status"] == "skipped"][0]
    assert skipped["exit_code"] is None and skipped["skip_reason"]


def test_stops_after_first_failure_without_keep_going(vr, fake, no_host, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setenv("FAKE_FAIL_GENERATE", "1")
    rc, out = run_main(vr, tmp_path, fake, "--profile", "smoke")
    receipt = json.loads((out / "receipt.json").read_text())
    assert rc == 1
    status = [s["status"] for s in receipt["steps"]]
    assert status == ["passed", "failed", "skipped", "skipped"]


def test_check_failure_fails_step_even_with_exit_zero(vr, fake, no_host, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setenv("FAKE_BAD_PNG", "1")
    rc, out = run_main(vr, tmp_path, fake, "--profile", "smoke", "--keep-going")
    receipt = json.loads((out / "receipt.json").read_text())
    gen = receipt["steps"][1]
    assert gen["exit_code"] == 0 and gen["status"] == "failed"
    assert any(not c["ok"] for c in gen["checks"])
    assert rc == 1


def test_main_refuses_to_overwrite_existing_receipt(vr, fake, no_host, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "receipt.json").write_text("{}")
    rc = vr.main(["--output-dir", str(out), "--entrypoint", fake, "--profile", "smoke"])
    assert rc == 2
    assert (out / "receipt.json").read_text() == "{}"


def test_invalid_profile_is_argparse_error(vr, tmp_path):
    with pytest.raises(SystemExit) as exc:
        vr.main(["--profile", "nightly", "--output-dir", str(tmp_path / "x")])
    assert exc.value.code == 2


def test_default_entrypoint_matches_console_script(vr):
    assert vr.DEFAULT_ENTRYPOINT == "gestalt_mps.cli:main"
    info = vr.entrypoint_info(vr.DEFAULT_ENTRYPOINT)
    assert info["used"] == "gestalt_mps.cli:main"
    assert "declared" in info

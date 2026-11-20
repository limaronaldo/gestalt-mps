import math
import subprocess
import sys
from pathlib import Path

import pytest

from gestalt_mps import cli
from gestalt_mps.artifacts import ArtifactPaths


@pytest.fixture
def calls(monkeypatch, tmp_path):
    rec = {"events": []}
    arts = ArtifactPaths(tmp_path / "g", tmp_path / "i.ckpt")

    def resolve_device(req):
        rec["events"].append("device")
        rec["device_req"] = req
        return "DEV"

    def resolve_dtype(req, dev):
        rec["events"].append("dtype")
        rec["dtype_req"] = (req, dev)
        return "DT"

    def resolve_artifacts(directory, *, require_ibq):
        rec["events"].append("artifacts")
        rec["artifacts"] = (directory, require_ibq)
        return arts

    def p_generate(prompt, output, **kw):
        rec["events"].append("generate")
        rec["generate"] = (prompt, output, kw)

    def p_ask(path, question, **kw):
        rec["events"].append("ask")
        rec["ask"] = (path, question, kw)
        return "a cat"

    def p_text(prompt, **kw):
        rec["events"].append("text")
        rec["text"] = (prompt, kw)
        return "hello"

    monkeypatch.setattr(cli.device, "resolve_device", resolve_device)
    monkeypatch.setattr(cli.device, "resolve_dtype", resolve_dtype)
    monkeypatch.setattr(cli.artifacts, "resolve_artifacts", resolve_artifacts)
    monkeypatch.setattr(cli.pipeline, "generate", p_generate)
    monkeypatch.setattr(cli.pipeline, "ask", p_ask)
    monkeypatch.setattr(cli.pipeline, "text", p_text)
    rec["arts"] = arts
    return rec


def test_generate_defaults_and_explicit_kwargs(calls, tmp_path, capsys):
    out = tmp_path / "o.png"
    assert cli.main(["generate", "a dog", "-o", str(out)]) == 0
    prompt, output, kw = calls["generate"]
    assert prompt == "a dog" and output == out
    assert kw == dict(
        artifacts=calls["arts"], device="DEV", dtype="DT", steps=64,
        cfg_scale=4.0, lat_h=32, lat_w=32, seed=None, system_prompt="",
        save_tokens=None, low_memory=False,
    )
    assert calls["artifacts"] == (None, True)
    assert calls["device_req"] == "auto" and calls["dtype_req"] == ("auto", "DEV")
    assert capsys.readouterr().out == ""


def test_generate_all_flags(calls, tmp_path):
    out = tmp_path / "o.png"
    toks = tmp_path / "t.NPY"
    rc = cli.main([
        "generate", "x", "-o", str(out), "--steps", "8", "--cfg-scale", "2.5",
        "--lat-h", "4", "--lat-w", "6", "--seed", "7", "--system-prompt", "S",
        "--save-tokens", str(toks), "--low-memory", "--dir", str(tmp_path),
        "--device", "cpu", "--dtype", "fp32",
    ])
    assert rc == 0
    kw = calls["generate"][2]
    assert (kw["steps"], kw["cfg_scale"], kw["lat_h"], kw["lat_w"], kw["seed"]) == (8, 2.5, 4, 6, 7)
    assert kw["system_prompt"] == "S" and kw["save_tokens"] == toks and kw["low_memory"] is True
    assert calls["artifacts"] == (tmp_path, True)
    assert calls["device_req"] == "cpu" and calls["dtype_req"][0] == "fp32"


@pytest.mark.parametrize("bad", [
    ["--steps", "0"], ["--steps", "-1"], ["--steps", "1.5"], ["--steps", "nan"],
    ["--cfg-scale", "nan"], ["--cfg-scale", "inf"], ["--cfg-scale", "-inf"],
    ["--cfg-scale", "-1"], ["--cfg-scale", "abc"],
    ["--lat-h", "0"], ["--lat-w", "-3"], ["--seed", "-1"],
    ["--lat-h", "65", "--lat-w", "64"],
    ["--save-tokens", "t.txt"],
])
def test_generate_invalid_args_before_any_loading(calls, tmp_path, capsys, bad):
    rc = cli.main(["generate", "x", "-o", str(tmp_path / "o.png")] + bad)
    assert rc == 2
    assert calls["events"] == []
    cap = capsys.readouterr()
    assert cap.out == "" and "error" in cap.err.lower()


def test_generate_missing_output_or_empty_prompt(calls, tmp_path, capsys):
    assert cli.main(["generate", "x"]) == 2
    assert cli.main(["generate", "  ", "-o", str(tmp_path / "o.png")]) == 2
    assert calls["events"] == []


def test_generate_output_must_be_png(calls, tmp_path):
    assert cli.main(["generate", "x", "-o", str(tmp_path / "o.jpg")]) == 2
    assert cli.main(["generate", "x", "-o", str(tmp_path / "o.PNG")]) == 0


def test_ask_image_requires_ibq_and_prints_answer(calls, tmp_path, capsys):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert cli.main(["ask", str(img), "what?", "--low-memory"]) == 0
    assert capsys.readouterr().out == "a cat\n"
    assert calls["artifacts"][1] is True
    path, q, kw = calls["ask"]
    assert path == img and q == "what?"
    assert kw == dict(artifacts=calls["arts"], device="DEV", dtype="DT",
                      lat_h=32, lat_w=32, low_memory=True)


@pytest.mark.parametrize("name", ["t.npy", "t.NPY", "t.Npy"])
def test_ask_npy_case_insensitive_does_not_require_ibq(calls, tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"x")
    assert cli.main(["ask", str(f), "q", "--lat-h", "2", "--lat-w", "3"]) == 0
    assert calls["artifacts"][1] is False
    assert calls["ask"][2]["lat_h"] == 2 and calls["ask"][2]["lat_w"] == 3


def test_ask_missing_input_and_bad_grid(calls, tmp_path, capsys):
    assert cli.main(["ask", str(tmp_path / "nope.png"), "q"]) == 1
    f = tmp_path / "a.npy"
    f.write_bytes(b"x")
    assert cli.main(["ask", str(f), "q", "--lat-h", "0"]) == 2
    # 4096 grid + 128 answer tokens already exceeds the 4096 limit
    assert cli.main(["ask", str(f), "q", "--lat-h", "64", "--lat-w", "64"]) == 2
    assert cli.main(["ask", str(f), " "]) == 2
    assert calls["events"] == []
    assert "error" in capsys.readouterr().err.lower()


def test_ask_has_no_generation_flags(calls, tmp_path):
    f = tmp_path / "a.npy"
    f.write_bytes(b"x")
    assert cli.main(["ask", str(f), "q", "--steps", "4"]) == 2


def test_text_prints_response_and_skips_ibq(calls, capsys):
    assert cli.main(["text", "hi", "--system-prompt", "S"]) == 0
    assert capsys.readouterr().out == "hello\n"
    assert calls["artifacts"][1] is False
    assert calls["text"] == ("hi", dict(artifacts=calls["arts"], device="DEV",
                                        dtype="DT", system_prompt="S"))


def test_text_has_no_low_memory(calls):
    assert cli.main(["text", "hi", "--low-memory"]) == 2
    assert calls["events"] == []


def test_text_help_marks_experimental(capsys):
    assert cli.main(["text", "--help"]) == 0
    assert "experimental" in capsys.readouterr().out.lower()
    assert cli.main(["--help"]) == 0
    assert "experimental" in capsys.readouterr().out.lower()


def test_pipeline_error_goes_to_stderr_exit_1(calls, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("sequence too long")

    monkeypatch.setattr(cli.pipeline, "text", boom)
    assert cli.main(["text", "hi"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "sequence too long" in cap.err


def test_device_error_reported(calls, monkeypatch, capsys):
    def boom(req):
        raise RuntimeError("mps unavailable")

    monkeypatch.setattr(cli.device, "resolve_device", boom)
    assert cli.main(["text", "hi", "--device", "mps"]) == 1
    assert "mps unavailable" in capsys.readouterr().err


def test_download_prints_paths(monkeypatch, tmp_path, capsys):
    arts = ArtifactPaths(tmp_path / "gestalt", tmp_path / "ibq" / "m.ckpt")
    seen = []
    monkeypatch.setattr(cli.artifacts, "download_artifacts",
                        lambda d: seen.append(d) or arts)
    assert cli.main(["download", "--dir", str(tmp_path)]) == 0
    assert seen == [tmp_path]
    out = capsys.readouterr().out
    assert str(arts.gestalt_dir) in out and str(arts.ibq_checkpoint) in out


def test_download_failure(monkeypatch, capsys):
    def boom(d):
        raise RuntimeError("incomplete download")

    monkeypatch.setattr(cli.artifacts, "download_artifacts", boom)
    assert cli.main(["download"]) == 1
    assert "incomplete download" in capsys.readouterr().err


def test_doctor_reports_status_and_files(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.device, "device_status",
                        lambda: {"platform": "Darwin", "mps_available": True, "mem": None})
    seen = []
    monkeypatch.setattr(cli.artifacts, "inspect_artifacts",
                        lambda d: seen.append(d) or {"model.safetensors": True, "ibq.ckpt": False})
    assert cli.main(["doctor", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert seen == [tmp_path]
    assert "platform: Darwin" in out and "mps_available: True" in out
    assert "model.safetensors: present" in out and "ibq.ckpt: missing" in out


def test_no_command_is_usage_error(capsys):
    assert cli.main([]) == 2


def test_nan_helper():
    with pytest.raises(Exception):
        cli._cfg_scale(str(math.nan))


def test_module_entrypoint_runs():
    r = subprocess.run([sys.executable, "-m", "gestalt_mps.cli", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "generate" in r.stdout

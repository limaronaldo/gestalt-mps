import hashlib
import shutil
import socket
from pathlib import Path

import pytest

from gestalt_mps import artifacts
from gestalt_mps.artifacts import ArtifactError, FileSpec

GREV = "a" * 40
IREV = "b" * 40
CONTENT = {
    "config.json": b"{}",
    "tokenizer.json": b"tok",
    "weights.safetensors": b"w" * 50,
    "ibq.ckpt": b"i" * 70,
}


def _specs():
    def mk(repo, rev, sub, name):
        data = CONTENT[name]
        return FileSpec(repo, rev, sub, name, len(data), hashlib.sha256(data).hexdigest())

    return (
        mk("o/g", GREV, "gestalt", "config.json"),
        mk("o/g", GREV, "gestalt", "tokenizer.json"),
        mk("o/g", GREV, "gestalt", "weights.safetensors"),
        mk("o/i", IREV, "ibq", "ibq.ckpt"),
    )


class FakeHub:
    def __init__(self, tmp_path):
        self.calls = []
        self.cache = tmp_path / "hubcache"
        self.corrupt = set()

    def __call__(self, *, repo_id, filename, revision, local_dir=None, local_files_only=False):
        self.calls.append(
            dict(repo_id=repo_id, filename=filename, revision=revision,
                 local_dir=local_dir, local_files_only=local_files_only)
        )
        base = Path(local_dir) if local_dir else self.cache / repo_id.replace("/", "--") / revision
        target = base / filename
        if target.exists():
            return str(target)
        if local_files_only:
            raise FileNotFoundError("not cached: " + filename)
        data = CONTENT[filename]
        if filename in self.corrupt:
            data = b"x" * len(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = FakeHub(tmp_path)
    monkeypatch.setattr(artifacts, "ARTIFACT_SPECS", _specs())
    monkeypatch.setattr(artifacts, "_hf_hub_download", h)
    monkeypatch.setattr(artifacts, "_cache_root", lambda: h.cache)
    artifacts._VERIFIED.clear()
    return h


def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def test_production_specs_pinned():
    by = {s.filename: s for s in artifacts.ARTIFACT_SPECS}
    assert by["model.safetensors"].revision == "42a8b5d94f80352515603a5b4bab9d2742b6ed80"
    assert by["model.safetensors"].size == 17_507_601_848
    assert by["imagenet256_16384.ckpt"].revision == "ef2200997f97118050c5b17dde48e8b78690f866"
    assert by["imagenet256_16384.ckpt"].size == 888_207_474
    assert by["imagenet256_16384.ckpt"].sha256.startswith("26ecf8be")
    assert {s.filename for s in artifacts.ARTIFACT_SPECS if s.subdir == "gestalt"} == {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "model.safetensors",
    }


def test_download_dir_then_resolve_offline(hub, tmp_path, monkeypatch):
    d = tmp_path / "root"
    paths = artifacts.download_artifacts(d)
    assert paths.gestalt_dir == d / "gestalt"
    assert paths.ibq_checkpoint == d / "ibq" / "ibq.ckpt"
    assert all(c["revision"] in (GREV, IREV) for c in hub.calls)
    hub.calls.clear()
    _no_network(monkeypatch)
    assert artifacts.resolve_artifacts(d, require_ibq=True) == paths
    assert hub.calls == []
    assert artifacts.resolve_artifacts(d, require_ibq=False).ibq_checkpoint is None


def test_cache_mode_resolve_is_local_only(hub, monkeypatch):
    paths = artifacts.download_artifacts(None)
    assert paths.gestalt_dir.name == GREV
    hub.calls.clear()
    _no_network(monkeypatch)
    out = artifacts.resolve_artifacts(None, require_ibq=True)
    assert out.gestalt_dir == paths.gestalt_dir
    assert hub.calls and all(c["local_files_only"] for c in hub.calls)
    assert all(c["local_dir"] is None for c in hub.calls)
    hub.calls.clear()
    assert all(artifacts.inspect_artifacts(None).values())
    assert hub.calls and all(c["local_files_only"] for c in hub.calls)


def test_cache_mode_missing_is_actionable(hub):
    with pytest.raises(ArtifactError, match="gestalt download"):
        artifacts.resolve_artifacts(None, require_ibq=False)
    assert not any(artifacts.inspect_artifacts(None).values())
    assert all(c["local_files_only"] for c in hub.calls)


def test_custom_dir_incomplete_fails(hub, tmp_path):
    d = tmp_path / "root"
    (d / "gestalt").mkdir(parents=True)
    (d / "gestalt" / "config.json").write_bytes(CONTENT["config.json"])
    with pytest.raises(ArtifactError, match="tokenizer.json") as e:
        artifacts.resolve_artifacts(d, require_ibq=False)
    assert "gestalt download --dir" in str(e.value)
    ins = artifacts.inspect_artifacts(d)
    assert ins["gestalt/config.json"] is True
    assert ins["gestalt/tokenizer.json"] is False
    assert hub.calls == []


def test_custom_dir_truncated_weight_fails(hub, tmp_path):
    d = tmp_path / "root"
    artifacts.download_artifacts(d)
    (d / "gestalt" / "weights.safetensors").write_bytes(b"w")
    with pytest.raises(ArtifactError, match="weights.safetensors"):
        artifacts.resolve_artifacts(d, require_ibq=False)
    assert artifacts.inspect_artifacts(d)["gestalt/weights.safetensors"] is False


def test_custom_dir_rejects_same_size_metadata_from_another_snapshot(hub, tmp_path):
    directory = tmp_path / "root"
    artifacts.download_artifacts(directory)
    (directory / "gestalt" / "config.json").write_bytes(b"[]")

    with pytest.raises(ArtifactError, match="config.json.*SHA-256"):
        artifacts.resolve_artifacts(directory, require_ibq=False)


def test_ibq_not_required_ignores_missing_ibq(hub, tmp_path):
    d = tmp_path / "root"
    artifacts.download_artifacts(d)
    (d / "ibq" / "ibq.ckpt").unlink()
    assert artifacts.resolve_artifacts(d, require_ibq=False).ibq_checkpoint is None
    with pytest.raises(ArtifactError, match="ibq.ckpt"):
        artifacts.resolve_artifacts(d, require_ibq=True)


def test_bad_weight_hash_fails(hub, tmp_path):
    hub.corrupt = {"weights.safetensors"}
    with pytest.raises(ArtifactError, match="SHA-256"):
        artifacts.download_artifacts(tmp_path / "root")


def test_partial_download_is_idempotent(hub, tmp_path):
    d = tmp_path / "root"
    hub.corrupt = {"ibq.ckpt"}
    with pytest.raises(ArtifactError):
        artifacts.download_artifacts(d)
    # corrupt file is removed in dir mode so a rerun can repair it
    assert not (d / "ibq" / "ibq.ckpt").exists()
    hub.corrupt = set()
    hub.calls.clear()
    artifacts.download_artifacts(d)
    assert [c["filename"] for c in hub.calls].count("ibq.ckpt") == 1
    artifacts.download_artifacts(d)  # all present: still succeeds
    assert artifacts.resolve_artifacts(d, require_ibq=True).ibq_checkpoint.exists()


def test_download_reports_space_shortage(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(ArtifactError, match="free") as e:
        artifacts.download_artifacts(tmp_path / "new" / "root")
    assert "bytes" in str(e.value)
    assert hub.calls == []


def test_verify_ibq_checkpoint(hub, tmp_path, monkeypatch):
    p = artifacts.download_artifacts(tmp_path / "root").ibq_checkpoint
    artifacts.verify_ibq_checkpoint(p)
    calls = []
    orig = artifacts._sha256_file
    monkeypatch.setattr(artifacts, "_sha256_file", lambda *a, **k: calls.append(1) or orig(*a, **k))
    artifacts.verify_ibq_checkpoint(p)  # cached per process
    assert calls == []
    p.write_bytes(b"j" * 70)
    with pytest.raises(ArtifactError, match="SHA-256"):
        artifacts.verify_ibq_checkpoint(p)
    p.write_bytes(b"short")
    with pytest.raises(ArtifactError, match="size"):
        artifacts.verify_ibq_checkpoint(p)
    with pytest.raises(ArtifactError, match="not found"):
        artifacts.verify_ibq_checkpoint(tmp_path / "nope.ckpt")


def test_sha256_streams_in_chunks(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc" * 1000)
    assert artifacts._sha256_file(p, chunk_size=7) == hashlib.sha256(b"abc" * 1000).hexdigest()

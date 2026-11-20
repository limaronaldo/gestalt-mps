"""Pinned Hugging Face artifacts: download, offline lookup, verification.

Only files in ``ARTIFACT_SPECS`` are ever fetched, always at a pinned
revision.  Lookup (``resolve_artifacts`` / ``inspect_artifacts``) never uses
the network: cache mode asks the Hub client with ``local_files_only=True`` and
``--dir`` mode only looks at the filesystem.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

GESTALT_REPO = "GeWu-Lab/Gestalt"
GESTALT_REVISION = "42a8b5d94f80352515603a5b4bab9d2742b6ed80"
IBQ_REPO = "TencentARC/IBQ-Tokenizer-16384"
IBQ_REVISION = "ef2200997f97118050c5b17dde48e8b78690f866"

_CHUNK = 8 * 1024 * 1024


class ArtifactError(RuntimeError):
    """Actionable artifact problem (missing, incomplete or corrupt files)."""


@dataclass(frozen=True)
class ArtifactPaths:
    gestalt_dir: Path
    ibq_checkpoint: Path | None


@dataclass(frozen=True)
class FileSpec:
    repo_id: str
    revision: str
    subdir: str  # "gestalt" or "ibq"; also the --dir subdirectory
    filename: str
    size: int | None = None  # bytes, when published
    sha256: str | None = None  # hex, when published

    @property
    def key(self) -> str:
        return self.subdir + "/" + self.filename


def _g(name: str, size: int | None = None, sha: str | None = None) -> FileSpec:
    return FileSpec(GESTALT_REPO, GESTALT_REVISION, "gestalt", name, size, sha)


ARTIFACT_SPECS: tuple[FileSpec, ...] = (
    _g("config.json"),
    _g("generation_config.json"),
    _g("tokenizer.json"),
    _g("tokenizer_config.json"),
    _g(
        "model.safetensors",
        17_507_601_848,
        "a6066afc26b7b28527e68e28b5fc90f302469194270ed24999fa5ffdede5e269",
    ),
    FileSpec(
        IBQ_REPO,
        IBQ_REVISION,
        "ibq",
        "imagenet256_16384.ckpt",
        888_207_474,
        "26ecf8be32c2796edd03d9b1aec1fe1fc18e9dae2ded51ef54b504cc32d05144",
    ),
)

# (resolved path, size, mtime_ns) of files hashed OK in this process.
_VERIFIED: set[tuple[str, int, int]] = set()


# -- injectable Hub access ---------------------------------------------------

def _hf_hub_download(*, repo_id, filename, revision, local_dir=None, local_files_only=False):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=None if local_dir is None else str(local_dir),
        local_files_only=local_files_only,
    )


def _cache_root() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


# -- helpers -----------------------------------------------------------------

def _sha256_file(path: Path, chunk_size: int = _CHUNK) -> str:
    digest = hashlib.sha256()
    buf = bytearray(chunk_size)
    view = memoryview(buf)
    with open(path, "rb", buffering=0) as handle:
        while True:
            n = handle.readinto(buf)
            if not n:
                break
            digest.update(view[:n])
    return digest.hexdigest()


def _cache_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return (str(path.resolve()), st.st_size, st.st_mtime_ns)


def _check_file(spec: FileSpec, path: Path, *, full: bool) -> None:
    """Raise ArtifactError unless ``path`` matches the spec (size, and hash if ``full``)."""
    if not path.is_file():
        raise ArtifactError(f"{spec.key} not found at {path}.")
    size = path.stat().st_size
    if spec.size is not None and size != spec.size:
        raise ArtifactError(
            f"{spec.key} has size {size} bytes, expected {spec.size} "
            f"(incomplete or wrong file at {path})."
        )
    if full and spec.sha256 is not None:
        key = _cache_key(path)
        if key in _VERIFIED:
            return
        actual = _sha256_file(path)
        if actual != spec.sha256:
            raise ArtifactError(
                f"{spec.key} failed SHA-256 verification (got {actual}, expected "
                f"{spec.sha256}) at {path}."
            )
        _VERIFIED.add(key)


def _fix_hint(directory: Path | None) -> str:
    if directory is None:
        return "Run `gestalt download` (needs network) to populate the Hugging Face cache."
    return f"Run `gestalt download --dir {directory}` (needs network) to complete it."


def _dir_path(directory: Path, spec: FileSpec) -> Path:
    return Path(directory) / spec.subdir / spec.filename


def _cache_lookup(spec: FileSpec) -> Path | None:
    try:
        found = _hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_files_only=True,
        )
    except Exception:  # not cached at this exact revision (or cache unreadable)
        return None
    path = Path(found)
    return path if path.is_file() else None


def _locate(spec: FileSpec, directory: Path | None) -> Path | None:
    """Existing, size-consistent path for ``spec`` or None.  Never uses the network."""
    path = _dir_path(directory, spec) if directory is not None else _cache_lookup(spec)
    if path is None or not path.is_file():
        return None
    if spec.size is not None and path.stat().st_size != spec.size:
        return None
    return path


def _specs_for(subdir: str) -> list[FileSpec]:
    return [s for s in ARTIFACT_SPECS if s.subdir == subdir]


# -- public API --------------------------------------------------------------

def inspect_artifacts(directory: Path | None) -> dict[str, bool]:
    """Presence of every expected file; no weight reads and no network."""
    return {s.key: _locate(s, directory) is not None for s in ARTIFACT_SPECS}


def resolve_artifacts(directory: Path | None, *, require_ibq: bool) -> ArtifactPaths:
    wanted = [s for s in ARTIFACT_SPECS if require_ibq or s.subdir != "ibq"]
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for spec in wanted:
        path = _locate(spec, directory)
        if path is None:
            missing.append(spec.key)
        else:
            paths[spec.key] = path
    if missing:
        where = "Hugging Face cache" if directory is None else f"directory {directory}"
        raise ArtifactError(
            f"Missing or incomplete artifacts in {where}: {', '.join(missing)}. "
            + _fix_hint(directory)
        )
    gestalt_files = [paths[s.key] for s in _specs_for("gestalt")]
    gestalt_dir = gestalt_files[0].parent
    if any(p.parent != gestalt_dir for p in gestalt_files):
        raise ArtifactError(
            "Gestalt files come from different cache snapshots; " + _fix_hint(directory)
        )
    ibq = [paths[s.key] for s in _specs_for("ibq")] if require_ibq else []
    return ArtifactPaths(gestalt_dir=gestalt_dir, ibq_checkpoint=ibq[0] if ibq else None)


def _destination_free_bytes(directory: Path | None) -> tuple[Path, int]:
    probe = Path(directory) if directory is not None else _cache_root()
    probe = probe.absolute()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe, shutil.disk_usage(probe).free


def download_artifacts(directory: Path | None) -> ArtifactPaths:
    """Fetch the allowlisted files at pinned revisions and verify the weights.

    Safe to re-run: complete files are kept, partial or corrupt ones in
    ``--dir`` mode are removed and fetched again.
    """
    todo: list[FileSpec] = []
    present: dict[str, Path] = {}
    for spec in ARTIFACT_SPECS:
        path = _locate(spec, directory)
        if path is not None:
            present[spec.key] = path
        else:
            todo.append(spec)

    need = sum(s.size or 0 for s in todo)
    if need:
        probe, free = _destination_free_bytes(directory)
        if free < need:
            raise ArtifactError(
                f"Not enough free space at {probe}: {need} bytes missing to download, "
                f"{free} bytes free. Free up space or choose another --dir."
            )

    for spec in ARTIFACT_SPECS:
        path = present.get(spec.key)
        if path is not None:
            try:
                _check_file(spec, path, full=True)
                continue
            except ArtifactError:
                if directory is None:
                    raise ArtifactError(
                        f"Cached {spec.key} is corrupt at {path}. Delete that file "
                        "(and its blob) from the Hugging Face cache, then rerun `gestalt download`."
                    ) from None
                path.unlink()
        _fetch(spec, directory)
    return resolve_artifacts(directory, require_ibq=True)


def _fetch(spec: FileSpec, directory: Path | None) -> None:
    local_dir = None if directory is None else Path(directory) / spec.subdir
    try:
        found = _hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=local_dir,
            local_files_only=False,
        )
    except Exception as exc:
        raise ArtifactError(
            f"Download of {spec.key} from {spec.repo_id}@{spec.revision[:12]} failed: {exc}. "
            "Check the network and rerun the same command; finished files are reused."
        ) from exc
    path = Path(found)
    try:
        _check_file(spec, path, full=True)
    except ArtifactError as exc:
        if directory is not None and path.is_file():
            path.unlink()
            hint = "The bad file was removed; rerun the same command."
        else:
            hint = "Delete it from the Hugging Face cache and rerun."
        raise ArtifactError(f"{exc} {hint}") from exc


def verify_ibq_checkpoint(checkpoint: Path) -> None:
    """Size + streamed SHA-256 check; must run before the checkpoint is deserialized.

    Each file is hashed at most once per process (keyed by path, size, mtime).
    """
    spec = _specs_for("ibq")[0]
    _check_file(spec, Path(checkpoint), full=True)

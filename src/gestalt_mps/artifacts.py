"""Interface reserved by the implementation plan."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    gestalt_dir: Path
    ibq_checkpoint: Path | None



def resolve_artifacts(directory, *, require_ibq: bool):
    raise NotImplementedError



def download_artifacts(directory):
    raise NotImplementedError



def inspect_artifacts(directory):
    raise NotImplementedError



def verify_ibq_checkpoint(checkpoint):
    raise NotImplementedError

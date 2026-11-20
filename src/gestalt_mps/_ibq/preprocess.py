"""Fixed image preprocessing of the adaptation (not part of the SEED-Voken sources).

EXIF orientation -> RGB -> bicubic resize preserving aspect until the target is covered ->
centre crop -> pixels normalised to ``[-1, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


def check_grid(lat_h: object, lat_w: object) -> tuple[int, int]:
    for name, value in (("lat_h", lat_h), ("lat_w", lat_w)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(lat_h), int(lat_w)


def preprocess_image(path: Path, *, lat_h: int, lat_w: int, patch: int = 16) -> torch.Tensor:
    """Return a float32 CPU tensor ``[1, 3, patch*lat_h, patch*lat_w]`` in ``[-1, 1]``."""
    lat_h, lat_w = check_grid(lat_h, lat_w)
    target_h, target_w = patch * lat_h, patch * lat_w
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        image = image.convert("RGB")
        width, height = image.size
        scale = max(target_w / width, target_h / height)
        new_w = max(target_w, round(width * scale))
        new_h = max(target_h, round(height * scale))
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0  # H, W, 3
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))[None]

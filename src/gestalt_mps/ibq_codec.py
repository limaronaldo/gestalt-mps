"""IBQ-16384 image codec: pixels -> raw token IDs and back (inference only, FP32).

The encoder, decoder, quantizer and EMA handling are a small port of TencentARC/SEED-Voken
(Apache License 2.0, revision c1544efff959ba18d453089071307a7115698551); see
``gestalt_mps/_ibq`` and ``THIRD_PARTY.md``. IDs handled here are raw codebook indices in
``[0, n_embed)``; the Gestalt vocabulary offset is applied only at the Gestalt boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import artifacts
from ._ibq.checkpoint import (
    EMA_EXCLUDED_PARAMS,
    IBQCheckpointError,
    map_ema_state_dict,
    read_state_dict,
)
from ._ibq.config import OFFICIAL_CONFIG, IBQConfig
from ._ibq.model import IBQModel
from ._ibq.preprocess import check_grid, preprocess_image

__all__ = ["IBQCodec", "IBQCheckpointError", "preprocess_image"]


class IBQCodec:
    """Wraps an :class:`IBQModel` that is already in FP32, eval mode, on ``device``."""

    def __init__(self, model: IBQModel, device: torch.device, config: IBQConfig = OFFICIAL_CONFIG):
        self.config = config
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, device: torch.device, *, config: IBQConfig = OFFICIAL_CONFIG
    ) -> "IBQCodec":
        """Verify, safely deserialise and load the EMA weights of the IBQ checkpoint."""
        checkpoint = Path(checkpoint)
        artifacts.verify_ibq_checkpoint(checkpoint)  # hash check before any deserialisation
        state_dict = read_state_dict(checkpoint)
        model = IBQModel(config)
        mapped = map_ema_state_dict(model, state_dict, ema_excluded=EMA_EXCLUDED_PARAMS)
        model.load_state_dict(mapped, strict=True)
        return cls(model, device, config)

    # ------------------------------------------------------------------ encode

    def encode_image(self, path: Path, *, lat_h: int, lat_w: int) -> np.ndarray:
        """Preprocess ``path`` and return int32 raw IDs of length ``lat_h*lat_w`` (row-major)."""
        x = preprocess_image(Path(path), lat_h=lat_h, lat_w=lat_w, patch=self.config.downsample)
        return self.encode_tensor(x)

    def encode_tensor(self, x: torch.Tensor) -> np.ndarray:
        """Encode an already preprocessed ``[1, 3, H, W]`` tensor in ``[-1, 1]``."""
        f = self.config.downsample
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
            raise ValueError(f"expected a [1, 3, H, W] tensor, got shape {tuple(x.shape)}")
        height, width = int(x.shape[2]), int(x.shape[3])
        if height % f or width % f or height == 0 or width == 0:
            raise ValueError(f"image size {height}x{width} must be a positive multiple of {f}")
        with torch.inference_mode():
            ids = self.model.encode_indices(x.to(device=self.device, dtype=torch.float32))
        expected = (height // f) * (width // f)
        if ids.numel() != expected:
            raise RuntimeError(f"encoder produced {ids.numel()} indices, expected {expected}")
        return np.ascontiguousarray(ids.cpu().numpy().astype(np.int32))

    # ------------------------------------------------------------------ decode

    def _validate_codes(self, codes: object, lat_h: int, lat_w: int) -> np.ndarray:
        if not isinstance(codes, np.ndarray):
            raise ValueError(f"codes must be a numpy ndarray, got {type(codes).__name__}")
        if codes.dtype.kind not in "iu":
            raise ValueError(f"codes must have an integer dtype, got {codes.dtype}")
        if codes.ndim == 1:
            pass
        elif codes.ndim == 2:
            if codes.shape != (lat_h, lat_w):
                raise ValueError(f"codes shape {codes.shape} does not match grid ({lat_h}, {lat_w})")
        else:
            raise ValueError(f"codes shape must be (H*W,) or (H, W), got {codes.shape}")
        if codes.size != lat_h * lat_w:
            raise ValueError(f"codes size {codes.size} != lat_h*lat_w = {lat_h * lat_w}")
        if codes.size and (codes.min() < 0 or codes.max() >= self.config.n_embed):
            raise ValueError(f"codes outside range [0, {self.config.n_embed})")
        return codes.reshape(-1)

    def decode_tokens(self, codes: np.ndarray, *, lat_h: int, lat_w: int) -> Image.Image:
        """Decode raw IDs to an RGB image of ``f*lat_w`` x ``f*lat_h`` pixels."""
        lat_h, lat_w = check_grid(lat_h, lat_w)
        flat = self._validate_codes(codes, lat_h, lat_w)
        with torch.inference_mode():
            ids = torch.from_numpy(flat.astype(np.int64)).to(self.device)
            out = self.model.decode_indices(ids, height=lat_h, width=lat_w)
        return _tensor_to_image(out)


def _tensor_to_image(out: torch.Tensor) -> Image.Image:
    """``[1, 3, H, W]`` in ``[-1, 1]`` -> RGB, with the reference's ``custom_to_pil`` maths."""
    x = out.detach().float().cpu()[0]
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    pixels = (255 * x.permute(1, 2, 0).numpy()).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")

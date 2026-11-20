"""Inference-only IBQ module with the same state-dict names as the original ``IBQ``.

Adapted from TencentARC/SEED-Voken, revision c1544efff959ba18d453089071307a7115698551,
``src/IBQ/models/ibqgan.py`` (``VQModel`` / ``IBQ``), licensed under the Apache License,
Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

Changes from the original: no Lightning, loss, discriminator or EMA module; only
``encoder``, ``decoder``, ``quant_conv``, ``post_quant_conv`` and ``quantize.embedding``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import IBQConfig
from .modules import Decoder, Encoder
from .quantizer import IndexPropagationQuantizer


class IBQModel(nn.Module):
    def __init__(self, config: IBQConfig):
        super().__init__()
        ddconfig = dict(config.ddconfig)
        self.config = config
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = IndexPropagationQuantizer(config.n_embed, config.embed_dim)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, ddconfig["z_channels"], 1)

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` in ``[-1, 1]`` of shape ``[b, 3, H, W]`` -> flat int64 indices."""
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantize.indices(h)

    def decode_indices(self, codes: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Flat raw indices -> reconstructed tensor in roughly ``[-1, 1]``, ``[1, 3, H, W]``."""
        z_q = self.quantize.lookup(codes, batch=1, height=height, width=width)
        z_q = self.post_quant_conv(z_q)
        return self.decoder(z_q)

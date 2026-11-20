"""IBQ index-propagation quantizer (inference only).

Adapted from TencentARC/SEED-Voken, revision c1544efff959ba18d453089071307a7115698551,
``src/IBQ/modules/vqvae/quantize.py`` (``IndexPropagationQuantize``), licensed under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

Changes from the original (inference only):
  * kept only the codebook and the index selection; losses, entropy terms, the
    straight-through one-hot and the ``remap`` option (unused by IBQ-16384) were removed;
  * ``lookup`` replaces ``get_codebook_entry`` and the missing ``embed_code`` that the
    inherited ``decode_code`` expects.

Index selection follows the original exactly: scores are the inner products
``z . e_n`` (NOT Euclidean distances), turned into a softmax over the codebook axis,
and the index is the arg-max of that softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class IndexPropagationQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_embed, embed_dim)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        """Inner-product scores ``[b, n, h, w]`` for latents ``z`` of shape ``[b, d, h, w]``."""
        return einsum("b d h w, n d -> b n h w", z, self.embedding.weight)

    def indices(self, z: torch.Tensor) -> torch.Tensor:
        """Raw codebook indices, flattened in ``(b, h, w)`` row-major order (int64)."""
        soft_one_hot = F.softmax(self.logits(z), dim=1)
        ind = soft_one_hot.max(1, keepdim=True)[1]
        return torch.flatten(ind)

    def lookup(self, codes: torch.Tensor, *, batch: int, height: int, width: int) -> torch.Tensor:
        """Embeddings for flat raw indices as ``[b, d, h, w]`` (BHWC lookup, then BCHW)."""
        z_q = self.embedding(codes.reshape(-1))
        z_q = z_q.view(batch, height, width, self.embed_dim)
        return z_q.permute(0, 3, 1, 2).contiguous()

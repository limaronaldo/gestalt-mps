"""IBQ-16384 architecture constants.

Adapted from TencentARC/SEED-Voken, revision c1544efff959ba18d453089071307a7115698551,
``configs/IBQ/gpu/imagenet_ibqgan_16384.yaml`` (``model.init_args``), licensed under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

Changes: only the values needed for inference were kept (no loss, optimiser, EMA or
training settings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IBQConfig:
    ddconfig: dict[str, Any]
    n_embed: int
    embed_dim: int

    @property
    def downsample(self) -> int:
        """Pixels per latent cell: ``2 ** num_down`` with ``num_down = len(ch_mult) - 1``."""
        return 2 ** (len(self.ddconfig["ch_mult"]) - 1)


OFFICIAL_CONFIG = IBQConfig(
    ddconfig=dict(
        double_z=False,
        z_channels=256,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=4,
        attn_resolutions=(16,),
        dropout=0.0,
    ),
    n_embed=16384,
    embed_dim=256,
)

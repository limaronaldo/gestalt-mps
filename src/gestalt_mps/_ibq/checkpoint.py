"""Checkpoint reading and EMA weight mapping for the official IBQ checkpoint.

Adapted from TencentARC/SEED-Voken, revision c1544efff959ba18d453089071307a7115698551:
``src/IBQ/modules/ema.py`` (``LitEma``) and ``evaluation_image.py`` (evaluation with
``model.ema_scope()``), licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

How the reference builds its EMA weights
----------------------------------------
``LitEma`` registers one buffer per ``model.named_parameters()`` entry whose
``requires_grad`` is true, named ``name.replace('.', '')``. In a Lightning checkpoint they
appear as ``model_ema.<name without dots>``. Evaluation runs under ``ema_scope()``, which
copies each shadow tensor onto the matching parameter. Parameters with
``requires_grad == False`` are not tracked and keep their regular tensor.

Every tensor of the inference model is such a trainable parameter (it has no buffers), so
for the official architecture the exclusion set is empty. The mapping nevertheless takes
the set explicitly so that regular tensors are used only where the reference provably
excludes EMA, and never as a silent substitute for a missing shadow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

EMA_PREFIX = "model_ema."
EMA_EXCLUDED_PARAMS: frozenset[str] = frozenset()

_MAX_LISTED = 12


class IBQCheckpointError(RuntimeError):
    """The IBQ checkpoint cannot be read or does not contain the required tensors."""


def ema_key(param_name: str) -> str:
    """Key of the EMA shadow of ``param_name`` (dots removed, as ``LitEma`` does)."""
    return EMA_PREFIX + param_name.replace(".", "")


def read_state_dict(checkpoint: Path) -> Mapping[str, Any]:
    """Read ``state_dict`` with the restricted unpickler. The file must be hash-verified."""
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surfaced with context, original chained
        raise IBQCheckpointError(
            f"{checkpoint}: could not be read with torch.load(weights_only=True); the IBQ codec "
            f"is blocked until a trusted conversion exists ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise IBQCheckpointError(f"{checkpoint}: expected a mapping with a 'state_dict' mapping")
    return payload["state_dict"]


def _summary(items: Iterable[str]) -> str:
    items = list(items)
    shown = ", ".join(items[:_MAX_LISTED])
    extra = len(items) - _MAX_LISTED
    return shown + (f", ... (+{extra} more)" if extra > 0 else "")


def map_ema_state_dict(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    ema_excluded: frozenset[str] | set[str] = EMA_EXCLUDED_PARAMS,
) -> dict[str, torch.Tensor]:
    """Return ``{model tensor name: float32 tensor}`` taken from the EMA shadows.

    Every parameter must have its ``model_ema.*`` shadow, except names in ``ema_excluded``
    which use the regular tensor. Buffers (none in the official model) use regular tensors.
    All required tensors are strictly checked for presence, type and shape.
    """
    wanted: dict[str, tuple[str, torch.Size]] = {}  # model name -> (checkpoint key, shape)
    seen: dict[str, str] = {}
    for name, param in model.named_parameters():
        key = name if name in ema_excluded else ema_key(name)
        if key in seen:
            raise IBQCheckpointError(f"EMA key collision: {seen[key]!r} and {name!r} -> {key!r}")
        seen[key] = name
        wanted[name] = (key, param.shape)
    for name, buf in model.named_buffers():
        wanted[name] = (name, buf.shape)

    missing: list[str] = []
    not_tensor: list[str] = []
    bad_shape: list[str] = []
    mapped: dict[str, torch.Tensor] = {}
    for name, (key, shape) in wanted.items():
        if key not in state_dict:
            missing.append(key)
            continue
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            not_tensor.append(key)
        elif value.shape != shape:
            bad_shape.append(f"{key} {tuple(value.shape)} != {tuple(shape)}")
        else:
            mapped[name] = value.detach().to(device="cpu", dtype=torch.float32)

    problems = []
    if missing:
        problems.append(f"missing {len(missing)} required key(s): {_summary(missing)}")
    if not_tensor:
        problems.append(f"value is not a tensor for: {_summary(not_tensor)}")
    if bad_shape:
        problems.append(f"shape mismatch for: {_summary(bad_shape)}")
    if problems:
        raise IBQCheckpointError("invalid IBQ checkpoint: " + "; ".join(problems))
    return mapped

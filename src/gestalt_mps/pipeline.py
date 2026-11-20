"""Local inference and file I/O for the Gestalt command-line entry points."""

import gc
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from gestalt.model.builder import load_tokenizer
from gestalt.model.config import VOCAB_CONFIG
from gestalt.model.templates import (
    DEFAULT_MMU_SYSTEM_PROMPT,
    build_mmu_condition,
    build_t2i_condition,
    build_text_condition,
)

from .artifacts import ArtifactPaths
from .ibq_codec import IBQCodec
from .model import load_gestalt


_MAX_SEQUENCE_LENGTH = 4096
_ANSWER_LENGTH = 128
_ANSWER_STEPS = 64
_BLOCK_LENGTH = 64
_CODEBOOK_SIZE = VOCAB_CONFIG.VQVAE_CODEBOOK_SIZE
_VISUAL_OFFSET = VOCAB_CONFIG.VISUAL_TOKEN_OFFSET

_phase_observer: Callable[[str], None] | None = None


def set_phase_observer(observer: Callable[[str], None] | None) -> None:
    """Observe phase boundaries in a standalone validation process."""
    global _phase_observer
    _phase_observer = observer


@contextmanager
def _phase(name: str) -> Iterator[None]:
    observer = _phase_observer
    if observer is None:
        yield
        return
    observer(f"{name}:start")
    try:
        yield
    finally:
        observer(f"{name}:end")


def _release_unused_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _check_length(length: int) -> None:
    if length > _MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence requires {length} positions, exceeding the Gestalt limit "
            f"of {_MAX_SEQUENCE_LENGTH}. Shorten the prompt or reduce the image grid."
        )


def _grid_size(lat_h: int, lat_w: int) -> int:
    if lat_h <= 0 or lat_w <= 0:
        raise ValueError("Image grid dimensions must be positive.")
    size = lat_h * lat_w
    _check_length(size)
    return size


def _encode_prompts(directory: Path, system_prompt: str, user_prompt: str) -> tuple[list[int], list[int]]:
    tokenizer = load_tokenizer(str(directory))
    return (
        tokenizer.encode(system_prompt, add_special_tokens=False),
        tokenizer.encode(user_prompt, add_special_tokens=False),
    )


def _require_checkpoint(artifacts: ArtifactPaths) -> Path:
    if artifacts.ibq_checkpoint is None:
        raise ValueError("An IBQ checkpoint is required to process image pixels.")
    return artifacts.ibq_checkpoint


def _read_tokens(path: Path, lat_h: int, lat_w: int) -> np.ndarray:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError as exc:
        raise ValueError(f"Cannot read an .npy token array from {path}: {exc}") from exc

    if isinstance(values, np.lib.npyio.NpzFile):
        values.close()
        raise ValueError(f"Expected one .npy token array in {path}.")

    if values.shape not in {(lat_h * lat_w,), (lat_h, lat_w)}:
        raise ValueError(f"Invalid token array shape {values.shape} for a {lat_h} × {lat_w} image grid.")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Token array must have an integer dtype, got {values.dtype}.")
    if values.min() < 0 or values.max() >= _CODEBOOK_SIZE:
        raise ValueError(f"Raw IBQ tokens must be in the range [0, {_CODEBOOK_SIZE}).")
    return np.array(values, dtype=np.int32, order="C", copy=True).reshape(-1)


def _raw_visual_ids(visual_ids: torch.Tensor, grid_size: int) -> np.ndarray:
    values = visual_ids.detach().cpu().numpy()
    if values.shape != (1, grid_size) or not np.issubdtype(values.dtype, np.integer):
        raise RuntimeError(f"Gestalt returned an invalid visual token array for {grid_size} positions.")
    if values.min() < _VISUAL_OFFSET or values.max() >= _VISUAL_OFFSET + _CODEBOOK_SIZE:
        raise RuntimeError("Gestalt returned visual token IDs outside the model-space range.")
    return values[0].astype(np.int32) - _VISUAL_OFFSET


def _decode_answer(tokenizer, tokens: list[int]) -> str:
    # The model has extra rows for interplay and reserved IDs absent from the tokenizer.
    decodable = [token for token in tokens if 0 <= token < VOCAB_CONFIG.VISUAL_TOKEN_END]
    return tokenizer.decode(decodable, skip_special_tokens=True)


@torch.inference_mode()
def generate(
    prompt: str,
    output: Path,
    *,
    artifacts: ArtifactPaths,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    cfg_scale: float,
    lat_h: int,
    lat_w: int,
    seed: int | None,
    system_prompt: str,
    save_tokens: Path | None,
    low_memory: bool,
) -> None:
    size = _grid_size(lat_h, lat_w)
    if steps <= 0:
        raise ValueError("Generation steps must be positive.")
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("CFG scale must be finite and non-negative.")
    if seed is not None and not -(2**63) <= seed < 2**64:
        raise ValueError("The seed must be between -2**63 and 2**64 - 1.")
    if save_tokens is not None and output.resolve() == save_tokens.resolve():
        raise ValueError("The PNG and token array require distinct output paths.")
    checkpoint = _require_checkpoint(artifacts)

    # T2I follows the condition with ANSWER_START, IMAGE_START, the grid,
    # IMAGE_END and ANSWER_END. The CFG branch omits the user text span.
    _check_length(len(build_t2i_condition([], [])) + size + 4)
    system, user = _encode_prompts(artifacts.gestalt_dir, system_prompt, prompt)
    _check_length(len(build_t2i_condition(system, user)) + size + 4)

    with _phase("model_load"):
        model, _ = load_gestalt(artifacts.gestalt_dir, device, dtype)
    with _phase("model_run"):
        if seed is not None:
            torch.manual_seed(seed)
        codes = _raw_visual_ids(model.generate_t2i(
            system_prompt_tokens=system,
            user_prompt_tokens=user,
            lat_h=lat_h,
            lat_w=lat_w,
            timesteps=steps,
            temperature=1.0,
            cfg_scale=cfg_scale,
        ), size)
    if low_memory:
        with _phase("component_release"):
            del model
            _release_unused_memory(device)

    with _phase("codec_load"):
        codec = IBQCodec.from_checkpoint(checkpoint, device)
    with _phase("codec_run"):
        image = codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w)
    image.save(output, format="PNG")
    if save_tokens is not None:
        with save_tokens.open("wb") as stream:
            np.save(stream, codes, allow_pickle=False)


@torch.inference_mode()
def ask(
    input_path: Path,
    question: str,
    *,
    artifacts: ArtifactPaths,
    device: torch.device,
    dtype: torch.dtype,
    lat_h: int,
    lat_w: int,
    low_memory: bool,
) -> str:
    size = _grid_size(lat_h, lat_w)
    # One placeholder accounts for the image markers in the shared condition.
    # The response then adds ANSWER_START, BOS, mask positions and ANSWER_END.
    _check_length(len(build_mmu_condition([], [[_VISUAL_OFFSET]], [])) - 1 + size + 3 + _ANSWER_LENGTH)
    from_tokens = input_path.suffix.lower() == ".npy"
    if from_tokens:
        codes = _read_tokens(input_path, lat_h, lat_w)
    else:
        checkpoint = _require_checkpoint(artifacts)

    system, user = _encode_prompts(artifacts.gestalt_dir, DEFAULT_MMU_SYSTEM_PROMPT, question)
    _check_length(len(build_mmu_condition(system, [[_VISUAL_OFFSET]], user)) - 1 + size + 3 + _ANSWER_LENGTH)

    if not from_tokens:
        with _phase("codec_load"):
            codec = IBQCodec.from_checkpoint(checkpoint, device)
        with _phase("codec_run"):
            codes = codec.encode_image(input_path, lat_h=lat_h, lat_w=lat_w)
        if low_memory:
            with _phase("component_release"):
                del codec
                _release_unused_memory(device)

    with _phase("model_load"):
        model, tokenizer = load_gestalt(artifacts.gestalt_dir, device, dtype)
    with _phase("model_run"):
        tokens = model.generate_mmu(
            image_tokens=(codes + _VISUAL_OFFSET).tolist(),
            token_h=lat_h,
            token_w=lat_w,
            system_tokens=system,
            question_tokens=user,
            gen_length=_ANSWER_LENGTH,
            steps=_ANSWER_STEPS,
            block_length=_BLOCK_LENGTH,
            temperature=0.0,
            remasking="low_confidence",
        )
    return _decode_answer(tokenizer, tokens)


@torch.inference_mode()
def text(
    prompt: str,
    *,
    artifacts: ArtifactPaths,
    device: torch.device,
    dtype: torch.dtype,
    system_prompt: str,
) -> str:
    system, user = _encode_prompts(artifacts.gestalt_dir, system_prompt, prompt)
    condition_length = len(build_text_condition(system, user))
    _check_length(condition_length + 3 + _ANSWER_LENGTH)

    with _phase("model_load"):
        model, tokenizer = load_gestalt(artifacts.gestalt_dir, device, dtype)
    with _phase("model_run"):
        tokens = model.generate_text(
            system_tokens=system,
            user_tokens=user,
            gen_length=_ANSWER_LENGTH,
            steps=_ANSWER_STEPS,
            block_length=_BLOCK_LENGTH,
        )
    return _decode_answer(tokenizer, tokens)

# gestalt-mps

Local inference for [Gestalt](https://github.com/GeWu-Lab/Gestalt) (text-to-image and image understanding with a block-diffusion language model) in plain PyTorch, with Apple Silicon (MPS) as the primary target. CPU and CUDA are selectable explicitly.

> **Status: v0.1 in development.** Everything below describes the planned behavior. No hardware support, speed, memory figure or output quality has been measured yet. See [Support matrix](#support-matrix).

## What it is

- A Python package (`gestalt-mps`, pure Python) that vendors the Gestalt inference code at a pinned revision under the `gestalt` namespace and adds the Apple Silicon adaptation in `gestalt_mps`.
- A CLI, `gestalt`, with five commands: `download`, `doctor`, `generate`, `ask`, `text`.
- A small inference-only IBQ image codec (pixels <-> token IDs), so no training dependencies are required.
- Out of scope for v0.1: training, MLX, quantization.

## Requirements

- Python 3.11
- About 18.4 GB of free disk for the two checkpoints, plus room for the Hub cache and temporary files
- `torch==2.10.0` and `transformers==5.2.0` initially (as upstream). If a newer PyTorch is needed for MPS, the version that passes the tests will be recorded here.

## Installation

The project is not published yet; install from a source checkout.

### macOS (Apple Silicon)

The standard PyTorch wheel from PyPI includes MPS.

```bash
# uv
uv sync            # add --extra dev for tests
uv run gestalt --help

# or pip
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install .
gestalt --help
```

### Linux / Windows

Install the official PyTorch wheel for your backend **first**, then the project. Pick the exact command for your platform on the [PyTorch selector](https://pytorch.org/get-started/previous-versions/) and keep `torch==2.10.0`.

```bash
python3.11 -m venv .venv && source .venv/bin/activate

# CPU-only
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# or CUDA (replace cuXXX with the tag listed for your driver on pytorch.org)
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cuXXX

python -m pip install .
```

With `uv`, install the project into the same environment after the PyTorch step: `uv pip install .`. A plain `uv sync` may resolve a different PyTorch build from PyPI.

## Usage

```bash
gestalt doctor                  # platform, PyTorch, MPS/CUDA, memory, expected files; no network
gestalt download                # fetch pinned weights into the Hugging Face cache
gestalt generate "a red bicycle leaning on a brick wall" -o out.png
gestalt ask photo.jpg "What is in this image?"
gestalt text "Explain block diffusion in two sentences."
```

All inference commands accept `--dir`, `--device auto|mps|cuda|cpu` and `--dtype auto|bf16|fp16|fp32`.

### `download` and `--dir`

`gestalt download [--dir DIR]` downloads the pinned Gestalt weights and the IBQ tokenizer checkpoint, verifies the SHA-256 of both weight files, and prints the resulting paths. Re-running reuses the cache; an incomplete download fails explicitly.

- Without `--dir`, files go to the Hugging Face cache and are read from it in local-only mode.
- With `--dir DIR`, files go to `DIR/gestalt` and `DIR/ibq`. Pass the same `--dir` to `doctor`, `generate`, `ask` and `text`.
- Inference never downloads anything implicitly.

### `generate`

```bash
gestalt generate PROMPT -o OUTPUT.png \
  [--steps 64] [--cfg-scale 4] [--lat-h 32] [--lat-w 32] \
  [--seed N] [--system-prompt TEXT] [--save-tokens tokens.npy] [--low-memory]
```

Writes an RGB PNG of `16 * lat-w` by `16 * lat-h` pixels (512x512 by default). `--save-tokens` also writes the raw image token IDs.

### `ask` and `.npy` token files

```bash
gestalt ask INPUT QUESTION [--lat-h 32] [--lat-w 32] [--low-memory]
```

`INPUT` is an image or a `.npy` file of IBQ token IDs. Decoding uses 128 new tokens, 64 steps, block length 64 and temperature 0. The answer goes to stdout; status and errors go to stderr.

`.npy` format:

- integer dtype, shape `(H*W,)` or `(H, W)` matching `--lat-h`/`--lat-w`
- raw IBQ IDs in `[0, 16384)`, as written by `generate --save-tokens` (flat int32)
- loaded with `allow_pickle=False`; wrong sizes, zero dimensions, ragged arrays and out-of-range or offset IDs are rejected

Images are EXIF-rotated, converted to RGB, resized to cover the target with bicubic interpolation, and center-cropped.

### `text` (experimental)

```bash
gestalt text PROMPT [--system-prompt TEXT]
```

Text-only block-diffusion generation (128 tokens, 64 steps, block length 64, temperature 0). The prompt template is an adaptation choice built from the text-only branch of the upstream Stage III data code, not a documented public interface. It does not load the IBQ codec. Treat it as a smoke-level feature.

## Memory and failure notes

- Weights are about 17.5 GB (Gestalt) and 0.9 GB (IBQ). Default mode keeps both components loaded until the command returns.
- `--low-memory` (`generate`, `ask`) releases the previous stage before loading the next and keeps only IDs/text on CPU in between. It preserves dtype, grid and steps. The gain depends on which stage sets the peak.
- Sequences longer than 4,096 positions (markers, grid and answer included) are rejected before the forward pass.
- Default dtype: BF16 on MPS and CUDA, FP32 on CPU. FP16/FP32 on MPS are explicit fallbacks only.
- An explicitly requested backend that is unavailable is an error. There is no silent fallback to another device. Operator failures, load errors and out-of-memory conditions stop with a diagnostic. An out-of-memory configuration should be treated as unsupported.
- Unified memory on Macs is shared with the system; `doctor` reports physical memory, the recommended MPS limit and, when available, memory pressure.
- Macs with 24-32 GiB have not been tested.
- The released Gestalt weights are the **Continual Pretraining** checkpoint. SFT results from the paper should not be attributed to it.

## Support matrix

| Backend | Hardware / OS | PyTorch | dtype | Level of evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| MPS | Apple M5 Pro, 64 GiB, macOS 27.2 | 2.10.0 | BF16 Gestalt, FP32 IBQ | Smoke and default profiles passed | `generate` at 512×512 and 64 steps, `ask` from pixels and tokens, and experimental `text` |
| MPS | 24–32 GiB Macs | — | — | Not measured | |
| CPU | Apple M5 Pro, macOS 27.2 | 2.10.0 | FP32 | Full-checkpoint short forward and IBQ parity | Full CPU CLI profile remains unmeasured |
| CUDA | — | — | — | No hardware run | Device selection is covered by tests |

On the tested Mac, the first 512×512 generation took 400 seconds and reached a sampled MPS driver peak of 19.98 GiB. The highest driver peak in the default profile was 22.49 GiB during `ask`. See the [validation report](docs/validation/2026-11-20-apple-silicon.md) for timing, reproducibility and low-memory measurements.

## Tests

```bash
uv run --extra dev pytest -q
# or: python -m pip install ".[dev]" && python -m pytest -q
```

The suite runs without checkpoints, using synthetic weights and a fake Hub client. Real-weight checks are performed by `scripts/validate_release.py` on the target machine.

## License, provenance and citations

This project is licensed under Apache 2.0 (see `LICENSE`). Third-party code, pinned revisions and change notices are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md). Adapted files carry change notices.

| Component | Pinned revision |
| --- | --- |
| [Gestalt code](https://github.com/GeWu-Lab/Gestalt) | `a385cfc26195480a6cd8f1ad4ef6436ce399c8e0` |
| [Gestalt weights and tokenizer](https://huggingface.co/GeWu-Lab/Gestalt) | `42a8b5d94f80352515603a5b4bab9d2742b6ed80` |
| [SEED-Voken (IBQ code)](https://github.com/TencentARC/SEED-Voken) | `c1544efff959ba18d453089071307a7115698551` |
| [IBQ-Tokenizer-16384 weights](https://huggingface.co/TencentARC/IBQ-Tokenizer-16384) | `ef2200997f97118050c5b17dde48e8b78690f866` |

The CLI downloads model weights directly from their publishers. See the model cards for terms.

If you use this work, please cite the original papers:

- Yang et al., [Gestalt](https://arxiv.org/abs/2610.00576)
- Sun et al., [IBQ](https://arxiv.org/abs/2412.02692)

This is an independent adaptation and is not affiliated with the original authors.

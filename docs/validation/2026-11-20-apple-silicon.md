# Gestalt MPS validation — 2026-11-20

The tested application commit is `16615e3`. The host was an Apple M5 Pro with 64 GiB, macOS 27.2, Python 3.11.15 and PyTorch 2.10.0. All MPS inference processes ran with `PYTORCH_ENABLE_MPS_FALLBACK` unset. Runtime inference used the local pinned Hugging Face cache.

## Installation and checkpoints

- The final checkpoint-free suite passed all 169 tests on the Mac, including its CPU and native MPS fixtures.
- `uv sync --locked --python 3.11 --extra dev` and `uv build --python 3.11` succeeded. A clean pip environment installed the built wheel. `pip check`, all five commands in `gestalt --help`, and offline `gestalt doctor` passed.
- Wheel and source distribution contained the Apache license and source provenance, with zero checkpoints or project cache files.
- `gestalt download` completed. The 17,507,601,848-byte Gestalt Safetensors and the 888,207,474-byte IBQ checkpoint matched the sizes and SHA-256 digests in the implementation plan.
- Real IBQ deserialized successfully through `torch.load(weights_only=True, map_location="cpu")` after digest verification. It contained 1,088 state entries, of which 543 EMA tensors were independently matched to the reference inference architecture.

## Checkpoint and backend parity

- The same 24-position full-checkpoint input produced finite logits in CPU FP32 and MPS BF16. Top-1 matched in 8/8 masked positions. Mean absolute logit difference was 0.0338.
- The official IBQ reference primitives and this codec matched every encoded ID on the same input tensors at 16×16, 32×32 and 16×24. CPU decode matched pixel-for-pixel. MPS decode differed by at most one level per channel using fixed IDs.
- The comparison used the independently reconstructed reference EMA parameters from the pinned checkpoint.
- The codec fixture PNG SHA-256 was 15978d6110d9d36f07aaeec5c6ead4f5f7a854ce1b3233a127b7aec1f15735cb.

## Installed CLI on MPS BF16

The smoke profile result was **pass**.

| Step | Result | Wall seconds | Peak process RSS (GiB) | Sampled peak MPS driver (GiB) |
| --- | --- | ---: | ---: | ---: |
| doctor | passed | 3.62 | 0.30 | 0.00 |
| generate-smoke | passed | 36.99 | 1.50 | 18.28 |
| ask-smoke | passed | 112.62 | 0.45 | 18.79 |
| text-smoke | passed | 58.03 | 0.45 | 18.28 |

The default 512×512 profile passed all 10 commands and four comparisons in 39 min 0 s.

| Step | Result | Wall seconds | Peak process RSS (GiB) | Sampled peak MPS driver (GiB) |
| --- | --- | ---: | ---: | ---: |
| doctor | passed | 4.43 | 0.30 | 0.00 |
| generate-normal-a | passed | 400.35 | 1.78 | 19.98 |
| generate-normal-b | passed | 386.64 | 2.00 | 19.98 |
| generate-low-memory | passed | 409.60 | 1.94 | 19.91 |
| ask-red-square | passed | 233.90 | 1.64 | 22.49 |
| ask-blue-stripes | passed | 252.03 | 1.64 | 22.49 |
| ask-generated-tokens | passed | 250.60 | 0.45 | 21.70 |
| ask-red-square-low-memory | passed | 276.30 | 1.64 | 21.70 |
| text-hello | passed | 67.33 | 0.45 | 18.28 |
| text-capital | passed | 58.68 | 0.45 | 18.28 |

Metrics are measured in each subprocess. Wall time includes start-up and weight loading. RSS and driver allocations are separate measures that can overlap on unified memory.

### Reproducibility and observed output

- The two normal runs produced byte-identical raw IDs with the same seed.
- The two normal runs produced byte-identical RGB PNGs with the same seed.
- Low-memory generation produced the same raw IDs. The PNG also matched byte for byte.
- Normal and low-memory image input returned the same text byte for byte.
- An offline run through `--dir` passed the red-square image's own 32×32 raw IDs to `ask`. Its stdout matched the pixel-input run byte for byte.
- ask-red-square advisory check passed. any of ['red']
- ask-blue-stripes advisory check passed. any of ['blue']
- ask-red-square-low-memory advisory check passed. any of ['red']
- text-capital advisory check passed. any of ['Paris']
- The inspected normal generation showed a red ring centered on a white background. PNG SHA-256 `0eab49c94380f9a1d56e44409a7714774ffc70526fb20b5b91f0279373b88cb3`. NPY SHA-256 `aec8f129ebdea2472c5ede4247bbb9428dbeb227f530f9186de68b0ab952253d`.

During image decoding, the sampled MPS driver peak dropped from 19.98 GiB to 2.18 GiB in low-memory mode. The overall generate peak moved from 19.98 GiB to 19.91 GiB because the Gestalt forward dominated it. The red-square ask peak moved from 22.49 GiB to 21.70 GiB.

The release validator is reproducible with the commands below.

```sh
uv sync --locked --python 3.11 --extra dev
uv run gestalt download
env -u PYTORCH_ENABLE_MPS_FALLBACK HF_HUB_OFFLINE=1 uv run python scripts/validate_release.py --profile default --device mps --dtype bf16
```

## Scope of the evidence

The tested configuration is the 64 GiB M5 Pro in the report. Driver allocations do not establish the minimum system RAM. Macs with 24–32 GiB, CUDA hardware and a full CPU CLI profile remain unmeasured. Text-only is experimental. The two prompts exercise generation and provide a small sanity check with the published Continual Pretraining checkpoint. Broader language and image quality were outside the test.

## Upstream candidates

Both local branches start at GeWu-Lab/Gestalt `a385cfc26195480a6cd8f1ad4ef6436ce399c8e0`. The live upstream `main` was checked again after the runtime tests.

| Local branch | Candidate | Verified |
| --- | --- | --- |
| `upstream/mmu-mps` | `041b69b646ab662abc8557b59d2b551bb32dc97d` | Seven checkpoint-free regression cases passed on CPU and MPS. The original script ran with the released checkpoint, MPS FP32, fallback unset, a 16×16 NPY grid, four steps and 16 answer tokens. It returned an answer in 492.65 seconds including loading. |
| `upstream/inference-install` | `f1b159f5db5c64d73cb34d84cb8d14df22ef55f1` | A clean arm64 Python 3.11 pip install, `pip check`, both inference `--help` commands and `hf download --help` passed. FlashAttention, DeepSpeed, Accelerate and PyArrow were absent. Training requirements and scripts were preserved. |

### Draft PR · Fix MMU sampling on MPS

The default MMU confidence calculation and positive-temperature sampling convert logits to FP64, which MPS cannot represent. Use FP32 for MPS confidence and a stable log-space ranking for Gumbel sampling. Preserve the denominator floor, the representable uniform range, and the current CPU and CUDA calculation.

Validation covered zero and positive temperatures, fixed-uniform reference rankings and the upper range of FP32 uniforms. On the M5 Pro, six MPS cases reproduced the FP64 failure on the base and all seven CPU/MPS regression cases passed with the patch. The original `run_mmu.py` produced an answer on the public checkpoint with fallback unset. CUDA was outside this hardware test.

### Draft PR · Document a separate inference installation

Add `requirements-inference.txt` for MMU and T2I, document the Python 3.11 environment and use the current `hf download` command. Stage III training continues to use `requirements.txt`.

A clean macOS arm64 pip installation accepted the six pins. Both inference help commands, `hf download --help`, and `pip check` passed. A tiny CPU Safetensors checkpoint also loaded with the expected state dict in the reference check. CUDA and training were outside this verification.

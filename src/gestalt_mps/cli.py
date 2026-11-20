"""Command line interface: download, doctor, generate, ask and text."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from gestalt_mps import artifacts, device, pipeline

MAX_SEQUENCE = 4096
ASK_ANSWER_TOKENS = 128
DEVICES = ("auto", "mps", "cuda", "cpu")
DTYPES = ("auto", "bf16", "fp16", "fp32")


class _Usage(Exception):
    """Invalid argument detected after parsing (exit code 2)."""


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _seed(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return number


def _cfg_scale(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a finite number, got {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"expected a finite number >= 0, got {value!r}")
    return number


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"error: {message}\n")


def _add_runtime(p: argparse.ArgumentParser, *, low_memory: bool) -> None:
    p.add_argument("--dir", type=Path, default=None, help="artifact directory (default: Hub cache)")
    p.add_argument("--device", choices=DEVICES, default="auto")
    p.add_argument("--dtype", choices=DTYPES, default="auto")
    if low_memory:
        p.add_argument("--low-memory", action="store_true",
                       help="release each model before loading the next stage")


def _add_grid(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lat-h", type=_positive_int, default=32, help="latent grid height (default 32)")
    p.add_argument("--lat-w", type=_positive_int, default=32, help="latent grid width (default 32)")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="gestalt",
        description="Gestalt inference on Apple Silicon (MPS). "
        "The `text` command is experimental.",
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    p = sub.add_parser("download", help="download and verify pinned weights")
    p.add_argument("--dir", type=Path, default=None)

    p = sub.add_parser("doctor", help="report platform, device and artifact status")
    p.add_argument("--dir", type=Path, default=None)

    p = sub.add_parser("generate", help="generate a PNG from a prompt")
    p.add_argument("prompt")
    p.add_argument("-o", "--output", type=Path, required=True, metavar="OUTPUT.png")
    p.add_argument("--steps", type=_positive_int, default=64)
    p.add_argument("--cfg-scale", type=_cfg_scale, default=4.0)
    _add_grid(p)
    p.add_argument("--seed", type=_seed, default=None)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--save-tokens", type=Path, default=None, metavar="PATH.npy")
    _add_runtime(p, low_memory=True)

    p = sub.add_parser("ask", help="answer a question about an image or .npy token file")
    p.add_argument("input", type=Path, help="image or .npy token file")
    p.add_argument("question")
    _add_grid(p)
    _add_runtime(p, low_memory=True)

    p = sub.add_parser(
        "text", help="EXPERIMENTAL: block-diffusion text reply",
        description="EXPERIMENTAL: generate a text reply by block diffusion.")
    p.add_argument("prompt")
    p.add_argument("--system-prompt", default="")
    _add_runtime(p, low_memory=False)
    return parser


def _is_npy(path: Path) -> bool:
    return path.suffix.lower() == ".npy"


def _check_nonblank(value: str, name: str) -> None:
    if not value.strip():
        raise _Usage(f"{name} must not be empty")


def _validate(args: argparse.Namespace) -> None:
    """Checks that do not need the tokenizer. The pipeline re-checks exactly."""
    if args.command == "generate":
        _check_nonblank(args.prompt, "prompt")
        if args.output.suffix.lower() != ".png":
            raise _Usage("output must be a .png file")
        if args.save_tokens is not None and not _is_npy(args.save_tokens):
            raise _Usage("--save-tokens must be a .npy file")
        if args.lat_h * args.lat_w > MAX_SEQUENCE:
            raise _Usage(
                f"grid {args.lat_h}x{args.lat_w} needs more than {MAX_SEQUENCE} positions")
    elif args.command == "ask":
        _check_nonblank(args.question, "question")
        if args.lat_h * args.lat_w + ASK_ANSWER_TOKENS > MAX_SEQUENCE:
            raise _Usage(
                f"grid {args.lat_h}x{args.lat_w} plus {ASK_ANSWER_TOKENS} answer tokens "
                f"exceeds {MAX_SEQUENCE} positions")
    elif args.command == "text":
        _check_nonblank(args.prompt, "prompt")


def _runtime(args: argparse.Namespace, *, require_ibq: bool):
    dev = device.resolve_device(args.device)
    dtype = device.resolve_dtype(args.dtype, dev)
    arts = artifacts.resolve_artifacts(args.dir, require_ibq=require_ibq)
    return dev, dtype, arts


def _run(args: argparse.Namespace) -> int:
    cmd = args.command
    if cmd == "download":
        paths = artifacts.download_artifacts(args.dir)
        print(f"gestalt: {paths.gestalt_dir}")
        print(f"ibq: {paths.ibq_checkpoint}")
        return 0
    if cmd == "doctor":
        for key, value in device.device_status().items():
            print(f"{key}: {value}")
        for name, present in artifacts.inspect_artifacts(args.dir).items():
            print(f"{name}: {'present' if present else 'missing'}")
        return 0
    if cmd == "generate":
        dev, dtype, arts = _runtime(args, require_ibq=True)
        pipeline.generate(
            args.prompt, args.output, artifacts=arts, device=dev, dtype=dtype,
            steps=args.steps, cfg_scale=args.cfg_scale, lat_h=args.lat_h,
            lat_w=args.lat_w, seed=args.seed, system_prompt=args.system_prompt,
            save_tokens=args.save_tokens, low_memory=args.low_memory)
        print(f"wrote {args.output}", file=sys.stderr)
        return 0
    if cmd == "ask":
        if not args.input.is_file():
            raise FileNotFoundError(f"input not found: {args.input}")
        dev, dtype, arts = _runtime(args, require_ibq=not _is_npy(args.input))
        answer = pipeline.ask(
            args.input, args.question, artifacts=arts, device=dev, dtype=dtype,
            lat_h=args.lat_h, lat_w=args.lat_w, low_memory=args.low_memory)
        print(answer)
        return 0
    # text
    dev, dtype, arts = _runtime(args, require_ibq=False)
    reply = pipeline.text(
        args.prompt, artifacts=arts, device=dev, dtype=dtype,
        system_prompt=args.system_prompt)
    print(reply)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse: help (0) or usage error (2)
        return exc.code if isinstance(exc.code, int) else 2
    try:
        _validate(args)
    except _Usage as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - report any runtime failure on stderr
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Load the pinned Gestalt architecture from a local inference snapshot."""

import json
from itertools import chain
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from transformers import PreTrainedTokenizerBase

from gestalt.model.builder import load_tokenizer, validate_model_vocab
from gestalt.model.config import SPECIAL_TOKENS, VOCAB_CONFIG
from gestalt.model.modeling_gestalt import GestaltConfig, GestaltModelLM


def _read_config(path: Path) -> GestaltConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("model_type") != GestaltConfig.model_type:
        raise ValueError("config.json must describe a Gestalt model.")
    expected = {
        "bos_token_id": SPECIAL_TOKENS.BOS,
        "eos_token_id": SPECIAL_TOKENS.EOS,
        "mask_token_id": SPECIAL_TOKENS.MASK,
        "pad_token_id": SPECIAL_TOKENS.PADDING,
        "vocab_size": VOCAB_CONFIG.EXTENDED_VOCAB_SIZE,
        "embedding_size": VOCAB_CONFIG.EXTENDED_VOCAB_SIZE,
        "extended_vocab_size": VOCAB_CONFIG.EXTENDED_VOCAB_SIZE,
        "rope": True,
        "alibi": False,
        "flash_attention": False,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            raise ValueError(f"config.json {field} must be {value!r}. Found {data.get(field)!r}.")
    return GestaltConfig.from_dict(data)


def _load_weights(
    model: GestaltModelLM, path: Path, device: torch.device, dtype: torch.dtype
) -> None:
    expected = model.state_dict()
    parameters = dict(model.named_parameters())
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            missing = expected.keys() - checkpoint_keys
            unexpected = checkpoint_keys - expected.keys()
            if missing or unexpected:
                details = []
                if missing:
                    preview = ", ".join(sorted(missing)[:5])
                    details.append(f"missing {len(missing)} weight keys ({preview})")
                if unexpected:
                    preview = ", ".join(sorted(unexpected)[:5])
                    details.append(f"unexpected {len(unexpected)} weight keys ({preview})")
                raise ValueError(f"Incompatible {path.name}: {', '.join(details)}")
            for name, target in expected.items():
                checkpoint_shape = tuple(checkpoint.get_slice(name).get_shape())
                if checkpoint_shape != target.shape:
                    raise ValueError(
                        f"Incompatible {path.name}: shape of weight {name} is "
                        f"{checkpoint_shape}; expected {tuple(target.shape)}."
                    )

            # Safetensors maps the source on CPU. Transfer one tensor at a time
            # so a full CPU model is never allocated as a staging copy.
            for name, target in expected.items():
                owner_name, _, tensor_name = name.rpartition(".")
                owner = model.get_submodule(owner_name)
                value = checkpoint.get_tensor(name)
                if value.is_floating_point() != target.is_floating_point():
                    raise ValueError(
                        f"Incompatible {path.name}: dtype of weight {name} is {value.dtype}."
                    )
                if name in parameters:
                    value = value.to(device=device, dtype=dtype)
                    owner.register_parameter(
                        tensor_name,
                        torch.nn.Parameter(value, requires_grad=parameters[name].requires_grad),
                    )
                else:
                    # Restore persistent M-RoPE inverse frequencies at their
                    # compute dtype. The constructor ran on meta and has no data.
                    owner.register_buffer(tensor_name, value.to(device=device, dtype=target.dtype))
    except SafetensorError as exc:
        raise ValueError(f"Could not read Safetensors weights at {path}: {exc}") from exc


def load_gestalt(
    model_dir: Path, device: torch.device, dtype: torch.dtype
) -> tuple[GestaltModelLM, PreTrainedTokenizerBase]:
    """Return an evaluation model and matching tokenizer without contacting the Hub."""
    model_dir = Path(model_dir).expanduser()
    weights = model_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Gestalt weights are missing: {weights}. Run gestalt download.")

    # Construct the known local classes directly. Neither auto_map nor remote
    # Python code participates in loading config, tokenizer or model.
    config = _read_config(model_dir / "config.json")
    tokenizer = load_tokenizer(str(model_dir))
    if tokenizer.convert_tokens_to_ids("<|EOL|>") != SPECIAL_TOKENS.NEWLINE:
        raise ValueError(f"Tokenizer <|EOL|> must have ID {SPECIAL_TOKENS.NEWLINE}.")
    config.init_device = "meta"
    config.dtype = dtype
    with torch.device("meta"):
        model = GestaltModelLM(config)
    validate_model_vocab(model, VOCAB_CONFIG.EXTENDED_VOCAB_SIZE)
    _load_weights(model, weights, device, dtype)
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if tensor.device.type != device.type or (
            device.index is not None and tensor.device.index != device.index
        ):
            raise RuntimeError(f"Gestalt tensor {name} is on {tensor.device}; expected {device}.")
    for name, parameter in model.named_parameters():
        if parameter.dtype != dtype:
            raise RuntimeError(f"Gestalt parameter {name} has dtype {parameter.dtype}; expected {dtype}.")
    return model.eval(), tokenizer

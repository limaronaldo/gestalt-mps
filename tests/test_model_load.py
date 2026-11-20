import json
import shutil
import socket

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from gestalt.model.modeling_gestalt import GestaltConfig, GestaltModelLM
from gestalt_mps.model import load_gestalt


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gestalt-model")
    markers = {
        126080: "<|startoftext|>",
        126081: "<|endoftext|>",
        126084: "<|EOL|>",
        126332: "<system>",
        126333: "</system>",
        126334: "<user>",
        126335: "</user>",
        126336: "<|mdm_mask|>",
        126339: "<padding>",
        126349: "<IMAGE>",
        126350: "</IMAGE>",
        126354: "<answer>",
        126355: "</answer>",
    }
    vocab = {markers.get(index, f"word{index}"): index for index in range(142740)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="word0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(list(markers.values()))
    tokenizer.save(str(directory / "tokenizer.json"))

    config = GestaltConfig(
        vocab_size=142848,
        embedding_size=142848,
        bos_token_id=126080,
        eos_token_id=126081,
        mask_token_id=126336,
        pad_token_id=126339,
        d_model=8,
        n_heads=2,
        n_kv_heads=2,
        n_layers=1,
        inter_layer_num=1,
        block_type="llama",
        layer_norm_type="rms",
        max_sequence_length=16,
        rope=True,
        weight_tying=False,
        init_device="cpu",
        auto_map={
            "AutoConfig": "configuration_llada.LLaDAConfig",
            "AutoModelForCausalLM": "modeling_llada.LLaDAModelLM",
        },
    )
    model = GestaltModelLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.125)
    rotary = model.model.transformer.blocks[0].rotary_emb
    rotary.inv_freq = rotary.inv_freq.to(torch.bfloat16)
    model.save_pretrained(directory)
    # Mirror the pinned snapshot's config without changing how the fixture builds.
    config.init_device = "meta"
    config.save_pretrained(directory)
    return directory


@pytest.mark.parametrize("backend", [
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available() or not torch.backends.mps.is_macos_or_newer(14, 0),
            reason="MPS BF16 requires macOS 14 or newer and an MPS device",
        ),
    ),
])
def test_tiny_checkpoint_load_dtype_and_strict_keys(tiny_checkpoint, monkeypatch, backend):
    def deny_network(*args, **kwargs):
        pytest.fail("Local inference accessed the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    device = torch.device(backend)
    model, tokenizer = load_gestalt(tiny_checkpoint, device, torch.bfloat16)
    assert type(model) is GestaltModelLM
    assert not model.training
    assert model.config.dtype == torch.bfloat16
    assert all(p.device.type == backend and p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(b.device.type == backend for b in model.buffers())
    rotary = model.model.transformer.blocks[0].rotary_emb
    torch.testing.assert_close(rotary.inv_freq.cpu(), torch.tensor([1.0, 0.010009765625]))
    assert model.get_input_embeddings().weight[126080, 0].item() == 0.125
    assert tokenizer.encode("word42", add_special_tokens=False) == [42]
    assert tokenizer.decode([126080, 42, 126081], skip_special_tokens=True) == "word42"
    ids = torch.tensor([[42, 43]], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, document_ids=torch.zeros_like(ids))[0].logits
    assert logits.shape == (1, 2, 142848)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    ("damage", "diagnosis"),
    [("missing", "missing"), ("extra", "unexpected"), ("shape", "shape")],
)
def test_rejects_damaged_checkpoint(tiny_checkpoint, tmp_path, damage, diagnosis):
    shutil.copytree(tiny_checkpoint, tmp_path, dirs_exist_ok=True)
    weight_file = tmp_path / "model.safetensors"
    weights = load_file(weight_file)
    name = "model.transformer.wte.weight"
    if damage == "missing":
        del weights[name]
    elif damage == "extra":
        weights["extra.weight"] = torch.zeros(1)
    else:
        weights[name] = torch.zeros(1)
    replacement = tmp_path / "replacement.safetensors"
    save_file(weights, replacement)
    replacement.replace(weight_file)
    with pytest.raises(ValueError, match=rf"(?i){diagnosis}.*weight"):
        load_gestalt(tmp_path, torch.device("cpu"), torch.float32)


def test_rejects_noncanonical_config_ids(tiny_checkpoint, tmp_path):
    shutil.copytree(tiny_checkpoint, tmp_path, dirs_exist_ok=True)
    config = tmp_path / "config.json"
    data = json.loads(config.read_text())
    data["mask_token_id"] = 0
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mask_token_id"):
        load_gestalt(tmp_path, torch.device("cpu"), torch.float32)


def test_requires_safetensors_file(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"invalid")
    with pytest.raises(FileNotFoundError, match=r"model\.safetensors"):
        load_gestalt(tmp_path, torch.device("cpu"), torch.float32)


def test_rejects_wrong_newline_token(tiny_checkpoint, tmp_path):
    shutil.copytree(tiny_checkpoint, tmp_path, dirs_exist_ok=True)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(tokenizer.read_text().replace("<|EOL|>", "invalid-newline"))
    with pytest.raises(ValueError, match="EOL"):
        load_gestalt(tmp_path, torch.device("cpu"), torch.float32)

import os
import subprocess
import sys

import pytest
import torch

from gestalt.model.config import SPECIAL_TOKENS as S
from gestalt.model.configuration_llada import ModelConfig, ActivationType, BlockType
from gestalt.model.flex_attention import (
    _make_document_mask_mod, _make_isolated_mask_mod, build_dense_attn_mask,
)
from gestalt.model.modeling_llada import BufferCache, InteractionStageOneLlamaBlock, LLaDAModel
from gestalt.model.mrope import compute_2d_position_ids
import gestalt.model.flex_attention as flex_module


def test_import_does_not_compile_flex_attention():
    code = '''import torch

def fail(*args, **kwargs):
    raise AssertionError("compiled during inference import")
torch.compile = fail
import gestalt.model.modeling_llada
from gestalt.model.flex_attention import flex_attention, create_block_mask, _compiled_flex_attn
assert callable(flex_attention) and callable(create_block_mask) and callable(_compiled_flex_attn)
'''
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())


def test_flex_wrapper_compiles_on_first_use_and_preserves_arguments(monkeypatch):
    calls = []

    def kernel(q, k, v, **kwargs):
        return q + k + v + kwargs['block_mask']

    def compile_kernel(fn):
        calls.append(fn)
        return fn

    monkeypatch.setattr(flex_module, 'flex_attention', kernel)
    monkeypatch.setattr(flex_module, '_flex_attn_impl', None)
    monkeypatch.setattr(torch, 'compile', compile_kernel)
    assert flex_module._compiled_flex_attn(1, 2, 3, block_mask=4) == 10
    assert flex_module._compiled_flex_attn(2, 3, 4, block_mask=5) == 14
    assert calls == [kernel]


def test_zone_i_and_ii_masks():
    ids = torch.arange(7).unsqueeze(0)
    doc = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
    img = torch.tensor([[False, True, False, False, False, True, False]])
    inter = torch.tensor([[False, False, True, False, False, False, False]])
    pad = torch.tensor([[False, False, False, True, False, False, True]])
    expected_full = torch.tensor([
        [1,1,1,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0], [0,0,0,0,1,1,0], [0,0,0,0,1,1,0], [0,0,0,0,0,0,0],
    ], dtype=torch.bool)
    expected_iso = torch.tensor([
        [1,0,1,0,0,0,0], [0,1,1,0,0,0,0], [1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0], [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,0],
    ], dtype=torch.bool)
    for isolated, expected, mod in (
        (False, expected_full, _make_document_mask_mod(doc, pad)),
        (True, expected_iso, _make_isolated_mask_mod(doc, pad, img, inter)),
    ):
        mask = build_dense_attn_mask(ids, doc, img, pad, inter, isolated=isolated)[0, 0]
        assert torch.equal(mask, torch.where(expected, 0.0, -1e4))
        for q in range(7):
            for k in range(7):
                assert bool(mod(0, 0, q, k)) == bool(expected[q, k])


def test_model_routes_isolated_and_full_masks_to_the_right_layers():
    config = ModelConfig(d_model=16, n_heads=2, n_layers=2, mlp_hidden_size=32,
                         activation_type=ActivationType.silu, block_type=BlockType.llama,
                         rope=True, vocab_size=32, embedding_size=128, init_device='cpu',
                         attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0)
    config.inter_layer_num = 1
    model = LLaDAModel(config).eval()
    ids = torch.tensor([[8, S.IMAGE_START, S.VISUAL_TOKEN_OFFSET, S.IMAGE_END,
                         S.INTERACTION_TOKEN_START, S.PADDING]])
    seen = []
    handles = [b.register_forward_pre_hook(lambda _m, _a, kw: seen.append(kw['attention_bias']), with_kwargs=True)
               for b in model.transformer.blocks]
    try:
        with torch.no_grad():
            model(ids, input_embeddings=torch.randn(1, 6, 16) * 0.1,
                  document_ids=torch.zeros_like(ids))
    finally:
        for handle in handles:
            handle.remove()
    assert seen[0][0, 0, 0, 2] == -1e4
    assert seen[1][0, 0, 0, 2] == 0
    assert seen[0][0, 0, 0, 4] == 0
    assert torch.all(seen[0][0, 0, 5] == -1e4)


def test_rectangular_mrope_and_document_restart():
    ids = torch.tensor([[10, S.IMAGE_START, *range(S.VISUAL_TOKEN_OFFSET, S.VISUAL_TOKEN_OFFSET + 6), S.IMAGE_END, 11, 12]])
    docs = torch.tensor([[0,0,0,0,0,0,0,0,0,0,1]])
    pos = compute_2d_position_ids(ids, image_grids=[(2, 3)], document_ids=docs)
    assert pos.tolist() == [[[0,1,2,2,2,3,3,3,4,5,0]], [[0,1,2,3,4,2,3,4,4,5,0]]]


def _block(dtype):
    torch.manual_seed(27)
    config = ModelConfig(d_model=16, n_heads=2, n_layers=1, mlp_hidden_size=32,
                         activation_type=ActivationType.silu, rope=True,
                         attention_dropout=0.0, residual_dropout=0.0, init_device='cpu')
    return InteractionStageOneLlamaBlock(0, config, BufferCache()).eval().to(dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_real_block_padding_and_cpu_mps_parity(dtype):
    if not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    assert 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ
    ids = torch.tensor([[8, S.IMAGE_START, S.VISUAL_TOKEN_OFFSET, S.IMAGE_END, S.INTERACTION_TOKEN_START, 9, S.PADDING]])
    doc = torch.zeros_like(ids)
    img = torch.tensor([[False, True, True, True, False, False, False]])
    inter = torch.tensor([[False, False, False, False, True, False, False]])
    pad = ids == S.PADDING
    mask = build_dense_attn_mask(ids, doc, img, pad, inter, isolated=True)
    positions = torch.arange(ids.shape[1]).expand_as(ids).repeat(2, 1, 1)
    generator = torch.Generator().manual_seed(5)
    x = (torch.randn((1, 7, 16), generator=generator) * 0.15).to(dtype)
    cpu = _block(dtype)
    mps = _block(dtype).to('mps')
    atol, rtol = ((1e-4, 1e-3) if dtype == torch.float32 else (2e-2, 2e-2))
    with torch.no_grad():
        a, _ = cpu(x, input_ids=ids, attention_bias=mask, position_ids=positions)
        b, _ = mps(x.to('mps'), input_ids=ids.to('mps'), attention_bias=mask.to('mps'), position_ids=positions.to('mps'))
        changed = x.clone()
        changed[:, -1] += 20
        cpu_changed, _ = cpu(changed, input_ids=ids, attention_bias=mask, position_ids=positions)
        c, _ = mps(changed.to('mps'), input_ids=ids.to('mps'), attention_bias=mask.to('mps'), position_ids=positions.to('mps'))
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a.float(), b.cpu().float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(a[:, :-1].float(), cpu_changed[:, :-1].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(b[:, :-1].float(), c[:, :-1].float(), atol=atol, rtol=rtol)


def test_interaction_ffn_routing():
    block = _block(torch.float32)
    ids = torch.tensor([[7, S.IMAGE_START, S.VISUAL_TOKEN_OFFSET, S.IMAGE_END, S.INTERACTION_TOKEN_START]])
    x = torch.randn(1, 5, 16)
    pos = torch.arange(5).expand_as(ids).repeat(2, 1, 1)
    with torch.no_grad():
        block.attn_out.weight.zero_()
        if block.attn_out.bias is not None:
            block.attn_out.bias.zero_()
        norm = block.ff_norm(x)
        text = block.ff_out(block.act(block.ff_proj(norm)) * block.up_proj(norm))
        vision = block.vision_ff_out(block.act(block.vision_ff_proj(norm)) * block.vision_up_proj(norm))
        result, _ = block(x, input_ids=ids, position_ids=pos)
    torch.testing.assert_close(result[:, 0], x[:, 0] + text[:, 0])
    torch.testing.assert_close(result[:, 1:4], x[:, 1:4] + vision[:, 1:4])
    torch.testing.assert_close(result[:, 4], x[:, 4] + (text[:, 4] + vision[:, 4]) / 2)

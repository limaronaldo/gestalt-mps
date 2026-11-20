"""Regression tests for MMU sampling; no checkpoint or package runtime required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gestalt.model.config import SPECIAL_TOKENS as S
from gestalt.model.modeling_gestalt import GestaltModelLM


class FakeLogits(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.transformer = SimpleNamespace(wte=nn.Embedding(1, 1, device=device))
        self.inputs = []

    def forward(self, *, input_ids, position_ids, document_ids):
        self.inputs.append(input_ids.detach().cpu().clone())
        logits = torch.full((*input_ids.shape, 12), -10.0, device=input_ids.device)
        # The first answer token is more confident than the second.
        answer = (input_ids[0] == S.ANSWER_START).nonzero()[0].item() + 2
        logits[0, answer, 7] = 8.0
        logits[0, answer + 1, 8] = 2.0
        logits[0, answer + 1, 9] = 1.0
        return SimpleNamespace(logits=logits)


def fake_model(device):
    model = object.__new__(GestaltModelLM)
    nn.Module.__init__(model)
    model.model = FakeLogits(device)
    return model


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mmu_low_confidence_at_zero_temperature_on_mps():
    model = fake_model("mps")
    result = model.generate_mmu([S.VISUAL_TOKEN_OFFSET], 1, 1, gen_length=2, steps=2, block_length=2)
    assert result == [7, 8]
    first, second = model.model.inputs
    positions = (first[0] == S.MASK).nonzero().flatten().tolist()
    assert second[0, positions].tolist() == [7, S.MASK]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mmu_gumbel_matches_fixed_uniform_fp64_oracle_on_mps(monkeypatch):
    logits = torch.tensor([[1000.0, 1000.5, 999.0], [-1000.0, -999.0, -998.0]])
    uniforms = torch.tensor([[0.1, 0.8, 0.3], [0.9, 0.2, 0.5]])
    monkeypatch.setattr(torch, "rand_like", lambda value, **kwargs: uniforms.to(device=value.device, dtype=kwargs.get("dtype", value.dtype)))
    model = fake_model("mps")
    actual = model._add_gumbel_noise(logits.to("mps"), 0.7)
    oracle = logits.double() - 0.7 * torch.log(-torch.log(uniforms.double() + 1e-20))
    assert torch.isfinite(actual).all()
    assert actual.argmax(-1).cpu().tolist() == oracle.argmax(-1).tolist()


def test_mmu_gumbel_cpu_preserves_original_formula(monkeypatch):
    logits = torch.tensor([[1.0, 2.0, -1.0]])
    uniforms = torch.tensor([[0.2, 0.6, 0.9]])
    monkeypatch.setattr(torch, "rand_like", lambda value, **kwargs: uniforms.to(dtype=kwargs.get("dtype", value.dtype)))
    actual = fake_model("cpu")._add_gumbel_noise(logits, 0.7)
    expected = logits.double().exp() / ((-torch.log(uniforms.double() + 1e-20)) ** 0.7 + 1e-20)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mmu_positive_temperature_low_confidence_on_mps(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda value, **kwargs: torch.full_like(value, 0.5))
    model = fake_model("mps")
    result = model.generate_mmu([S.VISUAL_TOKEN_OFFSET], 1, 1, gen_length=2, steps=2, block_length=2, temperature=0.7)
    assert result == [7, 8]
    first, second = model.model.inputs
    positions = (first[0] == S.MASK).nonzero().flatten().tolist()
    assert second[0, positions].tolist() == [7, S.MASK]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
@pytest.mark.parametrize("logits,uniforms,temperature", [
    ([0.5, 0.0], [0.1, 0.9], 1.0),
    ([1.0, 0.0], [0.999, 0.9999], 8.0),
])
def test_mmu_gumbel_matches_reference_ranking_with_noise_and_floor(monkeypatch, logits, uniforms, temperature):
    logits = torch.tensor([logits])
    uniforms = torch.tensor([uniforms])
    monkeypatch.setattr(torch, "rand_like", lambda value, **kwargs: uniforms.to(device=value.device, dtype=kwargs.get("dtype", value.dtype)))
    oracle = logits.double().exp() / ((-torch.log(uniforms.double() + 1e-20)) ** temperature + 1e-20)
    actual = fake_model("mps")._add_gumbel_noise(logits.to("mps"), temperature)
    assert actual.argmax(-1).cpu().tolist() == oracle.argmax(-1).tolist()


class ImageLogits(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.transformer = SimpleNamespace(wte=nn.Embedding(1, 1, device=device))
        self.calls = []

    def forward(self, *, input_ids, position_ids, document_ids):
        self.calls.append((input_ids.clone(), document_ids.clone()))
        conditional = (input_ids == 25).any()
        logits = torch.zeros((*input_ids.shape, S.VISUAL_TOKEN_END), dtype=torch.float16 if input_ids.device.type == "mps" else torch.float32, device=input_ids.device)
        logits[..., S.VISUAL_TOKEN_OFFSET] = 60000 if conditional else -60000
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
@pytest.mark.parametrize("cfg_scale", [0.0, 4.0])
def test_t2i_diffusion_cfg_and_schedule_are_finite(cfg_scale, device):
    model = object.__new__(GestaltModelLM)
    nn.Module.__init__(model)
    model.model = ImageLogits(device)
    original = model._gumbel_max_sample
    counts = []

    def check_sample(logits, temperature):
        assert torch.isfinite(logits).all()
        counts.append(logits.shape[1])
        return original(logits, temperature)

    model._gumbel_max_sample = check_sample
    result = model.generate_t2i([], [25], lat_h=2, lat_w=3, timesteps=3, temperature=0, cfg_scale=cfg_scale)
    assert result.shape == (1, 6)
    assert ((result >= S.VISUAL_TOKEN_OFFSET) & (result < S.VISUAL_TOKEN_END)).all()
    assert (result == S.VISUAL_TOKEN_OFFSET).all()
    assert counts == [6, 5, 2]
    assert len(model.model.calls) == 3 * (2 if cfg_scale else 1)
    assert all(not documents.any() for _, documents in model.model.calls)
    if cfg_scale:
        for (cond, _), (uncond, _) in zip(model.model.calls[::2], model.model.calls[1::2]):
            assert (cond == S.MASK).sum() == (uncond == S.MASK).sum()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_mmu_gumbel_preserves_largest_distinct_float32_uniforms(monkeypatch):
    logits = torch.tensor([[0.0, 0.3]])
    uniforms = torch.tensor([[1 - 2**-24, 1 - 2**-23]])
    monkeypatch.setattr(torch, "rand_like", lambda value, **kwargs: uniforms.to(device=value.device, dtype=kwargs.get("dtype", value.dtype)))
    oracle = logits.double().exp() / (-torch.log(uniforms.double() + 1e-20) + 1e-20)
    actual = fake_model("mps")._add_gumbel_noise(logits.to("mps"), 1.0)
    assert oracle.argmax(-1).tolist() == [0]
    assert actual.argmax(-1).cpu().tolist() == [0]

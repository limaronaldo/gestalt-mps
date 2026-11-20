"""Experimental text-only Stage-III adaptation."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gestalt.model.config import SPECIAL_TOKENS as S
from gestalt.model.modeling_gestalt import GestaltModelLM
from gestalt.model import templates


class TextForward(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.transformer = SimpleNamespace(wte=nn.Embedding(1, 1, device=device))
        self.calls = []

    def forward(self, *, input_ids, position_ids, document_ids):
        self.calls.append((input_ids.clone(), position_ids.clone(), document_ids.clone()))
        start = (input_ids[0] == S.ANSWER_START).nonzero()[0].item() + 2
        logits = torch.full((*input_ids.shape, S.EOS + 1), -20.0, device=input_ids.device)
        logits[0, start, 42] = 10
        logits[0, start + 1, S.EOS] = 9
        logits[0, start + 2:, 43] = 8
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required"))])
def test_text_template_and_sampler_without_ibq(device):
    assert templates.build_text_condition([11], [21, 22]) == [S.SYSTEM_START, 11, S.SYSTEM_END, S.USER_START, S.BOS, 21, 22, S.EOS, S.USER_END]
    model = object.__new__(GestaltModelLM)
    nn.Module.__init__(model)
    model.model = TextForward(device)
    assert model.generate_text(system_tokens=[11], user_tokens=[21, 22], gen_length=3, steps=3, block_length=3) == [42]
    ids, positions, documents = model.model.calls[0]
    assert ids[0].tolist() == [S.SYSTEM_START, 11, S.SYSTEM_END, S.USER_START, S.BOS, 21, 22, S.EOS, S.USER_END, S.ANSWER_START, S.BOS, S.MASK, S.MASK, S.MASK, S.ANSWER_END]
    assert not documents.any()
    expected = torch.arange(ids.shape[1], device=ids.device)
    assert positions.shape == (2, 1, ids.shape[1])
    assert torch.equal(positions[0, 0], expected)
    assert torch.equal(positions[1, 0], expected)


def test_text_rejects_insufficient_steps():
    model = object.__new__(GestaltModelLM)
    nn.Module.__init__(model)
    model.model = TextForward("cpu")
    with pytest.raises(ValueError, match="steps"):
        model.generate_text(system_tokens=[], user_tokens=[], gen_length=3, steps=1, block_length=2)

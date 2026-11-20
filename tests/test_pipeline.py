import weakref

import numpy as np
import pytest
import torch
from PIL import Image

from gestalt.model.templates import DEFAULT_MMU_SYSTEM_PROMPT
from gestalt_mps import pipeline
from gestalt_mps.artifacts import ArtifactPaths


class TinyTokenizer:
    def encode(self, prompt, *, add_special_tokens):
        assert not add_special_tokens
        if prompt == DEFAULT_MMU_SYSTEM_PROMPT:
            return [999]
        return list(map(ord, prompt))

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(map(chr, tokens))


class TinyModel:
    def generate_t2i(self, system_prompt_tokens, user_prompt_tokens, lat_h, lat_w,
                     timesteps, temperature, cfg_scale):
        assert torch.is_inference_mode_enabled()
        assert system_prompt_tokens == [ord("s")]
        assert user_prompt_tokens[0] == ord("p")
        assert (lat_h, lat_w, timesteps, temperature, cfg_scale) == (2, 3, 2, 1.0, 4.0)
        return 126356 + torch.randint(16384, (1, 6))

    def generate_mmu(self, image_tokens, token_h, token_w, system_tokens,
                     question_tokens, gen_length, steps, block_length, temperature,
                     remasking):
        assert torch.is_inference_mode_enabled()
        assert image_tokens == [126356, 126357, 142739, 126358, 126359, 126360]
        assert (token_h, token_w) == (2, 3)
        assert system_tokens == [999]
        assert question_tokens[0] == ord("q")
        assert (gen_length, steps, block_length, temperature) == (128, 64, 64, 0.0)
        assert remasking == "low_confidence"
        return [ord("a")]

    def generate_text(self, *, system_tokens, user_tokens, gen_length, steps, block_length):
        assert torch.is_inference_mode_enabled()
        assert (system_tokens, user_tokens[0]) == ([ord("s")], ord("p"))
        assert (gen_length, steps, block_length) == (128, 64, 64)
        return [ord("t")]


class TinyCodec:
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle

    def encode_image(self, path, *, lat_h, lat_w):
        assert torch.is_inference_mode_enabled()
        assert Image.open(path).size == (48, 32)
        assert (lat_h, lat_w) == (2, 3)
        return np.array([0, 1, 16383, 2, 3, 4], dtype=np.int32)

    def decode_tokens(self, codes, *, lat_h, lat_w):
        assert torch.is_inference_mode_enabled()
        assert type(codes) is np.ndarray
        assert codes.shape == (6,)
        assert codes.dtype == np.int32 and codes.flags.c_contiguous
        assert 0 <= codes.min() <= codes.max() < 16384
        self.lifecycle["decoded"] = codes.copy()
        return Image.new("RGB", (16 * lat_w, 16 * lat_h))


@pytest.fixture
def rig(monkeypatch, tmp_path):
    lifecycle = {"events": []}
    artifacts = ArtifactPaths(tmp_path / "model", tmp_path / "codec.ckpt")
    options = {"artifacts": artifacts, "device": torch.device("cpu"), "dtype": torch.float32}

    def tokenizer_for(directory):
        assert directory == str(artifacts.gestalt_dir)
        lifecycle["events"].append("tokenize")
        return TinyTokenizer()

    def load_model(directory, device, dtype):
        assert (directory, device, dtype) == (artifacts.gestalt_dir, options["device"], options["dtype"])
        model = TinyModel()
        lifecycle["ibq_alive_at_model_load"] = bool(lifecycle.get("ibq") and lifecycle["ibq"]())
        lifecycle["model"] = weakref.ref(model)
        lifecycle["events"].append("load_model")
        return model, TinyTokenizer()

    def load_codec(path, device):
        assert (path, device) == (artifacts.ibq_checkpoint, options["device"])
        codec = TinyCodec(lifecycle)
        lifecycle["model_alive_at_ibq_load"] = bool(lifecycle.get("model") and lifecycle["model"]())
        lifecycle["ibq"] = weakref.ref(codec)
        lifecycle["events"].append("load_codec")
        return codec

    monkeypatch.setattr(pipeline, "load_tokenizer", tokenizer_for, raising=False)
    monkeypatch.setattr(pipeline, "load_gestalt", load_model, raising=False)
    monkeypatch.setattr(pipeline, "IBQCodec", type("CodecFactory", (), {
        "from_checkpoint": staticmethod(load_codec),
    }), raising=False)
    return options, lifecycle


def generate(output, options, *, prompt="p", tokens=None, low_memory=False, seed=91):
    pipeline.generate(prompt, output, **options, steps=2, cfg_scale=4, lat_h=2, lat_w=3,
                      seed=seed, system_prompt="s", save_tokens=tokens, low_memory=low_memory)


@pytest.mark.parametrize("low_memory", [False, True])
def test_generate_saves_raw_ids_and_png(tmp_path, rig, low_memory, capsys):
    options, lifecycle = rig
    output = tmp_path / "output.png"
    tokens = tmp_path / "my-tokens"
    generate(output, options, tokens=tokens, low_memory=low_memory)

    assert lifecycle["model_alive_at_ibq_load"] is not low_memory
    with Image.open(output) as image:
        assert (image.format, image.mode, image.size) == ("PNG", "RGB", (48, 32))
    saved = np.load(tokens, allow_pickle=False)
    assert saved.dtype == np.int32 and saved.shape == (6,)
    np.testing.assert_array_equal(saved, lifecycle["decoded"])
    assert capsys.readouterr().out == ""


def test_generate_repeats_with_same_seed(tmp_path, rig):
    options, _ = rig
    generate(tmp_path / "first.png", options, tokens=tmp_path / "first.npy")
    generate(tmp_path / "second.png", options, tokens=tmp_path / "second.npy", low_memory=True)
    np.testing.assert_array_equal(np.load(tmp_path / "first.npy"), np.load(tmp_path / "second.npy"))


@pytest.mark.parametrize("low_memory", [False, True])
def test_ask_image_and_tokens_match(tmp_path, rig, low_memory):
    options, lifecycle = rig
    image = tmp_path / "input.png"
    Image.new("RGB", (48, 32)).save(image)
    assert pipeline.ask(image, "q", **options, lat_h=2, lat_w=3, low_memory=low_memory) == "a"
    assert lifecycle["ibq_alive_at_model_load"] is not low_memory

    options["artifacts"] = ArtifactPaths(options["artifacts"].gestalt_dir, None)
    lifecycle["events"].clear()
    tokens = tmp_path / "input.npy"
    np.save(tokens, np.array([[0, 1, 16383], [2, 3, 4]], dtype=np.uint16, order="F"))
    assert pipeline.ask(tokens, "q", **options, lat_h=2, lat_w=3, low_memory=low_memory) == "a"
    assert "load_codec" not in lifecycle["events"]


@pytest.mark.parametrize(("value", "reason"), [
    (np.zeros(6, dtype=np.float32), "integer"),
    (np.zeros(6, dtype=np.bool_), "integer"),
    (np.zeros(6, dtype=object), "array"),
    (np.zeros((3, 2), dtype=np.int32), "shape"),
    (np.zeros(7, dtype=np.int32), "shape"),
    (np.array([2**64 - 1] * 6, dtype=np.uint64), "range"),
    (np.array([-1] * 6), "range"),
    (np.array([126356] * 6), "range"),
])
def test_ask_rejects_invalid_npy_before_loading_models(tmp_path, rig, value, reason):
    options, lifecycle = rig
    tokens = tmp_path / "input.npy"
    np.save(tokens, value)
    with pytest.raises(ValueError, match=reason):
        pipeline.ask(tokens, "q", **options, lat_h=2, lat_w=3, low_memory=False)
    assert not lifecycle["events"]


def test_text_does_not_require_ibq(rig):
    options, lifecycle = rig
    options["artifacts"] = ArtifactPaths(options["artifacts"].gestalt_dir, None)
    assert pipeline.text("p", **options, system_prompt="s") == "t"
    assert lifecycle["events"] == ["tokenize", "load_model"]


@pytest.mark.parametrize(("lat_h", "lat_w", "steps", "cfg_scale", "seed", "reason"), [
    (0, 3, 2, 4, 91, "positive"),
    (2, 2048, 2, 4, 91, "4096"),
    (2, 3, 0, 4, 91, "positive"),
    (2, 3, 2, float("nan"), 91, "finite"),
    (2, 3, 2, 4, 2**64, "seed"),
])
def test_generate_rejects_invalid_arguments_before_tokenizer(tmp_path, rig, lat_h, lat_w,
                                                               steps, cfg_scale, seed, reason):
    options, lifecycle = rig
    with pytest.raises(ValueError, match=reason):
        pipeline.generate("p", tmp_path / "out.png", **options, steps=steps, cfg_scale=cfg_scale,
                          lat_h=lat_h, lat_w=lat_w, seed=seed, system_prompt="s",
                          save_tokens=None, low_memory=False)
    assert not lifecycle["events"]


@pytest.mark.parametrize("command", ["generate", "ask", "text"])
def test_sequence_limit_includes_template_and_response(tmp_path, rig, command):
    options, lifecycle = rig
    if command == "generate":
        call = lambda n: generate(tmp_path / "out.png", options, prompt="p" * n)
        # Grid 6, system 1, 32 interplay, 4 framing, 2 text span, 4 answer/image markers.
        limit = 4047
    elif command == "ask":
        tokens = tmp_path / "input.npy"
        np.save(tokens, [0, 1, 16383, 2, 3, 4])
        call = lambda n: pipeline.ask(tokens, "q" * n, **options, lat_h=2, lat_w=3, low_memory=False)
        # Grid 6, system 1, 32 interplay, 4 framing, 2 image, 2 text span, 3 answer, 128 response.
        limit = 3918
    else:
        call = lambda n: pipeline.text("p" * n, **options, system_prompt="s")
        # System 1, 4 framing, 2 text span, 3 answer markers, 128 response.
        limit = 3958

    with pytest.raises(ValueError, match=r"4097.*4096"):
        call(limit + 1)
    assert lifecycle["events"] == ["tokenize"]

    call(limit)
    assert "load_model" in lifecycle["events"]


def test_mps_releases_previous_component_and_clears_unused_cache(tmp_path, rig, monkeypatch):
    options, lifecycle = rig
    options["device"] = torch.device("mps")
    events = lifecycle["events"]

    def synchronize():
        assert lifecycle["model"]() is None
        events.append("sync")

    monkeypatch.setattr(torch.mps, "synchronize", synchronize)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty_cache"))
    pipeline.set_phase_observer(events.append)
    try:
        generate(tmp_path / "out.png", options, low_memory=True, seed=None)
    finally:
        pipeline.set_phase_observer(None)

    assert events == [
        "tokenize", "model_load:start", "load_model", "model_load:end",
        "model_run:start", "model_run:end", "component_release:start", "sync", "empty_cache",
        "component_release:end", "codec_load:start", "load_codec", "codec_load:end",
        "codec_run:start", "codec_run:end",
    ]

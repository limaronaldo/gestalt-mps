"""Synthetic unit tests for the IBQ inference codec. No real weights are used."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from gestalt_mps import ibq_codec
from gestalt_mps._ibq import checkpoint as ibq_checkpoint
from gestalt_mps._ibq.config import OFFICIAL_CONFIG, IBQConfig
from gestalt_mps._ibq.model import IBQModel
from gestalt_mps._ibq.preprocess import preprocess_image
from gestalt_mps._ibq.quantizer import IndexPropagationQuantizer
from gestalt_mps.ibq_codec import IBQCheckpointError, IBQCodec

CPU = torch.device("cpu")

# Tiny architecture: GroupNorm(32) needs ch % 32 == 0; two downsamples -> factor 4.
TINY = IBQConfig(
    ddconfig=dict(
        double_z=False,
        z_channels=8,
        resolution=16,
        in_channels=3,
        out_ch=3,
        ch=32,
        ch_mult=(1, 1, 2),
        num_res_blocks=1,
        attn_resolutions=(4,),
        dropout=0.0,
    ),
    n_embed=32,
    embed_dim=8,
)


def _tiny_model(seed: int = 0) -> IBQModel:
    torch.manual_seed(seed)
    return IBQModel(TINY).eval()


def _ema_key(name: str) -> str:
    return "model_ema." + name.replace(".", "")


def _make_state_dict(model: IBQModel, *, ema_fill: float = 0.5, regular_fill: float = -7.0):
    """Lightning-like state dict: regular weights, EMA shadows, unrelated extras."""
    sd = {}
    for name, p in model.named_parameters():
        sd[name] = torch.full_like(p, regular_fill)
        sd[_ema_key(name)] = torch.full_like(p, ema_fill)
    sd["model_ema.decay"] = torch.tensor(0.999)
    sd["model_ema.num_updates"] = torch.tensor(-1, dtype=torch.int32)
    sd["loss.discriminator.main.0.weight"] = torch.zeros(3, 3)
    return sd


def _save_ckpt(path, state_dict):
    torch.save({"state_dict": state_dict, "epoch": 3}, path)
    return path


@pytest.fixture
def tiny_ckpt(tmp_path):
    model = _tiny_model()
    return _save_ckpt(tmp_path / "tiny.ckpt", _make_state_dict(model))


@pytest.fixture
def verified(monkeypatch):
    """Record verify_ibq_checkpoint calls and accept the file."""
    calls = []
    monkeypatch.setattr(
        ibq_codec.artifacts, "verify_ibq_checkpoint", lambda p: calls.append(p)
    )
    return calls


# --------------------------------------------------------------------------
# Configuration and architecture
# --------------------------------------------------------------------------


def test_official_config_and_state_keys():
    cfg = OFFICIAL_CONFIG
    assert cfg.n_embed == 16384 and cfg.embed_dim == 256
    assert tuple(cfg.ddconfig["ch_mult"]) == (1, 1, 2, 2, 4)
    assert cfg.ddconfig["num_res_blocks"] == 4
    assert cfg.ddconfig["z_channels"] == 256 and cfg.ddconfig["ch"] == 128
    assert tuple(cfg.ddconfig["attn_resolutions"]) == (16,)
    assert cfg.downsample == 16

    with torch.device("meta"):
        model = IBQModel(cfg)
    params = dict(model.named_parameters())
    assert params["quantize.embedding.weight"].shape == (16384, 256)
    assert params["quant_conv.weight"].shape == (256, 256, 1, 1)
    assert params["post_quant_conv.weight"].shape == (256, 256, 1, 1)
    assert params["encoder.conv_in.weight"].shape == (128, 3, 3, 3)
    # Attention only at 16 px latents (config resolution 256 -> level 4).
    assert "encoder.down.4.attn.3.q.weight" in params
    assert "encoder.down.3.attn.0.q.weight" not in params
    assert "decoder.up.4.attn.4.q.weight" in params
    assert "decoder.up.3.attn.0.q.weight" not in params
    assert "encoder.down.0.downsample.conv.weight" in params
    assert "encoder.down.4.downsample.conv.weight" not in params
    assert "decoder.up.0.upsample.conv.weight" not in params
    assert "decoder.up.4.upsample.conv.weight" in params
    assert not model.state_dict().keys() - params.keys(), "model must have no buffers"
    # Only generator tensors: nothing from loss / discriminator / EMA.
    prefixes = {k.split(".")[0] for k in params}
    assert prefixes == {"encoder", "decoder", "quant_conv", "post_quant_conv", "quantize"}


# --------------------------------------------------------------------------
# Quantizer: original inner-product indices
# --------------------------------------------------------------------------


def test_quantizer_uses_inner_product_not_euclidean():
    q = IndexPropagationQuantizer(n_embed=2, embed_dim=2)
    with torch.no_grad():
        q.embedding.weight.copy_(torch.tensor([[1.0, 0.0], [3.0, 0.0]]))
    z = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    # Euclidean nearest neighbour is row 0 (distance 0); IBQ picks max z.e -> row 1.
    assert q.indices(z).tolist() == [1]


def test_ibq_raw_indices_and_rectangular_layout(verified, tmp_path):
    # Identity codebook: logits are one-hot, so the argmax is unambiguous.
    cfg = IBQConfig(ddconfig=TINY.ddconfig, n_embed=8, embed_dim=8)
    torch.manual_seed(0)
    model = IBQModel(cfg).eval()
    with torch.no_grad():
        model.quantize.embedding.weight.copy_(torch.eye(8))
    ids = np.array([[5, 0, 7], [3, 3, 1]], dtype=np.int64)  # lat_h=2, lat_w=3
    lat_h, lat_w = ids.shape
    z = torch.zeros(1, cfg.embed_dim, lat_h, lat_w)
    for i in range(lat_h):
        for j in range(lat_w):
            z[0, ids[i, j], i, j] = 1.0
    model.encoder = _Const(z)
    model.quant_conv = nn.Identity()
    codec = IBQCodec(model, CPU, cfg)

    img = Image.new("RGB", (cfg.downsample * lat_w, cfg.downsample * lat_h))
    path = tmp_path / "rect.png"
    img.save(path)
    codes = codec.encode_image(path, lat_h=lat_h, lat_w=lat_w)

    assert isinstance(codes, np.ndarray)
    assert codes.dtype == np.int32 and codes.flags["C_CONTIGUOUS"]
    assert codes.shape == (lat_h * lat_w,)
    assert codes.tolist() == [5, 0, 7, 3, 3, 1]  # row-major raw IDs, no offset

    # Decoding looks the same raw IDs up in BHWC order, then permutes to BCHW.
    seen = {}
    model.decoder = _Record(seen, out_hw=(cfg.downsample * lat_h, cfg.downsample * lat_w))
    model.post_quant_conv = nn.Identity()
    image = codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w)
    assert image.mode == "RGB"
    assert image.size == (cfg.downsample * lat_w, cfg.downsample * lat_h)
    got = seen["z"]
    assert got.shape == (1, cfg.embed_dim, lat_h, lat_w)
    for i in range(lat_h):
        for j in range(lat_w):
            expected = model.quantize.embedding.weight[ids[i, j]]
            assert torch.equal(got[0, :, i, j], expected)
    # (H, W) shaped input is accepted too and gives the identical image.
    again = codec.decode_tokens(ids.astype(np.int32), lat_h=lat_h, lat_w=lat_w)
    assert list(again.getdata()) == list(image.getdata())


class _Const(nn.Module):
    def __init__(self, z):
        super().__init__()
        self.z = z

    def forward(self, x):
        assert x.shape[0] == 1 and x.shape[1] == 3
        return self.z.to(x.device)


class _Record(nn.Module):
    def __init__(self, store, out_hw):
        super().__init__()
        self.store = store
        self.out_hw = out_hw

    def forward(self, z):
        self.store["z"] = z.detach().clone()
        return torch.zeros(1, 3, *self.out_hw, device=z.device)


# --------------------------------------------------------------------------
# Decoding to RGB pixels
# --------------------------------------------------------------------------


def test_ibq_decode_rgb_pixels(verified):
    lat_h, lat_w = 2, 3
    H, W = TINY.downsample * lat_h, TINY.downsample * lat_w
    out = torch.zeros(1, 3, H, W)
    out[0, 0] = -1.0  # -> 0
    out[0, 1] = 0.0  # -> 127 (reference truncates 127.5)
    out[0, 2] = 1.0  # -> 255
    out[0, :, 0, 0] = torch.tensor([-3.0, 1.5, 0.5])  # clamped: 0, 255, 191
    out[0, :, H - 1, W - 1] = torch.tensor([1.0, -1.0, 0.0])

    class Fixed(nn.Module):
        def forward(self, z):
            return out.to(z.device)

    model = _tiny_model()
    model.decoder = Fixed()
    codec = IBQCodec(model, CPU, TINY)
    codes = np.arange(lat_h * lat_w, dtype=np.int32)
    img = codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w)

    assert img.mode == "RGB"
    assert img.size == (W, H)  # width = f*lat_w, height = f*lat_h
    assert img.getpixel((0, 0)) == (0, 255, 191)
    assert img.getpixel((W - 1, H - 1)) == (255, 0, 127)
    assert img.getpixel((5, 7)) == (0, 127, 255)


@pytest.mark.parametrize(
    "codes, lat_h, lat_w, match",
    [
        (np.array([0, 1, 2], dtype=np.int32), 2, 2, "size"),
        (np.array([0.0, 1.0, 2.0, 3.0]), 2, 2, "integer"),
        (np.array([True, False, True, False]), 2, 2, "integer"),
        (np.array([0, 1, 2, TINY.n_embed], dtype=np.int64), 2, 2, "range"),
        (np.array([0, 1, 2, -1], dtype=np.int64), 2, 2, "range"),
        (np.zeros((2, 3), dtype=np.int32), 2, 2, "shape|size"),
        (np.zeros((2, 2, 1), dtype=np.int32), 2, 2, "shape"),
        (np.zeros(4, dtype=np.int32), 0, 2, "positive"),
        (np.zeros(4, dtype=np.int32), 2, -2, "positive"),
        ([0, 1, 2, 3], 2, 2, "ndarray"),
    ],
)
def test_decode_rejects_invalid_ids_and_grid(verified, codes, lat_h, lat_w, match):
    codec = IBQCodec(_tiny_model(), CPU, TINY)
    with pytest.raises(ValueError, match=match):
        codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w)


def test_encode_rejects_bad_grid(tmp_path):
    codec = IBQCodec(_tiny_model(), CPU, TINY)
    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(p)
    for bad in [(0, 1), (1, 0), (-1, 2)]:
        with pytest.raises(ValueError, match="positive"):
            codec.encode_image(p, lat_h=bad[0], lat_w=bad[1])


def test_tiny_architecture_roundtrip_shapes():
    codec = IBQCodec(_tiny_model(), CPU, TINY)
    lat_h, lat_w = 2, 3
    x = torch.rand(1, 3, 4 * lat_h, 4 * lat_w) * 2 - 1
    codes = codec.encode_tensor(x)
    assert codes.dtype == np.int32 and codes.shape == (6,)
    assert codes.min() >= 0 and codes.max() < TINY.n_embed
    img = codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w)
    assert img.size == (12, 8) and img.mode == "RGB"


# --------------------------------------------------------------------------
# Preprocess
# --------------------------------------------------------------------------


def test_ibq_preprocess_exif_rgb_aspect(tmp_path):
    # Aspect + center crop: 64x32 with vertical stripes red|green|blue -> 32x32 crop.
    src = Image.new("RGB", (64, 32))
    for x in range(64):
        colour = (255, 0, 0) if x < 21 else (0, 255, 0) if x < 43 else (0, 0, 255)
        for y in range(32):
            src.putpixel((x, y), colour)
    p = tmp_path / "stripes.png"
    src.save(p)
    t = preprocess_image(p, lat_h=2, lat_w=2, patch=16)
    assert t.shape == (1, 3, 32, 32) and t.dtype == torch.float32
    assert float(t.min()) >= -1.0 and float(t.max()) <= 1.0
    # scale 1.0 (cover), crop columns 16..48: centre is pure green, edges red / blue.
    assert t[0, :, 16, 16].tolist() == [-1.0, 1.0, -1.0]
    assert t[0, :, 16, 1].tolist() == [1.0, -1.0, -1.0]
    assert t[0, :, 16, 30].tolist() == [-1.0, -1.0, 1.0]
    # Upscale to cover a rectangle: 8x8 source -> 32 (w) x 16 (h) target, bicubic.
    small = Image.new("RGB", (8, 8), (255, 255, 255))
    small.save(tmp_path / "w.png")
    t = preprocess_image(tmp_path / "w.png", lat_h=1, lat_w=2, patch=16)
    assert t.shape == (1, 3, 16, 32)
    assert torch.allclose(t, torch.ones_like(t))

    # EXIF orientation 6: stored 64x32, red top half / blue bottom half
    # -> displayed 32x64, with blue on the left and red on the right.
    img = Image.new("RGB", (64, 32))
    for y in range(32):
        for x in range(64):
            img.putpixel((x, y), (255, 0, 0) if y < 16 else (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6
    jp = tmp_path / "rot.png"
    img.save(jp, exif=exif)
    t = preprocess_image(jp, lat_h=2, lat_w=2, patch=16)
    assert t.shape == (1, 3, 32, 32)
    assert t[0, :, 16, 2].tolist() == [-1.0, -1.0, 1.0]
    assert t[0, :, 16, 29].tolist() == [1.0, -1.0, -1.0]

    # Mode conversion: RGBA and palette / grayscale all become 3 channels.
    for mode in ("RGBA", "L", "P"):
        q = Image.new("RGB", (32, 32), (10, 200, 30)).convert(mode)
        qp = tmp_path / f"m_{mode}.png"
        q.save(qp)
        t = preprocess_image(qp, lat_h=2, lat_w=2, patch=16)
        assert t.shape == (1, 3, 32, 32)


def test_preprocess_rejects_bad_grid(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(p)
    with pytest.raises(ValueError, match="positive"):
        preprocess_image(p, lat_h=0, lat_w=2, patch=16)
    with pytest.raises(ValueError, match="positive"):
        preprocess_image(p, lat_h=True, lat_w=2, patch=16)


# --------------------------------------------------------------------------
# Checkpoint: EMA names, strictness, order of verification, restricted load
# --------------------------------------------------------------------------


def test_ema_key_has_no_dots():
    assert ibq_checkpoint.ema_key("encoder.conv_in.weight") == "model_ema.encoderconv_inweight"
    assert ibq_checkpoint.ema_key("quantize.embedding.weight") == "model_ema.quantizeembeddingweight"


def test_ema_weights_are_applied_not_regular_ones(verified, tiny_ckpt):
    codec = IBQCodec.from_checkpoint(tiny_ckpt, CPU, config=TINY)
    assert verified == [tiny_ckpt]
    for name, p in codec.model.named_parameters():
        assert p.dtype == torch.float32
        assert torch.all(p == 0.5), name
    assert not codec.model.training
    assert all(not p.requires_grad for p in codec.model.parameters())


def test_ibq_missing_required_ema_keys(verified, tmp_path):
    model = _tiny_model()
    sd = _make_state_dict(model)
    victim = "decoder.conv_out.bias"
    del sd[_ema_key(victim)]
    del sd["model_ema.quantizeembeddingweight"]
    path = _save_ckpt(tmp_path / "missing.ckpt", sd)
    with pytest.raises(IBQCheckpointError) as err:
        IBQCodec.from_checkpoint(path, CPU, config=TINY)
    msg = str(err.value)
    assert "model_ema.decoderconv_outbias" in msg
    assert "model_ema.quantizeembeddingweight" in msg
    # Regular (non-EMA) tensors never stand in for missing EMA tensors.
    assert victim in msg or "decoderconv_outbias" in msg


def test_ibq_missing_encoder_key_and_shape_mismatch(verified, tmp_path):
    model = _tiny_model()
    sd = _make_state_dict(model)
    del sd[_ema_key("encoder.conv_in.weight")]
    sd[_ema_key("quant_conv.weight")] = torch.zeros(1, 1)
    path = _save_ckpt(tmp_path / "bad.ckpt", sd)
    with pytest.raises(IBQCheckpointError) as err:
        IBQCodec.from_checkpoint(path, CPU, config=TINY)
    msg = str(err.value)
    assert "model_ema.encoderconv_inweight" in msg
    assert "model_ema.quant_convweight" in msg and "shape" in msg


def test_ibq_state_dict_needs_dict_with_tensors(verified, tmp_path):
    p = tmp_path / "x.ckpt"
    torch.save({"weights": {}}, p)
    with pytest.raises(IBQCheckpointError, match="state_dict"):
        IBQCodec.from_checkpoint(p, CPU, config=TINY)
    sd = _make_state_dict(_tiny_model())
    sd[_ema_key("quant_conv.bias")] = [1.0]
    _save_ckpt(p, sd)
    with pytest.raises(IBQCheckpointError, match="tensor"):
        IBQCodec.from_checkpoint(p, CPU, config=TINY)


def test_ema_excluded_params_use_regular_tensor():
    model = _tiny_model()
    sd = _make_state_dict(model)
    excluded = {"quant_conv.bias"}
    del sd[_ema_key("quant_conv.bias")]  # EMA has no shadow for excluded params
    mapped = ibq_checkpoint.map_ema_state_dict(model, sd, ema_excluded=excluded)
    assert torch.all(mapped["quant_conv.bias"] == -7.0)  # regular tensor
    assert torch.all(mapped["quant_conv.weight"] == 0.5)  # EMA tensor
    assert set(mapped) == {n for n, _ in model.named_parameters()}
    # Not excluded -> same missing key is an error.
    with pytest.raises(IBQCheckpointError):
        ibq_checkpoint.map_ema_state_dict(model, sd)
    # Excluded but the regular tensor is absent too -> error, never silent init.
    del sd["quant_conv.bias"]
    with pytest.raises(IBQCheckpointError, match="quant_conv.bias"):
        ibq_checkpoint.map_ema_state_dict(model, sd, ema_excluded=excluded)


def test_official_ema_exclusion_set_is_documented_empty():
    # LitEma tracks every named parameter with requires_grad; the inference modules have
    # only such parameters, so nothing is excluded for the official architecture.
    assert ibq_checkpoint.EMA_EXCLUDED_PARAMS == frozenset()


def test_verify_runs_before_deserialisation_with_restricted_loader(
    monkeypatch, tiny_ckpt
):
    order = []
    monkeypatch.setattr(
        ibq_codec.artifacts, "verify_ibq_checkpoint", lambda p: order.append(("verify", p))
    )
    real_load = torch.load

    def spy(*args, **kwargs):
        order.append(("load", args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    IBQCodec.from_checkpoint(tiny_ckpt, CPU, config=TINY)
    assert [o[0] for o in order] == ["verify", "load"]
    kwargs = order[1][2]
    assert kwargs["weights_only"] is True and kwargs["map_location"] == "cpu"


def test_bad_hash_blocks_load(monkeypatch, tiny_ckpt):
    artifacts = ibq_codec.artifacts
    official = next(spec for spec in artifacts.ARTIFACT_SPECS if spec.subdir == "ibq")
    fixture_spec = replace(
        official, size=tiny_ckpt.stat().st_size, sha256="0" * 64,
    )
    monkeypatch.setattr(artifacts, "ARTIFACT_SPECS", (fixture_spec,))

    def no_load(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("torch.load called before verification")

    monkeypatch.setattr(torch, "load", no_load)
    with pytest.raises(artifacts.ArtifactError, match="SHA-256"):
        IBQCodec.from_checkpoint(tiny_ckpt, CPU, config=TINY)


def test_valid_hash_allows_restricted_codec_load(monkeypatch, tiny_ckpt):
    artifacts = ibq_codec.artifacts
    official = next(spec for spec in artifacts.ARTIFACT_SPECS if spec.subdir == "ibq")
    contents = tiny_ckpt.read_bytes()
    fixture_spec = replace(official, size=len(contents), sha256=hashlib.sha256(contents).hexdigest())
    monkeypatch.setattr(artifacts, "ARTIFACT_SPECS", (fixture_spec,))
    codec = IBQCodec.from_checkpoint(tiny_ckpt, CPU, config=TINY)
    assert codec.model.quantize.embedding.weight[0, 0].item() == 0.5


class _Unsafe:
    pass


def test_restricted_loader_incompatibility_is_reported(verified, tmp_path):
    p = tmp_path / "unsafe.ckpt"
    torch.save({"state_dict": _make_state_dict(_tiny_model()), "extra": _Unsafe()}, p)
    with pytest.raises(IBQCheckpointError, match="weights_only"):
        IBQCodec.from_checkpoint(p, CPU, config=TINY)


def test_from_checkpoint_is_fp32_on_requested_device(verified, tiny_ckpt):
    codec = IBQCodec.from_checkpoint(tiny_ckpt, CPU, config=TINY)
    assert codec.device == CPU
    assert {p.dtype for p in codec.model.parameters()} == {torch.float32}


# --------------------------------------------------------------------------
# Native MPS (lightweight, skipped when unavailable)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS unavailable",
)
def test_mps_matches_cpu_on_tiny_codec():
    mps = torch.device("mps")
    cpu_model = _tiny_model(seed=1)
    mps_model = IBQModel(TINY)
    mps_model.load_state_dict(cpu_model.state_dict())
    cpu_codec = IBQCodec(cpu_model, CPU, TINY)
    mps_codec = IBQCodec(mps_model, mps, TINY)

    torch.manual_seed(2)
    lat_h, lat_w = 2, 3
    x = torch.rand(1, 3, 4 * lat_h, 4 * lat_w) * 2 - 1
    cpu_ids = cpu_codec.encode_tensor(x)
    mps_ids = mps_codec.encode_tensor(x)
    assert mps_ids.dtype == np.int32 and mps_ids.shape == cpu_ids.shape

    # Compare IDs only where the CPU top-2 score margin exceeds numerical noise.
    with torch.inference_mode():
        z = cpu_model.quant_conv(cpu_model.encoder(x))
        logits = torch.einsum("bdhw,nd->bnhw", z, cpu_model.quantize.embedding.weight)
        top2 = logits.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).reshape(-1).numpy()
    safe = margin > 1e-3
    assert np.array_equal(cpu_ids[safe], mps_ids[safe])

    codes = np.array([3, 1, 4, 1, 5, 9], dtype=np.int32)
    a = np.asarray(cpu_codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w), dtype=np.int16)
    b = np.asarray(mps_codec.decode_tokens(codes, lat_h=lat_h, lat_w=lat_w), dtype=np.int16)
    assert np.abs(a - b).max() <= 2

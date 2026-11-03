"""Load and run the PyTorch stages, one model at a time, tiled where it matters."""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from old_photo_restore.imageops import tile_starts
from old_photo_restore.third_party.codeformer import CodeFormer
from old_photo_restore.third_party.nafnet import NAFNet
from old_photo_restore.third_party.parsenet import ParseNet
from old_photo_restore.third_party.rrdbnet import RRDBNet


def configure_threads(n: int) -> int:
    if n <= 0:
        n = min(8, os.cpu_count() or 1)
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return n


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state(obj):
    if isinstance(obj, dict):
        for key in ("params_ema", "params", "state_dict"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                obj = inner
                break
    if isinstance(obj, dict) and any(str(k).startswith("module.") for k in obj):
        obj = {str(k).replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def _load(module: torch.nn.Module, path: str) -> torch.nn.Module:
    state = extract_state(torch_load(path))
    missing, unexpected = module.load_state_dict(state, strict=False)
    # Buffers such as num_batches_tracked are not always in the checkpoint.
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        preview_m = ", ".join(missing[:8])
        preview_u = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"weight mismatch for {os.path.basename(path)}: "
            f"{len(missing)} missing ({preview_m}), {len(unexpected)} unexpected ({preview_u})"
        )
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def load_nafnet(path: str, runner) -> NAFNet:
    return runner.place(_load(NAFNet(), path))


def load_codeformer(path: str, runner) -> CodeFormer:
    return runner.place(
        _load(
            CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]),
            path,
        )
    )


def load_parser(path: str, runner) -> ParseNet:
    return runner.place(_load(ParseNet(in_size=512, out_size=512, parsing_ch=19), path))


def load_rrdb(path: str, scale: int, runner) -> RRDBNet:
    return runner.place(_load(RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale), path))


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((max(n, 1),), np.float32)
    window = np.hanning(n + 2)[1:-1].astype(np.float32)
    return np.clip(window, 1e-3, None)


def _hwc(pred: torch.Tensor) -> np.ndarray:
    """CHW or NCHW tensor → HWC float32 numpy on CPU."""
    if pred.ndim == 4:
        pred = pred[0]
    return pred.detach().float().cpu().permute(1, 2, 0).numpy()


def _bgr_to_rgb_nchw(bgr) -> torch.Tensor:
    """OpenCV BGR HWC → RGB NCHW. NAFNet, CodeFormer, and Real-ESRGAN are RGB models."""
    rgb = np.ascontiguousarray(np.asarray(bgr)[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).float().unsqueeze(0)


def run_nafnet_tiled(net: NAFNet, bgr_u8: np.ndarray, tile: int, overlap: int, runner) -> np.ndarray:
    """Return a float32 BGR image in 0..255.

    The network sees RGB. The residual it adds is converted back to BGR here,
    so a later colour match against the original does not swap red and blue.
    """
    bgr = bgr_u8.astype(np.float32) / 255.0
    height, width = bgr.shape[:2]
    tile = max(32, min(int(tile), height, width)) if min(height, width) >= 32 else min(height, width)
    if min(height, width) < 16:
        raise RuntimeError("image is too small to deblur")
    acc = np.zeros_like(bgr)
    weight = np.zeros((height, width, 1), np.float32)
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            patch = bgr[y : y + tile, x : x + tile]
            ph, pw = patch.shape[:2]

            def call(dev, patch=patch):
                return net(_bgr_to_rgb_nchw(patch).to(dev)).clamp(0, 1)

            pred = _hwc(runner.attempt(net, call))[:ph, :pw, ::-1]
            window = (_hann(ph)[:, None] * _hann(pw)[None, :])[..., None]
            acc[y : y + ph, x : x + pw] += pred * window
            weight[y : y + ph, x : x + pw] += window
    out = acc / np.maximum(weight, 1e-6)
    return np.clip(out * 255.0, 0, 255)


def run_codeformer(net: CodeFormer, bgr, fidelity: float, runner, adain: bool = True) -> np.ndarray:
    """Return float32 BGR in 0..255. Input and output stay in OpenCV's channel order."""
    bgr_f = np.clip(bgr, 0, 255).astype(np.float32) / 255.0

    def call(dev):
        tensor = (_bgr_to_rgb_nchw(bgr_f) - 0.5) / 0.5
        return net(tensor.to(dev), w=float(fidelity), adain=bool(adain))[0].clamp(-1, 1)

    pred = _hwc(runner.attempt(net, call))
    rgb_out = (pred + 1.0) * 127.5
    return rgb_out[:, :, ::-1]


def run_parser(net: ParseNet, bgr, runner) -> np.ndarray:
    def call(dev):
        tensor = (_bgr_to_rgb_nchw(np.clip(bgr, 0, 255).astype(np.float32) / 255.0) - 0.5) / 0.5
        return net(tensor.to(dev))[0]

    logits = runner.attempt(net, call)
    return logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)


def run_rrdb_tiled(net: RRDBNet, bgr_u8: np.ndarray, tile: int, overlap: int, runner) -> np.ndarray:
    """Return float32 BGR in 0..1 at ``net.scale`` times the input."""
    bgr = bgr_u8.astype(np.float32) / 255.0
    height, width = bgr.shape[:2]
    scale = int(net.scale)
    tile = max(32, min(int(tile), height, width))
    acc = np.zeros((height * scale, width * scale, 3), np.float32)
    weight = np.zeros((height * scale, width * scale, 1), np.float32)
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            patch = bgr[y : y + tile, x : x + tile]
            ph, pw = patch.shape[:2]
            pad_h, pad_w = ph % 2, pw % 2
            fed = patch
            if pad_h or pad_w:
                fed = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

            def call(dev, fed=fed):
                return net(_bgr_to_rgb_nchw(fed).to(dev)).clamp(0, 1)

            pred = _hwc(runner.attempt(net, call))[: ph * scale, : pw * scale, ::-1]
            window = (_hann(ph)[:, None] * _hann(pw)[None, :])
            window = cv2.resize(window, (pw * scale, ph * scale), interpolation=cv2.INTER_LINEAR)[..., None]
            y0, x0 = y * scale, x * scale
            acc[y0 : y0 + ph * scale, x0 : x0 + pw * scale] += pred * window
            weight[y0 : y0 + ph * scale, x0 : x0 + pw * scale] += window
    return acc / np.maximum(weight, 1e-6)

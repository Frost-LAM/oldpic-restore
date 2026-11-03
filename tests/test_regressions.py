"""Regression tests: colour cast, detection threshold, and download mirrors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from old_photo_restore.config import default_config
from old_photo_restore.detect import landmarks_plausible
from old_photo_restore.device import TorchDevice
from old_photo_restore.imageops import transfer_low_frequency
from old_photo_restore.nets import run_codeformer, run_nafnet_tiled, run_rrdb_tiled
from old_photo_restore.weights import Weight, download, rewrite_hf_url


def _warm_bgr(h=32, w=40) -> np.ndarray:
    """R > B, with a little spatial detail so a blur cannot hide a channel swap."""
    image = np.zeros((h, w, 3), np.float32)
    image[..., 0] = 40
    image[..., 1] = 90
    image[..., 2] = 180
    image[::2, :, 2] = 210
    image[:, ::3, 0] = 25
    return np.clip(image, 0, 255).astype(np.uint8)


def _assert_warm(result, source, tol=1.5):
    got = np.asarray(result, np.float32).mean(axis=(0, 1))
    src = np.asarray(source, np.float32)
    if src.max() <= 1.5:
        src = src * 255.0
    want = src.mean(axis=(0, 1))
    assert got[2] > got[0], f"red/blue swapped: BGR means {got}"
    assert np.all(np.abs(got - want) < tol), f"means {got} vs {want}"


class _Identity(torch.nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.scale != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="nearest")
        return x


class _IdentityCodeFormer(torch.nn.Module):
    def forward(self, x, w=0, adain=True):
        return x, None, None


def test_low_frequency_match_keeps_a_warm_cast():
    src = _warm_bgr().astype(np.float32)
    out = transfer_low_frequency(src, src, sigma=4)
    _assert_warm(out, src)


def test_nafnet_identity_does_not_swap_channels():
    src = _warm_bgr()
    out = run_nafnet_tiled(_Identity(), src, 32, 0, TorchDevice("cpu"))
    _assert_warm(out, src)
    matched = transfer_low_frequency(out, src.astype(np.float32), sigma=6)
    _assert_warm(matched, src)


def test_codeformer_identity_does_not_swap_channels():
    src = _warm_bgr()
    out = run_codeformer(_IdentityCodeFormer(), src, 0.9, TorchDevice("cpu"), True)
    _assert_warm(out, src, tol=2.0)
    matched = transfer_low_frequency(out, src.astype(np.float32), sigma=6)
    _assert_warm(matched, src, tol=2.0)


def test_esrgan_identity_stays_bgr_through_the_lanczos_mix():
    src = _warm_bgr()
    esr = run_rrdb_tiled(_Identity(scale=2), src, 32, 0, TorchDevice("cpu"))
    assert esr.shape[:2] == (64, 80)
    lanczos = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1).astype(np.float32)
    mixed = 0.75 * (esr * 255.0) + 0.25 * lanczos
    _assert_warm(mixed, lanczos)


def test_default_detection_threshold_rejects_false_face_scores():
    assert default_config()["detect"]["threshold"] == pytest.approx(0.60)
    # Background objects that resemble faces scored 0.45-0.49. A real face scored about 0.86.
    assert 0.493 < default_config()["detect"]["threshold"] < 0.859


def test_landmark_geometry_accepts_a_face_and_rejects_a_scramble():
    box = np.array([10, 20, 110, 150], np.float32)
    good = np.array([[40, 60], [80, 62], [58, 90], [42, 115], [78, 116]], np.float32)
    assert landmarks_plausible(good, box)
    nose_above_eyes = good.copy()
    nose_above_eyes[2] = [58, 40]
    assert not landmarks_plausible(nose_above_eyes, box)
    collapsed = good.copy()
    collapsed[1] = collapsed[0]
    assert not landmarks_plausible(collapsed, box)


def test_hf_mirror_env(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.delenv("OPR_HF_ENDPOINT", raising=False)
    url = "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth"
    assert rewrite_hf_url(url) == "https://hf-mirror.com/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth"
    assert rewrite_hf_url("https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth").startswith(
        "https://github.com/"
    )
    monkeypatch.setenv("OPR_HF_ENDPOINT", "https://example.test")
    assert rewrite_hf_url(url).startswith("https://example.test/")


def test_download_resumes_and_uses_certifi(tmp_path, monkeypatch):
    partial = tmp_path / "demo.pth.part"
    partial.write_bytes(b"hello")
    captured = {}

    class Response:
        status = 206

        def __init__(self):
            self.headers = {"Content-Length": "4"}
            self._sent = False

        def read(self, _n):
            if not self._sent:
                self._sent = True
                return b"!!!!"
            return b""

        def getcode(self):
            return 206

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout=120, context=None):
        captured["range"] = request.get_header("Range")
        captured["context"] = context
        return Response()

    monkeypatch.setattr("old_photo_restore.weights.urlopen", fake_urlopen)
    download(Weight("demo.pth", "https://example.com/demo.pth", min_bytes=1), tmp_path)
    assert captured["range"] == "bytes=5-"
    assert captured["context"] is not None
    assert (tmp_path / "demo.pth").read_bytes() == b"hello!!!!"

"""Device selection does not need an Apple Silicon machine or a GPU."""

from __future__ import annotations

import os

import pytest

from old_photo_restore.cli import build_parser
from old_photo_restore.config import apply_cli, default_config
from old_photo_restore.device import (
    TorchDevice,
    apply_device_profile,
    enable_mps_fallback,
    mps_error_should_fallback,
    probe_accelerators,
    select_device,
)
from old_photo_restore.onnx_util import providers_for


@pytest.mark.parametrize(
    ("requested", "mps", "cuda", "expect"),
    [
        ("auto", True, True, "mps"),
        ("auto", True, False, "mps"),
        ("auto", False, True, "cuda"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("mps", True, False, "mps"),
        ("cuda", False, True, "cuda"),
        (" MPS ", True, False, "mps"),
    ],
)
def test_select_device(requested, mps, cuda, expect):
    name, _reason = select_device(requested, mps=mps, cuda=cuda)
    assert name == expect


def test_missing_accelerator_falls_back_to_cpu():
    name, reason = select_device("mps", mps=False, cuda=True)
    assert name == "cpu"
    assert "mps" in reason
    name, reason = select_device("cuda", mps=True, cuda=False)
    assert name == "cpu"
    assert "cuda" in reason.lower()


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        select_device("metal", mps=True, cuda=False)


def test_tile_profile_fills_nulls_and_keeps_explicit_values():
    cfg = default_config()
    assert cfg["runtime"]["device"] == "auto"
    assert cfg["deblur"]["tile"] is None
    assert cfg["deblur"]["scale"] == pytest.approx(0.5)
    apply_device_profile(cfg, "cpu")
    assert cfg["deblur"]["tile"] == 384
    assert cfg["upscale"]["tile"] == 256

    cfg = default_config()
    cfg["deblur"]["tile"] = 128
    cfg["deblur"]["overlap"] = 0
    apply_device_profile(cfg, "mps")
    assert cfg["deblur"]["tile"] == 128
    assert cfg["deblur"]["overlap"] == 0
    assert cfg["upscale"]["tile"] == 384
    assert cfg["upscale"]["overlap"] == 32

    cfg = default_config()
    apply_device_profile(cfg, "cuda")
    assert cfg["deblur"]["tile"] == 512
    assert cfg["upscale"]["tile"] == 512


def test_cli_device_flag_reaches_config():
    args = build_parser().parse_args(["photo.jpg", "-o", "out", "--device", "cpu"])
    cfg = apply_cli(default_config(), args)
    assert cfg["runtime"]["device"] == "cpu"


def test_mps_fallback_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    enable_mps_fallback()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_mps_error_classifier():
    assert mps_error_should_fallback(RuntimeError("The operator is not currently implemented for the MPS device"))
    assert mps_error_should_fallback(RuntimeError("MPS backend out of memory"))
    assert not mps_error_should_fallback(RuntimeError("a plain cpu failure"))


def test_attempt_retries_on_cpu_after_an_mps_gap():
    runner = TorchDevice("cpu")
    runner.name = "mps"
    runner.torch_device = type("Dev", (), {"type": "mps"})()

    class Module:
        def to(self, _device):
            return self

    seen = []

    def call(dev):
        seen.append(dev.type)
        if dev.type == "mps":
            raise RuntimeError("The operator 'aten::foo' is not currently implemented for the MPS device")
        return "ok"

    assert runner.attempt(Module(), call) == "ok"
    assert runner.fell_back is True
    assert runner.torch_device.type == "cpu"
    assert seen == ["mps", "cpu"]


def test_cpu_errors_are_not_retried():
    runner = TorchDevice("cpu")

    def call(_dev):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        runner.attempt(object(), call)


def test_detector_does_not_use_coreml():
    available = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert providers_for("mps", available, use_coreml=False) == ["CPUExecutionProvider"]
    assert providers_for("mps", available, use_coreml=True)[0] == "CoreMLExecutionProvider"
    assert providers_for("cuda", available)[0] == "CUDAExecutionProvider"


def test_onnx_provider_order():
    assert providers_for("cpu", ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]
    assert providers_for("mps", ["CoreMLExecutionProvider", "CPUExecutionProvider"]) == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert providers_for("mps", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert providers_for("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_probe_does_not_crash():
    found = probe_accelerators()
    assert set(found) == {"mps", "cuda"}
    assert isinstance(found["mps"], bool)
    assert isinstance(found["cuda"], bool)

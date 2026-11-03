"""Where the PyTorch stages run.

``auto`` prefers Apple Silicon MPS, then CUDA, then CPU. Tile sizes follow
the device unless the config sets them. Half-scale deblur stays the default
on every device; a Mac can opt into full-scale with ``--deblur-scale 1``.

Face restoration stays one face at a time. A larger batch would raise memory
use without changing the identity blend.
"""

from __future__ import annotations

import os
from typing import Callable

# Positive sizes. A null tile in default.yaml means "use this profile".
PROFILES = {
    "cpu": {"deblur_tile": 384, "deblur_overlap": 64, "esrgan_tile": 256, "esrgan_overlap": 16},
    # Unified memory on M1–M4. Larger than the 4 GB CPU profile, still tiled.
    "mps": {"deblur_tile": 512, "deblur_overlap": 64, "esrgan_tile": 384, "esrgan_overlap": 32},
    "cuda": {"deblur_tile": 512, "deblur_overlap": 64, "esrgan_tile": 512, "esrgan_overlap": 32},
}

_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


def enable_mps_fallback() -> None:
    """Allow MPS to run unimplemented ops on CPU. Set this before importing torch."""
    os.environ.setdefault(_MPS_FALLBACK_ENV, "1")


def select_device(requested: str, *, mps: bool, cuda: bool) -> tuple[str, str]:
    """Return ``(device, reason)`` without touching the hardware.

    ``requested`` is ``auto``, ``mps``, ``cuda``, or ``cpu``. A requested
    accelerator that is not actually available becomes CPU, with the reason
    saying so.
    """
    name = (requested or "auto").strip().lower()
    if name not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError(f"device must be auto, mps, cuda, or cpu (got {requested!r})")
    if name == "cpu":
        return "cpu", "cpu requested"
    if name == "mps":
        if mps:
            return "mps", "mps requested"
        return "cpu", "mps was requested but this PyTorch build cannot use it; using CPU"
    if name == "cuda":
        if cuda:
            return "cuda", "cuda requested"
        return "cpu", "cuda was requested but no CUDA device is available; using CPU"
    if mps:
        return "mps", "Apple Silicon MPS"
    if cuda:
        return "cuda", "CUDA"
    return "cpu", "no MPS or CUDA device; using CPU"


def probe_accelerators() -> dict[str, bool]:
    import torch

    mps = False
    backend = getattr(torch.backends, "mps", None)
    if backend is not None:
        try:
            mps = bool(backend.is_available())
        except Exception:
            mps = False
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return {"mps": mps, "cuda": cuda}


def apply_device_profile(cfg: dict, device: str) -> dict:
    """Fill tile sizes that the config left unset. Explicit numbers win."""
    profile = PROFILES[device if device in PROFILES else "cpu"]
    mapping = {
        ("deblur", "tile"): profile["deblur_tile"],
        ("deblur", "overlap"): profile["deblur_overlap"],
        ("upscale", "tile"): profile["esrgan_tile"],
        ("upscale", "overlap"): profile["esrgan_overlap"],
    }
    for (section, key), value in mapping.items():
        current = cfg[section].get(key)
        if current is None or current == "auto":
            cfg[section][key] = value
    return cfg


def mps_error_should_fallback(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "mps",
        "not implemented",
        "not currently implemented",
        "out of memory",
        "mps backend",
    )
    return any(needle in text for needle in needles)


class TorchDevice:
    """fp32 PyTorch device with a one-time MPS → CPU fallback.

    The fallback covers an op MPS cannot run and an out-of-memory error on
    that module. Later modules still try MPS. Weights stay fp32: several MPS
    kernels reject fp16.
    """

    def __init__(self, name: str):
        import torch

        self.name = name
        self.torch_device = torch.device(name)
        self.fell_back = False
        if name == "mps":
            enable_mps_fallback()
        if name == "cuda":
            torch.backends.cudnn.benchmark = False

    def place(self, module):
        module = module.float()
        try:
            return module.to(self.torch_device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.torch_device.type == "mps" and mps_error_should_fallback(exc):
                self._fallback(exc)
                return module.to(self.torch_device)
            raise

    def attempt(self, module, call: Callable):
        import torch

        try:
            with torch.inference_mode():
                return call(self.torch_device)
        except (RuntimeError, NotImplementedError) as exc:
            if self.torch_device.type == "mps" and not self.fell_back and mps_error_should_fallback(exc):
                self._fallback(exc)
                try:
                    module.to(self.torch_device)
                except Exception:
                    pass
                with torch.inference_mode():
                    return call(self.torch_device)
            raise

    def _fallback(self, exc: BaseException) -> None:
        import torch

        self.fell_back = True
        self.name = "cpu"
        self.torch_device = torch.device("cpu")
        print(f"MPS could not run this step ({exc}); continuing on CPU", flush=True)


def release_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "empty_cache"):
            try:
                mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass

"""Load the packaged defaults and overlay a user YAML file."""

from __future__ import annotations

import copy
from importlib.resources import files
from pathlib import Path

import yaml


def default_config() -> dict:
    text = files("old_photo_restore").joinpath("default.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("default.yaml did not contain a mapping")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict:
    cfg = default_config()
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config {path} must be a YAML mapping")
        cfg = deep_merge(cfg, user)
    return cfg


def apply_cli(cfg: dict, args) -> dict:
    """Overlay the flags that have a CLI form. None means 'leave the config'."""
    if getattr(args, "blend", None) is not None:
        cfg["blend"]["default"] = float(args.blend)
    if getattr(args, "fidelity", None) is not None:
        cfg["codeformer"]["fidelity"] = float(args.fidelity)
    if getattr(args, "scale", None) is not None:
        cfg["output"]["scales"] = [1, 2] if args.scale == "both" else [int(args.scale)]
    if getattr(args, "deblur_scale", None) is not None:
        cfg["deblur"]["scale"] = float(args.deblur_scale)
    if getattr(args, "esrgan_mix", None) is not None:
        cfg["upscale"]["esrgan_weight"] = float(args.esrgan_mix)
    if getattr(args, "max_faces", None) is not None:
        cfg["runtime"]["max_faces"] = int(args.max_faces)
    if getattr(args, "cpu_threads", None) is not None:
        cfg["runtime"]["cpu_threads"] = int(args.cpu_threads)
    if getattr(args, "weights_dir", None):
        cfg["runtime"]["weights_dir"] = args.weights_dir
    if getattr(args, "device", None):
        cfg["runtime"]["device"] = str(args.device).lower()
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    fidelity = float(cfg["codeformer"]["fidelity"])
    blend = float(cfg["blend"]["default"])
    mix = float(cfg["upscale"]["esrgan_weight"])
    scale = float(cfg["deblur"]["scale"])
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError("--fidelity / codeformer.fidelity must be between 0 and 1")
    if not 0.0 <= blend <= 1.0:
        raise ValueError("--blend must be between 0 and 1")
    if not 0.0 <= mix <= 1.0:
        raise ValueError("--esrgan-mix must be between 0 and 1")
    if not 0.0 < scale <= 1.0:
        raise ValueError("--deblur-scale must be in (0, 1]")
    scales = list(cfg["output"]["scales"])
    if not scales or any(int(s) not in (1, 2) for s in scales):
        raise ValueError("output.scales must contain 1 and/or 2")
    if int(cfg["runtime"]["max_faces"]) < 0:
        raise ValueError("--max-faces must be >= 0")
    device = str(cfg["runtime"].get("device") or "auto").lower()
    if device not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("--device must be auto, mps, cuda, or cpu")
    cfg["runtime"]["device"] = device

"""Command line for old-photo-restore."""

from __future__ import annotations

import argparse
import sys

from old_photo_restore import __version__
from old_photo_restore.config import apply_cli, load_config
from old_photo_restore.device import enable_mps_fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="old-photo-restore",
        description=(
            "Restore a blurry, shaken, or old group photo without changing who the "
            "people are. Uses Apple Silicon MPS or CUDA when that is the device, "
            "otherwise CPU. Weights download on first run."
        ),
    )
    parser.add_argument("input", nargs="?", help="Input photo (JPEG, PNG, WEBP, TIFF)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--refs", help="Directory of reference photos (any age)")
    parser.add_argument("--ref-map", help="YAML mapping reference filename -> face number (P1 is 1)")
    parser.add_argument("--scale", choices=["1", "2", "both"], help="1 original size, 2 double size, both (default)")
    parser.add_argument("--blend", type=float, help="Face blend, 0 keeps the photo, 1 is fully restored (default 0.7)")
    parser.add_argument("--fidelity", type=float, help="CodeFormer fidelity w in [0, 1] (default 0.9)")
    parser.add_argument("--deblur-scale", type=float, help="Deblur scale (default 0.5)")
    parser.add_argument("--esrgan-mix", type=float, help="Real-ESRGAN weight vs Lanczos (default 0.75)")
    parser.add_argument("--config", help="YAML config merged over the packaged defaults")
    parser.add_argument("--weights-dir", help="Weight cache (default ~/.cache/old-photo-restore/weights)")
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        help="auto (default): MPS on Apple Silicon, else CUDA, else CPU",
    )
    parser.add_argument("--cpu-threads", type=int, help="CPU threads for CPU work and ONNX (default min(8, cores))")
    parser.add_argument("--max-faces", type=int, help="Restore only the N largest faces (0 = all)")
    parser.add_argument("--download-weights", action="store_true", help="Download weights, including glintr100, and exit")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    # Must be set before PyTorch is imported. auto and mps both want the fallback.
    if (args.device or "auto") in {"auto", "mps"}:
        enable_mps_fallback()
    if args.download_weights:
        from old_photo_restore.weights import prefetch, weights_dir

        dest = weights_dir(args.weights_dir)
        prefetch(dest, include_glint=True)
        print(f"weights ready in {dest}")
        return 0
    if not args.input or not args.output:
        parser.error("input and -o/--output are required")
    try:
        cfg = apply_cli(load_config(args.config), args)
        # Importing the pipeline loads PyTorch; keep --help and --version light.
        from old_photo_restore.pipeline import restore

        restore(args.input, args.output, cfg, refs_dir=args.refs, ref_map=args.ref_map)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0

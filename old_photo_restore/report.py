"""Plain-text report. One screen, enough to see which faces were treated gently."""

from __future__ import annotations

from pathlib import Path


def render_report(result: dict) -> str:
    lines = [
        "old-photo-restore",
        f"input: {result['input']}",
        f"output: {result['output']}",
        f"faces detected: {result['n_detected']}",
        f"faces restored: {result['n_restored']}",
        "",
        "settings:",
        f"  deblur: NAFNet-REDS-width64  scale={result['deblur_scale']}  tile={result['deblur_tile']}",
        f"  codeformer fidelity w={result['fidelity']}  adain={result['adain']}",
        f"  blend default={result['blend_default']}  weak={result['blend_weak']}",
        f"  esrgan mix={result['esrgan_weight']}  lanczos={1.0 - result['esrgan_weight']:.2f}",
        f"  scales: {', '.join(str(s) for s in result['scales'])}",
        f"  device: {result.get('device', 'cpu')} ({result.get('device_reason', '')})",
        f"  esrgan tile: {result.get('esrgan_tile', '')}",
        f"  threads: {result['threads']}",
    ]
    if result.get("references"):
        lines.append(f"  references: {result['references']}")
    lines.append("")
    weak = [f"P{n}" for n in result.get("weak_faces", [])]
    lines.append("weak faces: " + (", ".join(weak) if weak else "none"))
    if result.get("ambiguous_refs"):
        bits = [f"{a['ref_id']} (near P{a['face']}, {a['arc']:.3f})" for a in result["ambiguous_refs"]]
        lines.append("ambiguous references, left unmatched: " + ", ".join(bits))
    lines.append("")
    lines.append("faces:")
    for face in result["faces"]:
        parts = [
            f"P{face['number']}",
            f"det={face['det_score']:.3f}",
            f"bbox={_box(face['bbox'])}",
        ]
        if face.get("skipped"):
            parts.append(f"skipped={face.get('skip_reason', 'yes')}")
        else:
            parts.append(f"blend={face['blend']:.2f}")
            parts.append(f"reason={face['blend_reason']}")
            parts.append(f"identity_deblur={_num(face.get('identity_deblur'))}")
            parts.append(f"identity_full={_num(face.get('identity_full'))}")
            parts.append(f"identity_out={_num(face.get('identity_out'))}")
            if face.get("weak"):
                parts.append("weak")
            if face.get("ref_id"):
                parts.append(f"ref={face['ref_id']}")
                parts.append(f"arc_ref={_num(face.get('arc_ref'))}")
                parts.append(f"glint_ref={_num(face.get('glint_ref'))}")
                if face.get("params"):
                    p = face["params"]
                    parts.append(
                        "params kin={kin:.2f} w={fidelity:.2f} kp={kp:.2f} a={blend:.2f}".format(**p)
                    )
                if face.get("mouth_restored"):
                    parts.append("mouth=restored_from_non_reference")
                if face.get("flow_mag") is not None:
                    parts.append(f"flow_px={face['flow_mag']:.1f}")
        lines.append("  " + "  ".join(parts))
    lines.append("")
    lines.append("outputs:")
    for name in result.get("files", []):
        lines.append(f"  {name}")
    if result.get("timings"):
        lines.append("")
        lines.append("timings (seconds):")
        for key, value in result["timings"].items():
            lines.append(f"  {key}: {value:.1f}")
    lines.append("")
    return "\n".join(lines)


def write_report(path: str | Path, result: dict) -> str:
    text = render_report(result)
    Path(path).write_text(text, encoding="utf-8")
    return text


def _box(bbox) -> str:
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    return f"[{x0},{y0},{x1},{y1}]"


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"

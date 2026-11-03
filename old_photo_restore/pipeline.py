"""Identity-faithful restoration pipeline.

Stages, in order, each released before the next heavy model is loaded:

1. Downscale and NAFNet-REDS deblur, then put the original's colour back.
2. Detect every face and number them left to right.
3. Embed faces and any references (ArcFace, glintr100 as a second opinion).
4. CodeFormer on each face, optionally after a 36-combination reference search.
5. Real-ESRGAN x2 (original size) and/or x4 (2x size), mixed with Lanczos,
   then paste the restored faces on top so the upscaler does not repaint them.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import cv2
import numpy as np

from old_photo_restore.detect import ScrfdDetector, detect_faces
from old_photo_restore.device import (
    TorchDevice,
    apply_device_profile,
    probe_accelerators,
    release_memory,
    select_device,
)
from old_photo_restore.identity import Embedder, cosine
from old_photo_restore.imageops import (
    FEATURE_CLASSES,
    MOUTH_CLASSES,
    auto_color_sigma,
    class_mask,
    draw_overview,
    face_blend_mask,
    face_grid,
    paste_face,
    read_bgr,
    resize_to_scale,
    side_by_side,
    square_crop,
    to_u8,
    transfer_low_frequency,
    warp_to_512,
    write_image,
)
from old_photo_restore.nets import (
    configure_threads,
    load_codeformer,
    load_nafnet,
    load_parser,
    load_rrdb,
    run_codeformer,
    run_nafnet_tiled,
    run_parser,
    run_rrdb_tiled,
)
from old_photo_restore.reference import (
    assign_references,
    choose_candidate,
    decide_blend,
    detail_band,
    glint_veto,
    grid_combinations,
    mouth_delta,
    parse_ref_map,
    restore_mouth,
    split_ref_key,
    warp_reference,
)
from old_photo_restore.report import write_report
from old_photo_restore.weights import ensure_core, ensure_glint, weights_dir

_REF_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def log(message: str) -> None:
    print(message, flush=True)


def restore(input_path: str, output_dir: str, cfg: dict, refs_dir: str | None = None, ref_map: str | None = None) -> dict:
    started = time.perf_counter()
    timings = {}
    threads = configure_threads(int(cfg["runtime"]["cpu_threads"]))
    device_name, device_reason = select_device(cfg["runtime"].get("device") or "auto", **probe_accelerators())
    apply_device_profile(cfg, device_name)
    runner = TorchDevice(device_name)
    cfg["runtime"]["_runner"] = runner
    cfg["runtime"]["resolved_onnx_device"] = device_name
    log(f"device {device_name} ({device_reason}); deblur tile {cfg['deblur']['tile']}, esrgan tile {cfg['upscale']['tile']}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = weights_dir(cfg["runtime"].get("weights_dir"))
    log(f"weights cache: {cache}")
    paths = ensure_core(cache)

    t0 = time.perf_counter()
    original = read_bgr(input_path)
    if original.size == 0 or min(original.shape[:2]) < 16:
        raise RuntimeError("input image is empty or too small")
    work = resize_to_scale(original, float(cfg["deblur"]["scale"]))
    log(f"input {original.shape[1]}x{original.shape[0]} -> deblur canvas {work.shape[1]}x{work.shape[0]}")
    net = load_nafnet(str(paths["nafnet"]), runner)
    deblurred = run_nafnet_tiled(net, work, int(cfg["deblur"]["tile"]), int(cfg["deblur"]["overlap"]), runner)
    del net
    release_memory()
    sigma = auto_color_sigma(work.shape[0], work.shape[1], float(cfg["deblur"]["color_sigma"]))
    deblurred = transfer_low_frequency(deblurred, work, sigma)
    deblurred_u8 = to_u8(deblurred)
    timings["deblur"] = time.perf_counter() - t0
    log(f"deblur done in {timings['deblur']:.1f}s (colour sigma {sigma:.1f})")

    t0 = time.perf_counter()
    detector = ScrfdDetector(str(paths["det"]), threads, device_name)
    log(f"face detector providers: {', '.join(detector.session.providers)}")
    raw = detect_faces(original, detector, cfg["detect"])
    del detector
    gc.collect()
    faces = _package_faces(raw, original, work, deblurred_u8, cfg)
    timings["detect"] = time.perf_counter() - t0
    log(f"detected {len(faces)} face(s) in {timings['detect']:.1f}s")
    overview = draw_overview(original, faces, int(cfg["output"]["overview_max_side"]))
    write_image(out / "faces-overview.jpg", overview, int(cfg["output"]["jpeg_quality"]))

    ref_cfg = cfg["reference"]
    explicit = parse_ref_map(ref_map)
    if explicit and not refs_dir:
        raise RuntimeError("--ref-map requires --refs")
    ref_faces = []
    glint_path = None
    if refs_dir or explicit:
        ref_faces = _load_references(refs_dir, explicit, cfg, threads, paths, device_name)
        try:
            glint_path = ensure_glint(cache)
        except Exception as exc:
            log(f"glintr100 unavailable ({exc}); continuing with ArcFace only")
            glint_path = None

    t0 = time.perf_counter()
    _embed_faces(faces, ref_faces, paths, glint_path, threads, ref_cfg, explicit, device_name)
    timings["identity_match"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _restore_faces(faces, cfg, paths)
    timings["faces"] = time.perf_counter() - t0
    log(f"face restoration done in {timings['faces']:.1f}s")

    t0 = time.perf_counter()
    _score_and_blend(faces, cfg, paths, glint_path, threads)
    timings["score"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    written, networks = _upscale_and_compare(original, work, deblurred_u8, faces, cfg, paths, out)
    timings["upscale"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - started

    weak = [f["number"] for f in faces if f.get("weak")]
    result = {
        "input": str(input_path),
        "output": str(out),
        "n_detected": len(faces),
        "n_restored": sum(1 for f in faces if not f.get("skipped")),
        "deblur_scale": float(cfg["deblur"]["scale"]),
        "deblur_tile": int(cfg["deblur"]["tile"]),
        "fidelity": float(cfg["codeformer"]["fidelity"]),
        "adain": bool(cfg["codeformer"]["adain"]),
        "blend_default": float(cfg["blend"]["default"]),
        "blend_weak": float(cfg["blend"]["weak"]),
        "esrgan_weight": float(cfg["upscale"]["esrgan_weight"]),
        "scales": list(cfg["output"]["scales"]),
        "threads": threads,
        "device": runner.name,
        "device_reason": device_reason + ("; fell back to CPU during the run" if runner.fell_back else ""),
        "esrgan_tile": int(cfg["upscale"]["tile"]),
        "references": str(refs_dir) if refs_dir else "",
        "weak_faces": weak,
        "ambiguous_refs": _ambiguous(faces),
        "faces": faces,
        "files": written,
        "upscale_networks": networks,
        "timings": timings,
    }
    write_report(out / "report.txt", result)
    written.append("report.txt")
    log(f"wrote {len(written)} files to {out} in {timings['total']:.1f}s")
    if weak:
        log("weak faces (blend lowered): " + ", ".join(f"P{n}" for n in weak))
    return result


def _package_faces(raw, original, work, deblurred_u8, cfg) -> list[dict]:
    sx = work.shape[1] / original.shape[1]
    sy = work.shape[0] / original.shape[0]
    faces = []
    for item in raw:
        bbox = np.asarray(item["bbox"], np.float32)
        landmarks = np.asarray(item["landmarks"], np.float32)
        work_lm = landmarks * np.array([sx, sy], np.float32)
        try:
            cin, affine = warp_to_512(deblurred_u8, work_lm)
            orig_crop, _ = warp_to_512(work, work_lm)
        except RuntimeError:
            continue
        faces.append(
            {
                "bbox": bbox,
                "landmarks": landmarks,
                "det_score": float(item["score"]),
                "affine": affine,
                "cin": cin.astype(np.float32),
                "orig_crop": orig_crop,
                "skipped": False,
                "skip_reason": "",
            }
        )
    faces.sort(key=lambda f: (float(f["bbox"][0]), float(f["bbox"][1])))
    for i, face in enumerate(faces, start=1):
        face["number"] = i
    limit = int(cfg["runtime"]["max_faces"])
    if limit > 0 and len(faces) > limit:
        largest = sorted(
            faces,
            key=lambda f: -((f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])),
        )
        keep = {f["number"] for f in largest[:limit]}
        for face in faces:
            if face["number"] not in keep:
                face["skipped"] = True
                face["skip_reason"] = "max-faces"
                face["cin"] = None
                face["orig_crop"] = None
    return faces


def _load_references(refs_dir, explicit, cfg, threads, paths, device: str) -> list[dict]:
    files = []
    if refs_dir:
        root = Path(refs_dir)
        if not root.is_dir():
            raise RuntimeError(f"reference directory not found: {refs_dir}")
        files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in _REF_EXTS and not p.name.startswith(".")]
    names = [p.name for p in files]
    if len(names) != len(set(names)):
        raise RuntimeError("reference filenames must be unique")
    known = set(names)
    for key in explicit:
        filename, _ = split_ref_key(key)
        if filename not in known and key not in known:
            raise RuntimeError(f"reference map entry {key!r} does not match a file in {refs_dir}")
    if not files:
        return []
    detector = ScrfdDetector(str(paths["det"]), threads, device)
    found = []
    for path in files:
        image = read_bgr(path)
        detected = detect_faces(image, detector, cfg["detect"])
        detected.sort(key=lambda d: -((d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])))
        if not detected:
            log(f"reference {path.name}: no face detected, skipped")
            continue
        for index, det in enumerate(detected):
            found.append(
                {
                    "id": f"{path.name}#{index}",
                    "filename": path.name,
                    "index": index,
                    "image": image,
                    "bbox": det["bbox"],
                    "landmarks": det["landmarks"],
                    "det_score": float(det["score"]),
                }
            )
        log(f"reference {path.name}: {len(detected)} face(s)")
    del detector
    gc.collect()
    return found


def _embed_faces(faces, ref_faces, paths, glint_path, threads, ref_cfg, explicit, device: str) -> None:
    active = [f for f in faces if not f.get("skipped")]
    if not active and not ref_faces:
        return
    arc = Embedder(str(paths["arcface"]), threads, device)
    glint = Embedder(str(glint_path), threads, device) if glint_path and ref_faces else None
    if arc.session.providers:
        log(f"ArcFace providers: {', '.join(arc.session.providers)}")
    for face in active:
        face["e_orig"] = arc.embed_aligned_512(face["orig_crop"])
        face["e_deblur"] = arc.embed_aligned_512(to_u8(face["cin"]))
        face["identity_deblur"] = cosine(face["e_deblur"], face["e_orig"])
        if glint is not None:
            face["g_orig"] = glint.embed_aligned_512(face["orig_crop"])
            face["g_deblur"] = glint.embed_aligned_512(to_u8(face["cin"]))
    for ref in ref_faces:
        ref["e_arc"] = arc.embed_landmarks(ref["image"], ref["landmarks"])
        if glint is not None:
            ref["e_glint"] = glint.embed_landmarks(ref["image"], ref["landmarks"])
        # The full image is no longer needed once the embedding and landmarks exist,
        # except for the later warp. Keep it; reference photos are small relative
        # to the group photo.
    use_glint = glint is not None
    del arc, glint
    gc.collect()
    if not ref_faces or not active:
        return
    by_id = {ref["id"]: ref for ref in ref_faces}
    locked = {}
    for key, face_number in explicit.items():
        filename, index = split_ref_key(key)
        if index is None:
            index = 0
        ref_id = f"{filename}#{index}"
        if ref_id not in by_id:
            raise RuntimeError(f"no face {ref_id} for reference map entry {key}")
        numbers = {f["number"] for f in active}
        if int(face_number) not in numbers:
            known = ", ".join(f"P{n}" for n in sorted(numbers)) or "none"
            raise RuntimeError(f"reference map points {key} at P{face_number}, but restorable faces are: {known}")
        locked[ref_id] = int(face_number)
    assigned, ambiguous = assign_references(
        [f["number"] for f in active],
        [f["e_deblur"] for f in active],
        [r["id"] for r in ref_faces],
        [r["e_arc"] for r in ref_faces],
        locked,
        float(ref_cfg["min_cosine"]),
        float(ref_cfg["min_margin"]),
    )
    by_number = {f["number"]: f for f in active}
    if use_glint:
        glint_by_face = {f["number"]: f["g_deblur"] for f in active}
        kept = []
        for item in assigned:
            if item.get("locked"):
                kept.append(item)
                continue
            ref = by_id[item["ref_id"]]
            ok, g_score, best_face, best = glint_veto(
                ref["e_glint"],
                glint_by_face,
                item["face"],
                float(ref_cfg["glint_min"]),
                float(ref_cfg["glint_margin"]),
            )
            item["glint"] = g_score
            if not ok:
                log(f"reference {item['ref_id']} rejected by glintr100 (prefers P{best_face} at {best:.3f})")
                ambiguous.append({"ref_id": item["ref_id"], "face": item["face"], "arc": item["arc"], "second": best, "glint_veto": True})
                continue
            kept.append(item)
        assigned = kept
    used_refs = set()
    for item in assigned:
        face = by_number[item["face"]]
        if face.get("ref_id"):
            continue
        if item["ref_id"] in used_refs:
            continue
        face["ref_id"] = item["ref_id"]
        face["ref"] = by_id[item["ref_id"]]
        face["match_arc"] = item["arc"]
        face["match_glint"] = item.get("glint")
        used_refs.add(item["ref_id"])
        log(f"reference {item['ref_id']} -> P{item['face']}  arc={item['arc']:.3f}")
    for face in active:
        face["ambiguous_refs"] = [a for a in ambiguous if a["face"] == face["number"]]
    # Stash the leftovers on the first face so the report can list them once.
    if faces:
        faces[0]["_ambiguous_all"] = ambiguous


def _ambiguous(faces) -> list:
    for face in faces:
        if face.get("_ambiguous_all"):
            return face["_ambiguous_all"]
    return []


def _restore_faces(faces, cfg, paths) -> None:
    active = [f for f in faces if not f.get("skipped")]
    if not active:
        return
    runner = cfg["runtime"]["_runner"]
    codeformer = load_codeformer(str(paths["codeformer"]), runner)
    parser = load_parser(str(paths["parsenet"]), runner)
    fidelity = float(cfg["codeformer"]["fidelity"])
    adain = bool(cfg["codeformer"]["adain"])
    color_sigma = float(cfg["codeformer"]["color_sigma"])
    for face in active:
        if face.get("ref_id"):
            _reference_search(face, codeformer, parser, cfg, fidelity, adain, color_sigma)
        else:
            restored = run_codeformer(codeformer, face["cin"], fidelity, runner, adain)
            restored = transfer_low_frequency(restored, face["cin"], color_sigma)
            face["restored_full"] = restored
            face["parse"] = run_parser(parser, restored, runner)
            log(f"P{face['number']} codeformer w={fidelity}")
    del codeformer, parser
    release_memory()


def _reference_search(face, codeformer, parser, cfg, fidelity, adain, color_sigma) -> None:
    ref_cfg = cfg["reference"]
    runner = cfg["runtime"]["_runner"]
    cin = face["cin"].astype(np.float32)
    base = run_codeformer(codeformer, cin, fidelity, runner, adain)
    base = transfer_low_frequency(base, cin, color_sigma)
    face["restored_full"] = base
    base_parse = run_parser(parser, base, runner)
    face["parse"] = base_parse
    feat = class_mask(base_parse, FEATURE_CLASSES, dilate=int(ref_cfg["feat_dilate"]), blur=float(ref_cfg["feat_blur"]))
    mouth0 = class_mask(base_parse, MOUTH_CLASSES, dilate=21, blur=6.0)
    aligned, _ = warp_to_512(face["ref"]["image"], face["ref"]["landmarks"])
    warped, flow_mag = warp_reference(base, aligned, float(ref_cfg["flow_smooth"]), float(ref_cfg["flow_clip"]))
    band = detail_band(warped, float(ref_cfg["detail_sigma"]))
    face["ref_preview"] = to_u8(warped)
    face["flow_mag"] = flow_mag
    grid = cfg["reference"]["grid"]
    per_face = {int(k): float(v) for k, v in (cfg["blend"].get("per_face") or {}).items()}
    if face["number"] in per_face:
        grid = dict(grid)
        grid["blend"] = [per_face[face["number"]]]
    cache = {(0.0, round(fidelity, 5)): (base, base_parse, mouth0)}

    def variant(kin, w):
        key = (round(float(kin), 5), round(float(w), 5))
        if key in cache:
            return cache[key]
        source = cin if abs(kin) < 1e-6 else np.clip(cin + float(kin) * feat[..., None] * band, 0, 255)
        restored = transfer_low_frequency(run_codeformer(codeformer, source, w, runner, adain), cin, color_sigma)
        parsed = run_parser(parser, restored, runner)
        mouth = class_mask(parsed, MOUTH_CLASSES, dilate=21, blur=6.0)
        cache[key] = (restored, parsed, mouth)
        return cache[key]

    rows = []
    for combo in grid_combinations(grid):
        restored, parsed, mouth = variant(combo["kin"], combo["fidelity"])
        boosted = restored + float(combo["kp"]) * feat[..., None] * band
        mask = np.maximum(mouth, mouth0)
        delta = mouth_delta(boosted, base, mask)
        mouth_flag = delta > float(ref_cfg["mouth_delta"])
        if mouth_flag:
            boosted = restore_mouth(boosted, base, mask)
        mixed = float(combo["blend"]) * boosted + (1.0 - float(combo["blend"])) * cin
        rows.append(
            {
                "kin": combo["kin"],
                "fidelity": combo["fidelity"],
                "kp": combo["kp"],
                "blend": combo["blend"],
                "image": to_u8(mixed),
                "parse": parsed,
                "mouth_restored": mouth_flag,
                "mouth_delta": delta,
            }
        )
    face["candidates"] = rows
    n = len(rows)
    log(f"P{face['number']} reference {face['ref_id']}: searched {n} combinations (flow {flow_mag:.1f}px)")


def _score_and_blend(faces, cfg, paths, glint_path, threads) -> None:
    active = [f for f in faces if not f.get("skipped")]
    if not active:
        return
    needs_glint = any(f.get("candidates") for f in active) and glint_path is not None
    device = cfg["runtime"].get("resolved_onnx_device", "cpu")
    arc = Embedder(str(paths["arcface"]), threads, device)
    glint = Embedder(str(glint_path), threads, device) if needs_glint else None
    per_face = {int(k): float(v) for k, v in (cfg["blend"].get("per_face") or {}).items()}
    weights = dict(cfg["reference"]["score_weights"])
    if glint is None:
        weights = {"arc_ref": 0.55, "glint_ref": 0.0, "arc_orig": 0.45, "glint_orig": 0.0}
    for face in active:
        full = to_u8(face["restored_full"])
        e_full = arc.embed_aligned_512(full)
        face["identity_full"] = cosine(e_full, face["e_orig"])
        override = per_face.get(face["number"])
        blend, reason = decide_blend(
            face["det_score"], face["identity_full"], face.get("identity_deblur", 0.0), cfg, override
        )
        # Kept even if a reference search later picks the blend: a face whose
        # full restore drifted is still a weak face.
        policy_reason = reason
        face["blend_reason"] = reason
        if face.get("candidates"):
            baseline_img = to_u8(blend * face["restored_full"] + (1.0 - blend) * face["cin"])
            e_ref = face["ref"]["e_arc"]
            g_ref = face["ref"].get("e_glint")
            g_orig = face.get("g_orig")
            e_base = arc.embed_aligned_512(baseline_img)
            g_base = glint.embed_aligned_512(baseline_img) if glint is not None else None
            scored = [
                {
                    "kin": 0.0,
                    "fidelity": float(cfg["codeformer"]["fidelity"]),
                    "kp": 0.0,
                    "blend": blend,
                    "arc_orig": cosine(e_base, face["e_orig"]),
                    "arc_ref": cosine(e_base, e_ref),
                    "glint_orig": cosine(g_base, g_orig) if g_base is not None and g_orig is not None else 0.0,
                    "glint_ref": cosine(g_base, g_ref) if g_base is not None and g_ref is not None else 0.0,
                    "image": baseline_img,
                    "parse": face["parse"],
                    "mouth_restored": False,
                }
            ]
            for row in face["candidates"]:
                e_out = arc.embed_aligned_512(row["image"])
                g_out = glint.embed_aligned_512(row["image"]) if glint is not None else None
                record = {
                    "kin": row["kin"],
                    "fidelity": row["fidelity"],
                    "kp": row["kp"],
                    "blend": row["blend"],
                    "arc_orig": cosine(e_out, face["e_orig"]),
                    "arc_ref": cosine(e_out, e_ref),
                    "glint_orig": cosine(g_out, g_orig) if g_out is not None and g_orig is not None else 0.0,
                    "glint_ref": cosine(g_out, g_ref) if g_out is not None and g_ref is not None else 0.0,
                    "image": row["image"],
                    "parse": row["parse"],
                    "mouth_restored": row["mouth_restored"],
                }
                scored.append(record)
            best = choose_candidate(
                scored,
                scored[0]["arc_orig"],
                weights,
                float(cfg["reference"]["orig_drop_tol"]),
                float(cfg["reference"]["blend_bonus"]),
                float(cfg["reference"]["blend_bonus_max"]),
            )
            if best is None:
                face["blend"] = blend
                face["params"] = {"kin": 0.0, "fidelity": float(cfg["codeformer"]["fidelity"]), "kp": 0.0, "blend": blend}
                face["mouth_restored"] = False
                face["final"] = baseline_img
                log(f"P{face['number']} reference search kept the no-reference blend {blend:.2f}")
            else:
                face["blend"] = best["blend"]
                face["blend_reason"] = "config" if override is not None else "reference_search"
                face["params"] = {k: best[k] for k in ("kin", "fidelity", "kp", "blend")}
                face["mouth_restored"] = best["mouth_restored"]
                face["parse"] = best["parse"]
                face["final"] = best["image"]
                face["arc_ref"] = best["arc_ref"]
                face["glint_ref"] = best["glint_ref"] if glint is not None else None
                log(
                    f"P{face['number']} chose kin={best['kin']:.2f} w={best['fidelity']:.2f} "
                    f"kp={best['kp']:.2f} blend={best['blend']:.2f} "
                    f"arc_ref={best['arc_ref']:.3f} arc_orig={best['arc_orig']:.3f}"
                )
            face.pop("candidates", None)
        else:
            face["blend"] = blend
            face["final"] = to_u8(blend * face["restored_full"] + (1.0 - blend) * face["cin"])
        face["identity_out"] = cosine(arc.embed_aligned_512(face["final"]), face["e_orig"])
        face["weak"] = policy_reason in {"weak", "drift", "very_low_identity", "low_detection_confidence"}
        face["mask"] = face_blend_mask(face["parse"])
        # Drop bulky intermediates the upscaler does not need.
        for key in ("restored_full", "cin", "orig_crop", "e_orig", "e_deblur", "g_orig", "g_deblur", "ref"):
            face.pop(key, None)
    del arc, glint
    gc.collect()


def _upscale_and_compare(original, work, deblurred_u8, faces, cfg, paths, out: Path):
    written = ["faces-overview.jpg"]
    scales = [int(s) for s in cfg["output"]["scales"]]
    quality = int(cfg["output"]["jpeg_quality"])
    outputs = {}
    networks = {}
    targets = []
    if 1 in scales:
        targets.append(("restored.png", (original.shape[1], original.shape[0])))
    if 2 in scales:
        targets.append(("restored-2x.png", (original.shape[1] * 2, original.shape[0] * 2)))
    for name, (width, height) in targets:
        factor = height / float(work.shape[0])
        net_scale = 4 if factor >= 3.0 else 2 if factor >= 1.5 else 1
        background = _upscale(deblurred_u8, (width, height), net_scale, cfg, paths)
        scale = ((width / work.shape[1]) + (height / work.shape[0])) / 2.0
        order = sorted(
            (f for f in faces if not f.get("skipped")),
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
        for face in order:
            paste_face(background, face["final"], face["affine"], face["mask"], scale)
        write_image(out / name, background, quality)
        outputs[name] = background
        label = {1: "lanczos", 2: "RealESRGAN_x2plus", 4: "RealESRGAN_x4plus"}[net_scale]
        networks[name] = label
        written.append(name)
        log(f"{name}: {width}x{height} via {label}, faces pasted")
        del background
        gc.collect()

    compare_source = outputs.get("restored.png")
    if compare_source is None and "restored-2x.png" in outputs:
        compare_source = cv2.resize(outputs["restored-2x.png"], (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    sharp = outputs.get("restored-2x.png", outputs.get("restored.png"))
    sharp_scale = 2.0 if "restored-2x.png" in outputs else 1.0
    if compare_source is not None:
        plate = side_by_side(original, compare_source)
        write_image(out / "compare-side-by-side.jpg", plate, quality)
        written.append("compare-side-by-side.jpg")
    panels = []
    ref_panels = []
    cell = int(cfg["output"]["face_cell"])
    for face in faces:
        if face.get("skipped") or sharp is None:
            continue
        box = np.asarray(face["bbox"], np.float32) * sharp_scale
        restored_crop = square_crop(sharp, box)
        original_crop = square_crop(original, face["bbox"])
        panels.append([(f"P{face['number']} original", original_crop), (f"P{face['number']} restored", restored_crop)])
        if face.get("ref_preview") is not None:
            ref_panels.append(
                [
                    (f"P{face['number']} original", original_crop),
                    (f"P{face['number']} restored", restored_crop),
                    (f"P{face['number']} {face['ref_id']}", face["ref_preview"]),
                ]
            )
    if panels:
        grid = face_grid(panels, cell, int(cfg["output"]["face_columns"]))
        write_image(out / "compare-faces.jpg", grid, quality)
        written.append("compare-faces.jpg")
    if ref_panels:
        grid = face_grid(ref_panels, cell, 1)
        write_image(out / "compare-refs.jpg", grid, quality)
        written.append("compare-refs.jpg")
    return written, networks


def _upscale(work_u8, target_wh, net_scale, cfg, paths) -> np.ndarray:
    lanczos = cv2.resize(work_u8, target_wh, interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    weight = float(cfg["upscale"]["esrgan_weight"])
    if net_scale == 1 or weight <= 0:
        return to_u8(lanczos)
    key = "esrgan_x2" if net_scale == 2 else "esrgan_x4"
    runner = cfg["runtime"]["_runner"]
    net = load_rrdb(str(paths[key]), net_scale, runner)
    bgr = run_rrdb_tiled(net, work_u8, int(cfg["upscale"]["tile"]), int(cfg["upscale"]["overlap"]), runner)
    del net
    release_memory()
    # Already BGR. A second channel flip is what turned warm faces blue.
    esr = np.clip(bgr * 255.0, 0, 255)
    if (esr.shape[1], esr.shape[0]) != tuple(target_wh):
        esr = cv2.resize(esr, target_wh, interpolation=cv2.INTER_LANCZOS4)
    return to_u8(weight * esr + (1.0 - weight) * lanczos)

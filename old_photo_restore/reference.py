"""Reference matching, high-frequency transfer, and the 36-way parameter search.

References may be a different age from the group photo. Only brows, eyes,
glasses and nose contribute detail. The mouth is put back from the
no-reference result when the search changes teeth or lips.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from old_photo_restore.imageops import high_frequency


def parse_ref_map(path: str | Path | None) -> dict[str, int]:
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping of filename -> face number")
    if isinstance(data.get("mappings"), dict):
        data = data["mappings"]
    mapping = {}
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        mapping[str(key)] = int(value)
    return mapping


def split_ref_key(key: str) -> tuple[str, int | None]:
    if "#" in key:
        name, index = key.rsplit("#", 1)
        if index.isdigit():
            return name, int(index)
    return key, None


def decide_blend(det_score: float, sim_restored: float, sim_deblur: float, cfg: dict, override=None):
    """Return (blend, reason). An explicit per-face value wins."""
    if override is not None:
        return float(override), "config"
    blend = cfg["blend"]
    if sim_restored < float(blend["absolute_very_weak"]):
        return float(blend["very_weak"]), "very_low_identity"
    drifted = sim_restored < sim_deblur - float(blend["drift_margin"])
    low = sim_restored < float(blend["absolute_weak"]) or det_score < float(blend["weak_det_score"])
    if drifted or low:
        reason = "drift" if drifted and not low else "weak"
        if det_score < float(blend["weak_det_score"]) and sim_restored >= float(blend["absolute_weak"]) and not drifted:
            reason = "low_detection_confidence"
        return float(blend["weak"]), reason
    return float(blend["default"]), "default"


def assign_references(photo_numbers, photo_embs, ref_ids, ref_embs, locked, min_cosine, min_margin):
    """One-to-one ArcFace assignment.

    ``locked`` maps a ref id to a face number already chosen by the mapping
    file. Those faces and refs are not reused. Returns (assigned, ambiguous)
    where each item is a dict.
    """
    assigned = []
    used_faces = set()
    used_refs = set()
    for ref_id, face_number in locked.items():
        if ref_id not in ref_ids:
            continue
        r_index = ref_ids.index(ref_id)
        p_index = photo_numbers.index(face_number)
        score = float(photo_embs[p_index] @ ref_embs[r_index])
        assigned.append({"ref_id": ref_id, "face": int(face_number), "arc": score, "locked": True})
        used_faces.add(int(face_number))
        used_refs.add(ref_id)

    free_p = [i for i, n in enumerate(photo_numbers) if int(n) not in used_faces]
    free_r = [i for i, r in enumerate(ref_ids) if r not in used_refs]
    ambiguous = []
    if not free_p or not free_r:
        return assigned, ambiguous
    cost = np.ones((len(free_r), len(free_p)), np.float64)
    scores = np.zeros_like(cost)
    for row, r_index in enumerate(free_r):
        for col, p_index in enumerate(free_p):
            score = float(photo_embs[p_index] @ ref_embs[r_index])
            scores[row, col] = score
            if score >= min_cosine:
                cost[row, col] = 1.0 - score
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        score = float(scores[row, col])
        if score < min_cosine:
            continue
        others = np.delete(scores[row], col)
        second = float(others.max()) if others.size else -1.0
        ref_id = ref_ids[free_r[row]]
        face = int(photo_numbers[free_p[col]])
        if second >= min_cosine and score - second < min_margin:
            ambiguous.append({"ref_id": ref_id, "face": face, "arc": score, "second": second})
            continue
        assigned.append({"ref_id": ref_id, "face": face, "arc": score, "locked": False, "second": second})
    return assigned, ambiguous


def glint_veto(ref_emb, face_embs: dict[int, np.ndarray], proposed: int, glint_min: float, glint_margin: float):
    """Return (keep, proposed_score, best_face, best_score).

    glintr100 is a second opinion. It only rejects a match when the proposed
    pair is weak and another face is clearly better.
    """
    scores = {number: float(ref_emb @ emb) for number, emb in face_embs.items()}
    best_face = max(scores, key=scores.get)
    best = scores[best_face]
    proposed_score = scores.get(proposed, -1.0)
    reject = best_face != proposed and proposed_score < glint_min and best > proposed_score + glint_margin
    return (not reject), proposed_score, best_face, best


def warp_reference(target, ref, smooth: float, clip_mag: float):
    """Warp ``ref`` onto ``target``'s geometry. Both are 512 BGR."""

    def blur(image, sigma):
        return cv2.GaussianBlur(image, (0, 0), sigma)

    target_f = target.astype(np.float32)
    ref_f = ref.astype(np.float32)
    gray_t = cv2.cvtColor(np.clip(blur(target_f, 5), 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(np.clip(blur(ref_f, 5), 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.equalizeHist(gray_t), cv2.equalizeHist(gray_r), None, 0.5, 4, 41, 5, 7, 1.5, 0
    )
    flow = np.stack([blur(flow[..., 0], smooth), blur(flow[..., 1], smooth)], axis=-1)
    mag = float(np.median(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    if clip_mag and mag > clip_mag:
        flow *= clip_mag / mag
    yy, xx = np.mgrid[0:target_f.shape[0], 0:target_f.shape[1]].astype(np.float32)
    warped = cv2.remap(
        ref_f,
        xx + flow[..., 0],
        yy + flow[..., 1],
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped, mag


def detail_band(warped_ref, sigma: float) -> np.ndarray:
    return high_frequency(warped_ref, sigma)


def mouth_delta(image, source, mask) -> float:
    diff = np.abs(image.astype(np.float32) - source.astype(np.float32)).mean(axis=2)
    weight = mask.astype(np.float32)
    return float((diff * weight).sum() / (weight.sum() + 1e-6))


def restore_mouth(image, source, mask) -> np.ndarray:
    weight = mask.astype(np.float32)
    if weight.ndim == 2:
        weight = weight[..., None]
    return image.astype(np.float32) * (1.0 - weight) + source.astype(np.float32) * weight


def grid_combinations(grid: dict):
    keys = ("kin", "fidelity", "kp", "blend")
    values = [grid[key] for key in keys]
    for combo in product(*values):
        yield {key: float(value) for key, value in zip(keys, combo)}


def choose_candidate(rows, baseline_arc_orig, weights, orig_drop_tol, blend_bonus, blend_bonus_max):
    """Pick the parameter row with the best joint score that still looks like the original."""
    viable = []
    for row in rows:
        if row["arc_orig"] < baseline_arc_orig - orig_drop_tol:
            continue
        score = (
            weights["arc_ref"] * row["arc_ref"]
            + weights["glint_ref"] * row["glint_ref"]
            + weights["arc_orig"] * row["arc_orig"]
            + weights["glint_orig"] * row["glint_orig"]
        )
        if row["blend"] <= blend_bonus_max:
            score += blend_bonus
        chosen = dict(row)
        chosen["score"] = float(score)
        viable.append(chosen)
    if not viable:
        return None
    return max(viable, key=lambda row: row["score"])

"""Tests that do not download weights."""

from __future__ import annotations

import numpy as np
import pytest

from old_photo_restore.config import apply_cli, default_config, load_config
from old_photo_restore.imageops import (
    paste_face,
    similarity_transform,
    tile_starts,
    transfer_low_frequency,
    umeyama,
)
from old_photo_restore.reference import (
    assign_references,
    choose_candidate,
    decide_blend,
    grid_combinations,
    mouth_delta,
    parse_ref_map,
    restore_mouth,
)


def test_default_search_is_36_combinations():
    grid = default_config()["reference"]["grid"]
    assert len(list(grid_combinations(grid))) == 36


def test_tiles_cover_the_frame():
    height, width, tile, overlap = 1000, 777, 384, 64
    covered = np.zeros((height, width), dtype=bool)
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            covered[y : y + tile, x : x + tile] = True
    assert covered.all()


def test_low_frequency_colour_comes_from_the_reference():
    src = np.full((48, 48, 3), 10, np.float32)
    ref = np.full((48, 48, 3), 200, np.float32)
    out = transfer_low_frequency(src, ref, sigma=8)
    assert out.mean() > 150


def test_similarity_is_a_scaled_translation():
    src = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [5, 6]], np.float32)
    dst = src * 2 + np.array([3, -4], np.float32)
    for matrix in (umeyama(src, dst), similarity_transform(src, dst)):
        pred = src @ matrix[:, :2].T + matrix[:, 2]
        assert np.allclose(pred, dst, atol=1e-2)


def test_paste_lands_in_the_aligned_region():
    base = np.zeros((600, 600, 3), np.uint8)
    face = np.full((512, 512, 3), 255, np.uint8)
    affine = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    mask = np.ones((512, 512), np.float32)
    paste_face(base, face, affine, mask, scale=1.0)
    assert base[200, 200].mean() > 200
    assert base[590, 590].mean() == 0


def test_decide_blend_lowers_only_weak_faces():
    cfg = default_config()
    blend, reason = decide_blend(0.99, 0.70, 0.72, cfg, None)
    assert (blend, reason) == (0.7, "default")
    blend, reason = decide_blend(0.40, 0.70, 0.72, cfg, None)
    assert reason == "low_detection_confidence"
    assert blend == pytest.approx(0.55)
    blend, reason = decide_blend(0.99, 0.10, 0.50, cfg, None)
    assert reason == "very_low_identity"
    assert blend == pytest.approx(0.50)
    blend, reason = decide_blend(0.99, 0.50, 0.70, cfg, None)
    assert reason == "drift"
    blend, reason = decide_blend(0.2, 0.1, 0.1, cfg, 0.62)
    assert (blend, reason) == (0.62, "config")


def test_reference_assignment_is_one_to_one():
    photo = [np.array([1.0, 0.0, 0.0], np.float32), np.array([0.0, 1.0, 0.0], np.float32)]
    refs = [photo[1], photo[0]]
    assigned, ambiguous = assign_references([1, 2], photo, ["a.jpg#0", "b.jpg#0"], refs, {}, 0.5, 0.03)
    assert ambiguous == []
    pairs = {(item["ref_id"], item["face"]) for item in assigned}
    assert pairs == {("a.jpg#0", 2), ("b.jpg#0", 1)}


def test_explicit_lock_is_not_reused():
    photo = [np.eye(2, dtype=np.float32)[i] for i in range(2)]
    assigned, _ = assign_references([1, 2], photo, ["a#0", "b#0"], photo, {"a#0": 1}, 0.5, 0.03)
    assert sorted((item["ref_id"], item["face"]) for item in assigned) == [("a#0", 1), ("b#0", 2)]


def test_choose_candidate_rejects_identity_loss():
    rows = [
        {"blend": 0.85, "arc_ref": 0.90, "glint_ref": 0.90, "arc_orig": 0.20, "glint_orig": 0.20},
        {"blend": 0.50, "arc_ref": 0.55, "glint_ref": 0.50, "arc_orig": 0.60, "glint_orig": 0.58},
    ]
    weights = {"arc_ref": 0.4, "glint_ref": 0.15, "arc_orig": 0.3, "glint_orig": 0.15}
    best = choose_candidate(rows, baseline_arc_orig=0.62, weights=weights, orig_drop_tol=0.05, blend_bonus=0.01, blend_bonus_max=0.6)
    assert best["blend"] == 0.50
    assert best["arc_orig"] == 0.60


def test_search_keeps_the_candidate_that_still_matches_the_original():
    rows = [
        {"blend": 0.50, "arc_ref": 0.50, "glint_ref": 0.48, "arc_orig": 0.60, "glint_orig": 0.55},
        {"blend": 0.55, "arc_ref": 0.70, "glint_ref": 0.70, "arc_orig": 0.62, "glint_orig": 0.60},
    ]
    weights = {"arc_ref": 0.4, "glint_ref": 0.15, "arc_orig": 0.3, "glint_orig": 0.15}
    best = choose_candidate(rows, baseline_arc_orig=0.62, weights=weights, orig_drop_tol=0.05, blend_bonus=0.01, blend_bonus_max=0.6)
    assert best["arc_orig"] == 0.62
    assert best["blend"] == 0.55


def test_mouth_restore_only_inside_the_mask():
    source = np.zeros((8, 8, 3), np.float32)
    edited = np.full((8, 8, 3), 30, np.float32)
    mask = np.zeros((8, 8), np.float32)
    mask[2:5, 2:5] = 1
    assert mouth_delta(edited, source, mask) == pytest.approx(30)
    put_back = restore_mouth(edited, source, mask)
    assert put_back[3, 3].mean() == 0
    assert put_back[0, 0].mean() == 30


def test_ref_map_and_cli_overlay(tmp_path):
    path = tmp_path / "refs.yaml"
    path.write_text("alice.jpg: 2\nbob.png#1: 11\n", encoding="utf-8")
    assert parse_ref_map(path) == {"alice.jpg": 2, "bob.png#1": 11}
    cfg = load_config(None)
    args = type("A", (), {})()
    args.blend = 0.62
    args.fidelity = None
    args.scale = "1"
    args.deblur_scale = None
    args.esrgan_mix = None
    args.max_faces = 3
    args.cpu_threads = None
    args.weights_dir = None
    cfg = apply_cli(cfg, args)
    assert cfg["blend"]["default"] == pytest.approx(0.62)
    assert cfg["output"]["scales"] == [1]
    assert cfg["runtime"]["max_faces"] == 3


def test_help_exits_cleanly():
    from old_photo_restore.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_package_faces_numbers_left_to_right():
    from old_photo_restore.pipeline import _package_faces

    image = np.full((220, 320, 3), 128, np.uint8)

    def face(x, y, score):
        landmarks = np.array(
            [[x, y], [x + 28, y + 1], [x + 14, y + 16], [x + 6, y + 28], [x + 24, y + 27]],
            np.float32,
        )
        bbox = np.array([x - 10, y - 10, x + 40, y + 46], np.float32)
        return {"bbox": bbox, "landmarks": landmarks, "score": score}

    raw = [face(180, 60, 0.91), face(40, 70, 0.88), face(110, 80, 0.95)]
    faces = _package_faces(raw, image, image, image, default_config())
    assert [f["number"] for f in faces] == [1, 2, 3]
    assert [int(f["bbox"][0]) for f in faces] == sorted(int(f["bbox"][0]) for f in faces)
    assert all(f["cin"].shape == (512, 512, 3) for f in faces)

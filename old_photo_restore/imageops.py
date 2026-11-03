"""Image IO, colour transfer, tiling geometry, and comparison plates.

The pipeline keeps OpenCV's BGR layout. Model wrappers convert at the boundary.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Phone photos are large but not unbounded. Disable Pillow's default bomb check
# and let the caller decide; a large group photo is far under this.
Image.MAX_IMAGE_PIXELS = None

# ParseNet / facexlib 19-class order (the order the working prototype used):
# 0 background, 1 skin, 2 left brow, 3 right brow, 4 left eye, 5 right eye,
# 6 glasses, 7 left ear, 8 right ear, 9 earring, 10 nose, 11 mouth interior,
# 12 upper lip, 13 lower lip, 14 neck, 15 necklace, 16 cloth, 17 hair, 18 hat.
FACE_CLASSES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15)
# Brows, eyes, glasses, nose. Mouth and skin are deliberately absent.
FEATURE_CLASSES = (2, 3, 4, 5, 6, 10)
MOUTH_CLASSES = (11, 12, 13)

# 5-point templates. FFHQ is the 512 CodeFormer crop; ARC is ArcFace's 112 crop.
FFHQ_5 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    np.float32,
)
ARC_5 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    np.float32,
)


def read_bgr(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8, applying EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.asarray(im)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def write_image(path: str | Path, bgr: np.ndarray, jpeg_quality: int = 92) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    image = np.ascontiguousarray(bgr)
    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    if ext in {".jpg", ".jpeg"}:
        ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    elif ext == ".png":
        ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    else:
        ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed to write {path}")


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def resize_to_scale(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize by ``scale`` with even output dimensions (Real-ESRGAN x2 needs that)."""
    h, w = image.shape[:2]
    nh = max(2, int(round(h * scale)))
    nw = max(2, int(round(w * scale)))
    nh -= nh % 2
    nw -= nw % 2
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    return cv2.resize(image, (nw, nh), interpolation=interp)


def transfer_low_frequency(src: np.ndarray, ref: np.ndarray, sigma: float) -> np.ndarray:
    """Keep ``src`` detail, replace its low frequencies with ``ref``'s.

    NAFNet darkens the frame by a few levels; CodeFormer shifts skin colour.
    This is the correction used for both, with a wider sigma on the full frame
    and sigma 24 on a 512 face.
    """
    src_f = src.astype(np.float32)
    ref_f = ref.astype(np.float32)
    if ref_f.shape[:2] != src_f.shape[:2]:
        ref_f = cv2.resize(ref_f, (src_f.shape[1], src_f.shape[0]), interpolation=cv2.INTER_AREA)
    sigma = max(float(sigma), 0.1)
    low_ref = cv2.GaussianBlur(ref_f, (0, 0), sigma)
    low_src = cv2.GaussianBlur(src_f, (0, 0), sigma)
    return np.clip(src_f + low_ref - low_src, 0, 255)


def auto_color_sigma(height: int, width: int, configured: float) -> float:
    if configured and float(configured) > 0:
        return float(configured)
    return max(8.0, min(height, width) / 8.0)


def high_frequency(image: np.ndarray, sigma: float) -> np.ndarray:
    image = image.astype(np.float32)
    band = image - cv2.GaussianBlur(image, (0, 0), max(float(sigma), 0.1))
    band -= band.mean(axis=(0, 1), keepdims=True)
    return band


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Starts of a sliding window that covers ``length`` exactly."""
    if length <= tile:
        return [0]
    overlap = max(0, min(int(overlap), tile // 2))
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` points onto ``dst``."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    dst = np.asarray(dst, np.float64).reshape(-1, 2)
    n, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    cov = (dst_demean.T @ src_demean) / n
    u, singular, vh = np.linalg.svd(cov)
    sign = np.eye(dim)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vh
    variance = (src_demean ** 2).sum() / n
    scale = float((singular * np.diag(sign)).sum() / max(variance, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix = np.zeros((2, 3), np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix.astype(np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, np.float32).reshape(-1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 2)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        matrix = umeyama(src, dst)
    return matrix.astype(np.float32)


def affine_ok(matrix: np.ndarray | None) -> bool:
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return False
    return abs(float(np.linalg.det(matrix[:, :2]))) > 1e-3


def warp_to_512(image: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = similarity_transform(landmarks, FFHQ_5)
    if not affine_ok(matrix):
        raise RuntimeError("could not estimate a face alignment")
    crop = cv2.warpAffine(
        image,
        matrix,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135, 133, 132),
    )
    return crop, matrix


def face_blend_mask(parse: np.ndarray) -> np.ndarray:
    mask = np.isin(parse, FACE_CLASSES).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask = cv2.GaussianBlur(mask, (101, 101), 11)
    mask[:10] = 0
    mask[-10:] = 0
    mask[:, :10] = 0
    mask[:, -10:] = 0
    return np.clip(mask, 0, 1)


def class_mask(parse: np.ndarray, classes, dilate: int = 0, blur: float = 0.0) -> np.ndarray:
    mask = np.isin(parse, list(classes)).astype(np.float32)
    if dilate and dilate > 1:
        mask = cv2.dilate(mask, np.ones((int(dilate), int(dilate)), np.uint8))
    if blur and blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(blur))
    return np.clip(mask, 0, 1)


def paste_face(
    base: np.ndarray,
    face512: np.ndarray,
    affine: np.ndarray,
    soft_mask: np.ndarray,
    scale: float,
    half_pixel: bool = True,
) -> np.ndarray:
    """Blend a 512 face back onto ``base``.

    ``affine`` maps work-image coordinates into the 512 crop. ``base`` is
    ``scale`` times the work image (2 for original size, 4 for the 2x output
    when deblur ran at half resolution).
    """
    face = face512.astype(np.float32)
    mask = np.clip(soft_mask.astype(np.float32), 0, 1)
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = base.shape[:2]
    inv = cv2.invertAffineTransform(np.ascontiguousarray(affine.astype(np.float32)))
    inv = inv * float(scale)
    if half_pixel:
        inv = inv.copy()
        inv[:, 2] += 0.5 * float(scale)
    corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], np.float32) @ inv.T
    x0 = max(int(np.floor(corners[:, 0].min())) - 2, 0)
    y0 = max(int(np.floor(corners[:, 1].min())) - 2, 0)
    x1 = min(int(np.ceil(corners[:, 0].max())) + 2, width)
    y1 = min(int(np.ceil(corners[:, 1].max())) + 2, height)
    if x1 <= x0 or y1 <= y0:
        return base
    local = inv.copy()
    local[0, 2] -= x0
    local[1, 2] -= y0
    size = (x1 - x0, y1 - y0)
    warped = cv2.warpAffine(face, local, size, flags=cv2.INTER_CUBIC)
    coverage = cv2.warpAffine(np.ones((512, 512), np.float32), local, size)
    kernel = max(1, int(round(2 * scale)))
    coverage = cv2.erode(coverage, np.ones((kernel, kernel), np.uint8))
    widen = max(1, int(float(coverage.sum()) ** 0.5) // 20)
    coverage = cv2.erode(coverage, np.ones((2 * widen, 2 * widen), np.uint8))
    coverage = cv2.GaussianBlur(coverage, (2 * widen + 1, 2 * widen + 1), 0)
    warped_mask = cv2.warpAffine(mask, local, size, flags=cv2.INTER_LINEAR)
    alpha = np.minimum(np.clip(warped_mask, 0, 1), np.clip(coverage, 0, 1))[..., None]
    roi = base[y0:y1, x0:x1].astype(np.float32)
    base[y0:y1, x0:x1] = np.clip(alpha * warped + (1.0 - alpha) * roi, 0, 255).round().astype(np.uint8)
    return base


def square_crop(image: np.ndarray, bbox, margin: float = 1.55) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0, 2.0) * margin
    left, top, right, bottom = cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0
    height, width = image.shape[:2]
    ia, ib = int(np.floor(left)), int(np.floor(top))
    ic, id_ = int(np.ceil(right)), int(np.ceil(bottom))
    pad_l, pad_t = max(0, -ia), max(0, -ib)
    pad_r, pad_b = max(0, ic - width), max(0, id_ - height)
    ia, ib = max(ia, 0), max(ib, 0)
    ic, id_ = min(ic, width), min(id_, height)
    crop = image[ib:id_, ia:ic]
    if crop.size == 0:
        crop = np.zeros((2, 2, 3), np.uint8)
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
    return crop


def fit_square(image: np.ndarray, cell: int) -> np.ndarray:
    interp = cv2.INTER_AREA if max(image.shape[:2]) > cell else cv2.INTER_CUBIC
    return cv2.resize(image, (cell, cell), interpolation=interp)


def caption_bar(image: np.ndarray, text: str) -> np.ndarray:
    bar_h = 32
    bar = np.full((bar_h, image.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def hstack_gap(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    height = max(im.shape[0] for im in images)
    padded = []
    for im in images:
        if im.shape[0] != height:
            pad = height - im.shape[0]
            im = cv2.copyMakeBorder(im, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        padded.append(im)
    sep = np.full((height, gap, 3), 255, np.uint8)
    out = padded[0]
    for im in padded[1:]:
        out = np.hstack([out, sep, im])
    return out


def draw_overview(image: np.ndarray, faces: list[dict], max_side: int) -> np.ndarray:
    vis = image
    scale = 1.0
    long_side = max(image.shape[:2])
    if max_side and long_side > max_side:
        scale = max_side / float(long_side)
        vis = cv2.resize(
            image,
            (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    vis = vis.copy()
    for face in faces:
        x0, y0, x1, y1 = [int(round(float(v) * scale)) for v in face["bbox"]]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 180, 40), 2, cv2.LINE_AA)
        label = f"P{face['number']}"
        org = (x0, max(16, y0 - 6))
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(original: np.ndarray, restored: np.ndarray) -> np.ndarray:
    """Full-frame comparison at the original's resolution."""
    if restored.shape[:2] != original.shape[:2]:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    left = caption_bar(original, "Original")
    right = caption_bar(restored, "Restored")
    return hstack_gap([left, right], gap=12)


def face_grid(panels: list[list[tuple[str, np.ndarray]]], cell: int, columns: int) -> np.ndarray:
    """``panels`` is one list per face; each item is (caption, image)."""
    if not panels:
        blank = np.full((cell, cell, 3), 255, np.uint8)
        return caption_bar(blank, "No faces")
    rows_img = []
    columns = max(1, int(columns))
    for entry in panels:
        tiles = [caption_bar(fit_square(img, cell), text) for text, img in entry]
        rows_img.append(hstack_gap(tiles, gap=6))
    # Wrap faces into `columns` faces per row. Each face is already a horizontal strip.
    blank_row = np.full_like(rows_img[0], 255)
    while len(rows_img) % columns:
        rows_img.append(blank_row)
    strips = []
    for i in range(0, len(rows_img), columns):
        strips.append(hstack_gap(rows_img[i : i + columns], gap=16))
    out = strips[0]
    for strip in strips[1:]:
        gap = np.full((10, out.shape[1], 3), 255, np.uint8)
        if strip.shape[1] != out.shape[1]:
            strip = cv2.copyMakeBorder(strip, 0, 0, 0, out.shape[1] - strip.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out = np.vstack([out, gap, strip])
    return out

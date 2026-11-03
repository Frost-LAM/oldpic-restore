"""SCRFD-10GF face detection (InsightFace buffalo_l ``det_10g.onnx``).

Runs on CPU through ONNX Runtime. A tiled pass keeps recall on group photos
where a single 640-pixel view would shrink the back row to nothing, and a
second full-frame pass catches faces that straddle a tile boundary.
"""

from __future__ import annotations

import numpy as np
import cv2

from old_photo_restore.onnx_util import session as ort_session


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _centers(height: int, width: int, stride: int, num_anchors: int) -> np.ndarray:
    grid = np.stack(np.mgrid[0:height, 0:width][::-1], axis=-1).astype(np.float32)
    centers = (grid * stride).reshape(-1, 2)
    if num_anchors > 1:
        centers = np.stack([centers] * num_anchors, axis=1).reshape(-1, 2)
    return centers


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _layout(output_count: int) -> tuple[list[int], int, bool]:
    if output_count == 9:
        return [8, 16, 32], 2, True
    if output_count == 6:
        return [8, 16, 32], 2, False
    if output_count == 15:
        return [8, 16, 32, 64, 128], 1, True
    if output_count == 10:
        return [8, 16, 32, 64, 128], 1, False
    raise RuntimeError(f"unrecognised SCRFD output count: {output_count}")


class ScrfdDetector:
    def __init__(self, path: str, threads: int, device: str = "cpu"):
        # SCRFD's dynamic outputs crash the CoreML EP. CPU is about two seconds.
        self.session = ort_session(path, threads, device, use_coreml=False)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.strides, self.num_anchors, self.use_kps = _layout(len(outputs))
        shape = self.session.get_inputs()[0].shape
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            self.fixed_hw = (int(h), int(w))
        else:
            self.fixed_hw = None

    def _forward(self, bgr: np.ndarray, threshold: float):
        height, width = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            np.ascontiguousarray(bgr),
            scalefactor=1.0 / 128.0,
            size=(width, height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        fmc = len(self.strides)
        boxes, scores, landmarks = [], [], []
        for idx, stride in enumerate(self.strides):
            score = np.asarray(outputs[idx]).reshape(-1).astype(np.float32)
            if score.size and (score.min() < 0.0 or score.max() > 1.0):
                score = _sigmoid(score)
            distance = np.asarray(outputs[idx + fmc]).reshape(-1, 4).astype(np.float32) * stride
            kps = None
            if self.use_kps:
                kps = np.asarray(outputs[idx + fmc * 2]).reshape(-1, 10).astype(np.float32) * stride
            fm_h, fm_w = height // stride, width // stride
            centers = _centers(fm_h, fm_w, stride, self.num_anchors)
            if centers.shape[0] != score.shape[0] or centers.shape[0] != distance.shape[0]:
                raise RuntimeError(
                    f"SCRFD anchor mismatch at stride {stride}: "
                    f"centers {centers.shape[0]}, scores {score.shape[0]}, boxes {distance.shape[0]}"
                )
            keep = np.where(score >= threshold)[0]
            if keep.size == 0:
                continue
            decoded = _distance2bbox(centers, distance)
            boxes.append(decoded[keep])
            scores.append(score[keep])
            if kps is not None:
                landmarks.append(_distance2kps(centers, kps).reshape(-1, 5, 2)[keep])
        if not boxes:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, 5, 2), np.float32),
            )
        return np.concatenate(boxes), np.concatenate(scores), np.concatenate(landmarks) if landmarks else np.zeros((0, 5, 2), np.float32)

    def detect_on_canvas(self, bgr: np.ndarray, threshold: float, nms: float):
        boxes, scores, landmarks = self._forward(bgr, threshold)
        if len(boxes) == 0:
            return boxes, scores, landmarks
        xywh = boxes.copy()
        xywh[:, 2] = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
        xywh[:, 3] = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
        chosen = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), float(threshold), float(nms))
        if chosen is None or len(chosen) == 0:
            return (
                np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, 5, 2), np.float32),
            )
        chosen = np.array(chosen).reshape(-1)
        return boxes[chosen], scores[chosen], landmarks[chosen]


def _pad_to(image: np.ndarray, height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas


def _multiple_of_32(value: int) -> int:
    return max(32, int(np.ceil(value / 32.0) * 32))


def detect_faces(image: np.ndarray, detector: ScrfdDetector, cfg: dict) -> list[dict]:
    """Return faces in original-image pixels, sorted later by the caller."""
    threshold = float(cfg["threshold"])
    nms = float(cfg["nms"])
    tile = int(cfg["tile"])
    overlap = int(cfg["overlap"])
    min_px = float(cfg["min_face_px"])
    height, width = image.shape[:2]
    fixed = detector.fixed_hw
    if fixed is not None:
        tile = fixed[0]

    collected_b, collected_s, collected_k = [], [], []

    def accept(boxes, scores, landmarks, scale: float, origin_xy):
        if len(boxes) == 0:
            return
        ox, oy = origin_xy
        boxes = boxes.copy()
        landmarks = landmarks.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale + ox
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale + oy
        landmarks[:, :, 0] = landmarks[:, :, 0] / scale + ox
        landmarks[:, :, 1] = landmarks[:, :, 1] / scale + oy
        collected_b.append(boxes)
        collected_s.append(scores)
        collected_k.append(landmarks)

    # Full-frame pass. Upscale small inputs so a back-row face is still detectable.
    long_side = int(cfg["global_long_side"])
    scale = long_side / float(max(height, width))
    if fixed is not None:
        scale = min(scale, fixed[0] / float(height), fixed[1] / float(width))
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA,
    )
    if fixed is not None:
        canvas = _pad_to(resized, fixed[0], fixed[1])
    else:
        canvas = _pad_to(resized, _multiple_of_32(resized.shape[0]), _multiple_of_32(resized.shape[1]))
    boxes, scores, landmarks = detector.detect_on_canvas(canvas, threshold, nms)
    # Drop detections that fell in the padding.
    if len(boxes):
        keep = (boxes[:, 0] < resized.shape[1]) & (boxes[:, 1] < resized.shape[0])
        boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]
    accept(boxes, scores, landmarks, scale, (0.0, 0.0))

    # Tiled pass at the original resolution (or at the fixed network size).
    step = max(1, tile - int(overlap))
    if fixed is not None:
        net_h, net_w = fixed
    else:
        net_h = net_w = _multiple_of_32(tile)
    for y in range(0, height, step):
        for x in range(0, width, step):
            patch = image[y : min(y + tile, height), x : min(x + tile, width)]
            if patch.shape[0] < 32 or patch.shape[1] < 32:
                continue
            # Fit the patch into the network canvas without stretching, so
            # scale stays 1 when the patch already fits.
            canvas = _pad_to(patch, net_h, net_w) if (patch.shape[0] <= net_h and patch.shape[1] <= net_w) else None
            local_scale = 1.0
            if canvas is None:
                local_scale = min(net_h / patch.shape[0], net_w / patch.shape[1])
                fitted = cv2.resize(
                    patch,
                    (max(1, int(round(patch.shape[1] * local_scale))), max(1, int(round(patch.shape[0] * local_scale)))),
                    interpolation=cv2.INTER_AREA,
                )
                canvas = _pad_to(fitted, net_h, net_w)
            boxes, scores, landmarks = detector.detect_on_canvas(canvas, threshold, nms)
            if len(boxes):
                limit_w = patch.shape[1] * local_scale
                limit_h = patch.shape[0] * local_scale
                keep = (boxes[:, 0] < limit_w) & (boxes[:, 1] < limit_h)
                boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]
            accept(boxes, scores, landmarks, local_scale, (float(x), float(y)))
            if y + tile >= height and x + tile >= width:
                break
        if y + tile >= height:
            break

    if not collected_b:
        return []
    boxes = np.concatenate(collected_b)
    scores = np.concatenate(collected_s)
    landmarks = np.concatenate(collected_k)
    xywh = boxes.copy()
    xywh[:, 2] = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    xywh[:, 3] = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    chosen = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), threshold, nms)
    if chosen is None or len(chosen) == 0:
        return []
    chosen = np.array(chosen).reshape(-1)
    faces = []
    for i in chosen:
        box = boxes[i]
        side = min(box[2] - box[0], box[3] - box[1])
        if side < min_px:
            continue
        lm = landmarks[i]
        if not np.isfinite(lm).all() or not np.isfinite(box).all():
            continue
        if not landmarks_plausible(lm, box):
            continue
        faces.append({"bbox": box.astype(np.float32), "score": float(scores[i]), "landmarks": lm.astype(np.float32)})
    return faces


def landmarks_plausible(landmarks, bbox) -> bool:
    """Reject detections whose five points are not a face.

    Order is left eye, right eye, nose, left mouth corner, right mouth corner.
    Decorative carvings or patterns can score above a loose threshold and still fail
    this: the eyes sit above the nose, the nose above the mouth, and the
    mouth is about as wide as the eyes.
    """
    pts = np.asarray(landmarks, np.float32).reshape(5, 2)
    if not np.isfinite(pts).all():
        return False
    left_eye, right_eye, nose, left_mouth, right_mouth = pts
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 2.0:
        return False
    eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
    mouth_y = float((left_mouth[1] + right_mouth[1]) * 0.5)
    if not (eye_y + 0.05 * eye_dist < float(nose[1]) < mouth_y - 0.02 * eye_dist):
        return False
    eye_left = float(min(left_eye[0], right_eye[0]))
    eye_right = float(max(left_eye[0], right_eye[0]))
    span = max(eye_right - eye_left, 1.0)
    if float(nose[0]) < eye_left - 0.75 * span or float(nose[0]) > eye_right + 0.75 * span:
        return False
    mouth_dist = float(np.linalg.norm(right_mouth - left_mouth))
    if mouth_dist < 0.25 * eye_dist or mouth_dist > 2.8 * eye_dist:
        return False
    if mouth_y - float(nose[1]) < 0.08 * eye_dist:
        return False
    if float(nose[1]) - eye_y < 0.08 * eye_dist:
        return False
    x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
    pad = 0.35 * max(x1 - x0, y1 - y0, 1.0)
    if np.any(pts[:, 0] < x0 - pad) or np.any(pts[:, 0] > x1 + pad):
        return False
    if np.any(pts[:, 1] < y0 - pad) or np.any(pts[:, 1] > y1 + pad):
        return False
    return True

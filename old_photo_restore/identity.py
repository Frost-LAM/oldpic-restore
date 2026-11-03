"""ArcFace embeddings (buffalo_l ``w600k_r50``) and the glintr100 second opinion.

Both models expect a 112x112 RGB face, normalised with (pixel - 127.5) / 127.5.
The embedding is the L2-normalised sum of the original and its horizontal flip,
which is what the prototype used.
"""

from __future__ import annotations

import cv2
import numpy as np

from old_photo_restore.imageops import ARC_5, FFHQ_5, similarity_transform
from old_photo_restore.onnx_util import session as ort_session


class Embedder:
    def __init__(self, path: str, threads: int, device: str = "cpu"):
        self.session = ort_session(path, threads, device)
        self.input_name = self.session.get_inputs()[0].name

    def _run(self, rgb_nchw: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: rgb_nchw})[0][0]
        return np.asarray(out, np.float32).reshape(-1)

    def embed_112(self, bgr_112: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(bgr_112[:, :, ::-1])
        flipped = np.ascontiguousarray(rgb[:, ::-1, :])
        batch = np.stack([rgb, flipped]).astype(np.float32)
        batch = (batch - 127.5) / 127.5
        batch = np.transpose(batch, (0, 3, 1, 2))
        # Two forwards keep the peak activation smaller than a batch of 2 on
        # some CPU builds, and match the prototype exactly.
        emb = self._run(batch[0:1]) + self._run(batch[1:2])
        norm = float(np.linalg.norm(emb))
        if norm < 1e-8:
            return emb
        return emb / norm

    def embed_landmarks(self, bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        matrix = similarity_transform(landmarks, ARC_5)
        aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=(0, 0, 0))
        return self.embed_112(aligned)

    def embed_aligned_512(self, face512_bgr: np.ndarray) -> np.ndarray:
        matrix = similarity_transform(FFHQ_5, ARC_5)
        aligned = cv2.warpAffine(face512_bgr, matrix, (112, 112), borderValue=(0, 0, 0))
        return self.embed_112(aligned)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

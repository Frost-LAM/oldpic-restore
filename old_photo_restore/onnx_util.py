"""ONNX Runtime sessions for detection and ArcFace.

Apple Silicon (``device='mps'``) requests the CoreML execution provider and
keeps CPU as a fallback, including a retry if CoreML rejects the graph at
run time. CUDA requests the CUDA provider the same way. Anything else is CPU.
"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def providers_for(device: str, available: list[str] | None = None, *, use_coreml: bool = True) -> list[str]:
    """Provider list, most specific first, always ending in CPU.

    SCRFD's outputs are dynamic. On Apple Silicon the CoreML provider rejects
    them (``static output shape ... and inferred shape ... have different
    ranks``) and dumps a long error before we can fall back. The detector
    therefore passes ``use_coreml=False``. ArcFace is a fixed 112×112 graph
    and may still use CoreML.
    """
    have = list(available if available is not None else ort.get_available_providers())
    chosen: list[str] = []
    if device == "mps" and use_coreml and "CoreMLExecutionProvider" in have:
        chosen.append("CoreMLExecutionProvider")
    if device == "cuda" and "CUDAExecutionProvider" in have:
        chosen.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(threads))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # SCRFD's ONNX file advertises a 640-px output shape but runs dynamically.
    # The runtime warning is expected and not a failed detection.
    options.log_severity_level = 3
    return options


def _open(path: str, threads: int, providers: list[str]) -> ort.InferenceSession:
    return ort.InferenceSession(path, _options(threads), providers=providers)


class OnnxSession:
    """InferenceSession stand-in that can drop CoreML or CUDA after a failure."""

    def __init__(self, path: str | Path, threads: int, device: str = "cpu", *, use_coreml: bool = True):
        self.path = str(path)
        self.threads = int(threads)
        self.device = device
        self.providers = providers_for(device, use_coreml=use_coreml)
        try:
            self._session = _open(self.path, self.threads, self.providers)
        except Exception as exc:
            if self.providers == ["CPUExecutionProvider"]:
                raise
            print(
                f"onnx could not load with {self.providers[0]} ({exc}); using CPU",
                flush=True,
            )
            self.providers = ["CPUExecutionProvider"]
            self.device = "cpu"
            self._session = _open(self.path, self.threads, self.providers)
        self.providers = list(self._session.get_providers())

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def run(self, names, feeds):
        try:
            return self._session.run(names, feeds)
        except Exception as exc:
            if "CPUExecutionProvider" in self.providers and self.providers == ["CPUExecutionProvider"]:
                raise
            print(f"onnx run failed on {self.providers} ({exc}); retrying on CPU", flush=True)
            self.providers = ["CPUExecutionProvider"]
            self.device = "cpu"
            self._session = _open(self.path, self.threads, self.providers)
            self.providers = list(self._session.get_providers())
            return self._session.run(names, feeds)


def session(path: str | Path, threads: int, device: str = "cpu", *, use_coreml: bool = True) -> OnnxSession:
    return OnnxSession(path, threads, device, use_coreml=use_coreml)

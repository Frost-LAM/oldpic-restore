"""End-to-end CPU run on a small public-domain group photo.

The sample is the 1927 Solvay conference photograph (public domain), downloaded
into tests/data and not committed. Personal photos are never used.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/960px-Solvay_conference_1927.jpg"
FALLBACK_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Solvay_conference_1927.jpg/640px-Solvay_conference_1927.jpg"


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "old-photo-restore-smoke/1.0 (research; public-domain sample)"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _sample(tmp: Path) -> Path:
    dest = tmp / "solvay1927.jpg"
    try:
        _download(SAMPLE_URL, dest)
    except Exception:
        _download(FALLBACK_URL, dest)
    image = cv2.imread(str(dest))
    if image is None or min(image.shape[:2]) < 64:
        raise RuntimeError(f"sample download is not a readable image ({dest.stat().st_size} bytes)")
    return dest


def test_cli_restores_a_group_photo(tmp_path):
    sample = _sample(tmp_path)
    refs = tmp_path / "refs"
    refs.mkdir()
    shutil.copy(sample, refs / "solvay1927.jpg")
    output = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "old_photo_restore",
            str(sample),
            "-o",
            str(output),
            "--refs",
            str(refs),
            "--scale",
            "both",
            "--max-faces",
            "2",
            "--cpu-threads",
            "4",
        ],
        cwd=ROOT,
        check=False,
        text=True,
    )
    assert proc.returncode == 0, "CLI failed"
    expected = [
        "restored.png",
        "restored-2x.png",
        "compare-side-by-side.jpg",
        "compare-faces.jpg",
        "compare-refs.jpg",
        "faces-overview.jpg",
        "report.txt",
    ]
    for name in expected:
        path = output / name
        assert path.exists() and path.stat().st_size > 1000, name
    original = cv2.imread(str(sample))
    restored = cv2.imread(str(output / "restored.png"))
    restored_2x = cv2.imread(str(output / "restored-2x.png"))
    assert restored.shape[:2] == original.shape[:2]
    assert restored_2x.shape[0] == original.shape[0] * 2
    assert restored_2x.shape[1] == original.shape[1] * 2
    delta = abs(restored.astype("float32") - original.astype("float32")).mean()
    assert delta > 0.4
    report = (output / "report.txt").read_text(encoding="utf-8")
    assert "NAFNet-REDS-width64" in report
    assert "P1" in report and "P2" in report
    assert "weak faces:" in report
    assert "ref=" in report
    assert "identity_out=" in report

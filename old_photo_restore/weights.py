"""Download published weights into a cache directory. Nothing large is vendored."""

from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import certifi
from tqdm import tqdm


@dataclass(frozen=True)
class Weight:
    filename: str
    url: str
    min_bytes: int
    sha256: str | None = None


WEIGHTS: dict[str, Weight] = {
    "nafnet": Weight(
        "NAFNet-REDS-width64.pth",
        "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth",
        200_000_000,
        "175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3",
    ),
    "codeformer": Weight(
        "codeformer.pth",
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        300_000_000,
    ),
    "parsenet": Weight(
        "parsing_parsenet.pth",
        "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        50_000_000,
    ),
    "esrgan_x2": Weight(
        "RealESRGAN_x2plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        40_000_000,
    ),
    "esrgan_x4": Weight(
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        40_000_000,
    ),
    "buffalo_l": Weight(
        "buffalo_l.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        200_000_000,
    ),
    "antelopev2": Weight(
        "antelopev2.zip",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
        200_000_000,
    ),
}

# Files pulled out of the InsightFace packs. Detection + ArcFace are required.
# glintr100 is the second-opinion embedder used only when references are given.
ONNX_MEMBERS = {
    "det_10g.onnx": "buffalo_l",
    "w600k_r50.onnx": "buffalo_l",
    "glintr100.onnx": "antelopev2",
}


def weights_dir(explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
    elif os.environ.get("OPR_WEIGHTS_DIR"):
        path = Path(os.environ["OPR_WEIGHTS_DIR"])
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        path = root / "old-photo-restore" / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_hf_url(url: str) -> str:
    """Swap the Hugging Face host when a mirror is configured.

    ``OPR_HF_ENDPOINT`` wins over ``HF_ENDPOINT``. GitHub release URLs are
    left alone. A mirror such as ``https://hf-mirror.com`` is the practical
    fix when huggingface.co is only a couple of MB/s.
    """
    endpoint = (os.environ.get("OPR_HF_ENDPOINT") or os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    if not endpoint:
        return url
    prefix = "https://huggingface.co"
    if url.startswith(prefix + "/") or url == prefix:
        return endpoint + url[len(prefix) :]
    return url


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds do not use the system keychain, so urlopen
    # raises CERTIFICATE_VERIFY_FAILED unless we point at certifi's bundle.
    return ssl.create_default_context(cafile=certifi.where())


def _progress(total: int | None, initial: int, desc: str):
    # disable=None: bars only on a TTY. TQDM_DISABLE is still honoured.
    return tqdm(
        total=total,
        initial=initial,
        unit="B",
        unit_scale=True,
        desc=desc,
        leave=False,
        disable=None,
        file=sys.stderr,
    )


def download(weight: Weight, dest_dir: Path) -> Path:
    dest = dest_dir / weight.filename
    url = rewrite_hf_url(weight.url)
    if dest.exists() and dest.stat().st_size >= weight.min_bytes:
        if weight.sha256 and _sha256(dest) != weight.sha256:
            dest.unlink()
        else:
            return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_at = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "old-photo-restore/1.1"}
    if resume_at > 0:
        headers["Range"] = f"bytes={resume_at}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120, context=_ssl_context()) as response:
            status = getattr(response, "status", None) or response.getcode()
            if resume_at > 0 and status != 206:
                # The server ignored Range. Start this attempt from scratch.
                resume_at = 0
                tmp.write_bytes(b"")
            length = int(response.headers.get("Content-Length") or 0)
            total = (resume_at + length) if length else None
            mode = "ab" if resume_at and status == 206 else "wb"
            with tmp.open(mode) as out, _progress(total, resume_at if mode == "ab" else 0, weight.filename) as bar:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    bar.update(len(chunk))
    except HTTPError as exc:
        if exc.code == 416 and tmp.exists() and tmp.stat().st_size >= weight.min_bytes:
            pass
        else:
            raise RuntimeError(
                f"could not download {weight.filename} from {url}: {exc}. "
                "A partial file was kept so the next run can resume."
            ) from exc
    except Exception as exc:
        kept = f" Partial file kept at {tmp}." if tmp.exists() else ""
        raise RuntimeError(f"could not download {weight.filename} from {url}: {exc}.{kept}") from exc
    if not tmp.exists():
        raise RuntimeError(f"download of {weight.filename} produced no file")
    size = tmp.stat().st_size
    if size < weight.min_bytes:
        raise RuntimeError(
            f"download of {weight.filename} looks truncated ({size} bytes). "
            f"The partial file was kept at {tmp} so the next run can resume."
        )
    if weight.sha256:
        got = _sha256(tmp)
        if got != weight.sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {weight.filename}: {got}")
    tmp.replace(dest)
    return dest


def extract_onnx(pack: str, member: str, dest_dir: Path) -> Path:
    target = dest_dir / member
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    archive = download(WEIGHTS[pack], dest_dir)
    with zipfile.ZipFile(archive) as zipped:
        matches = [name for name in zipped.namelist() if Path(name).name == member]
        if not matches:
            raise RuntimeError(f"{archive.name} does not contain {member}")
        with zipped.open(matches[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def ensure_core(dest_dir: Path) -> dict[str, Path]:
    paths = {name: download(WEIGHTS[name], dest_dir) for name in ("nafnet", "codeformer", "parsenet", "esrgan_x2", "esrgan_x4")}
    paths["det"] = extract_onnx("buffalo_l", "det_10g.onnx", dest_dir)
    paths["arcface"] = extract_onnx("buffalo_l", "w600k_r50.onnx", dest_dir)
    return paths


def ensure_glint(dest_dir: Path) -> Path:
    return extract_onnx("antelopev2", "glintr100.onnx", dest_dir)


def prefetch(dest_dir: Path, include_glint: bool = True) -> dict[str, Path]:
    paths = ensure_core(dest_dir)
    if include_glint:
        paths["glint"] = ensure_glint(dest_dir)
    return paths

#!/usr/bin/env bash
# Apple Silicon (M1–M4) environment: PyTorch with MPS, onnxruntime with CoreML.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "setup_mac.sh is for macOS. On Linux run: bash scripts/setup_linux.sh"
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This Mac is $(uname -m), not Apple Silicon. MPS is not available."
  echo "Install a CPU environment instead: python3 -m venv .venv && source .venv/bin/activate && pip install torch && pip install -e ."
  exit 1
fi

PY="${PYTHON:-python3}"
"$PY" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python 3.10 or newer is required, found {sys.version.split()[0]}")
print(f"Using Python {sys.version.split()[0]}")
PY

"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-mac.txt
python -m pip install -e .

python - <<'PY'
import torch
import onnxruntime as ort
print(f"torch {torch.__version__}")
ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print(f"mps available: {ok}")
print("onnx providers:", ", ".join(ort.get_available_providers()))
if not ok:
    raise SystemExit("MPS is not available. The torch wheel may not be the macOS arm64 build.")
PY

echo "Ready. Example: old-photo-restore photo.jpg -o out/"

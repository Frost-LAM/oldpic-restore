#!/usr/bin/env bash
# Linux CPU environment. For an NVIDIA GPU, see requirements-linux-cuda.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

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
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-linux.txt
python -m pip install -e .

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda {torch.cuda.is_available()}")
PY

echo "Ready. Example: old-photo-restore photo.jpg -o out/ --device cpu"
echo "NVIDIA GPU instead: pip install -r requirements-linux-cuda.txt, then --device cuda"

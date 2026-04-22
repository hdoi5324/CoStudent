#!/usr/bin/env bash
# Create a local .venv with uv (torch first, then cvpods editable build needs torch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

PY="${PYTHON_VERSION:-3.9}"
echo "Using Python ${PY} (set PYTHON_VERSION to override, e.g. PYTHON_VERSION=3.8)."

uv venv --python "${PY}" .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Pin NumPy 1.x for compatibility with older detection code; install before torch resolves deps.
uv pip install "numpy>=1.19,<2"

# Install the requested CUDA 12.8 PyTorch wheels.
uv pip install "torch==2.8.0+cu128" "torchvision==0.23.0+cu128" \
  --extra-index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
# setup.py imports torch at top level; isolated builds do not see the venv's torch.
uv pip install -e . --no-build-isolation

echo
echo "Done. Activate with: source .venv/bin/activate"

#!/usr/bin/env bash
# imgsearch setup for Linux / macOS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV=0
CPU=0

for arg in "$@"; do
  case "$arg" in
    --venv) VENV=1 ;;
    --cpu) CPU=1 ;;
  esac
done

if [ "$VENV" = "1" ]; then
  [ -d .venv ] || "$PYTHON" -m venv .venv
  PYTHON="$ROOT/.venv/bin/python"
fi

echo "== imgsearch setup =="
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,9), "Python 3.9+ required"'

if "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "torch: already installed"
elif [ "$CPU" = "1" ]; then
  "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
  "$PYTHON" -m pip install torch torchvision
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo
echo "Next: $PYTHON scripts/download_models.py"
echo "Then: $PYTHON app.py"

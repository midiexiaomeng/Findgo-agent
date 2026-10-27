#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python app.py
else
  exec python3 app.py
fi

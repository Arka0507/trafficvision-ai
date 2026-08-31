#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run ./setup_unix.sh first."
  exit 1
fi
source .venv/bin/activate
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

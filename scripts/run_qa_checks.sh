#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "[1/5] Environment verification"
"$PYTHON_BIN" scripts/verify_environment.py

echo "[2/5] Unit tests"
"$PYTHON_BIN" -m unittest discover -s tests -v

echo "[3/5] Cleanup Python cache artifacts"
find src tests scripts -type d -name "__pycache__" -prune -exec rm -rf {} +
find src tests scripts -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

if git ls-files | grep -Eq "__pycache__/|\.py[co]$"; then
  echo "Tracked Python cache artifacts detected in git index."
  exit 1
fi

echo "[4/5] Compile check"
"$PYTHON_BIN" -m compileall src

echo "[5/5] Post-QA cleanup"
find src tests scripts -type d -name "__pycache__" -prune -exec rm -rf {} +
find src tests scripts -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

echo "QA checks passed."

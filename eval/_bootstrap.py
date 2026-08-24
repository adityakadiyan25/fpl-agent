"""Ensure repo root and eval/ are on sys.path (run eval scripts from repo root)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval"
for path in (ROOT, EVAL_DIR):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

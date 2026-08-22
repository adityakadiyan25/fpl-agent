"""Ensure repo root and scripts/ are on sys.path (run scripts from repo root)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

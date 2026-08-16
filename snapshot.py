"""Save a GW1 snapshot: FPL API dumps, my squad, and the projected score."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests

# --- 1. Create snapshots/gw1/ if it doesn't exist ---
out_dir = Path("snapshots/gw1")
out_dir.mkdir(parents=True, exist_ok=True)

saved = []

# --- 2. Download bootstrap-static and fixtures; save as JSON in that folder ---
downloads = [
    ("https://fantasy.premierleague.com/api/bootstrap-static/", "bootstrap-static.json"),
    ("https://fantasy.premierleague.com/api/fixtures/", "fixtures.json"),
]
for url, filename in downloads:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest = out_dir / filename
    dest.write_text(json.dumps(resp.json(), indent=2), encoding="utf-8")
    saved.append(dest)

# --- 3. Copy my_team.json into the folder ---
team_dest = out_dir / "my_team.json"
shutil.copy2("my_team.json", team_dest)
saved.append(team_dest)

# --- 4. Write prediction.json (GW1 projected score snapshot) ---
prediction = {
    "gameweek": 1,
    "projected_score": 33.2,
    "method": "sum of ep_next for XI, captain doubled",
    "created_at": datetime.now(timezone.utc).isoformat(),
}
pred_dest = out_dir / "prediction.json"
pred_dest.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
saved.append(pred_dest)

# --- 5. Print what was saved and where ---
print(f"Saved {len(saved)} files to {out_dir.resolve()}:")
for path in saved:
    print(f"  {path}")

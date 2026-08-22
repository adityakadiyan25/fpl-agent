"""Fetch each player's past-season totals and save snapshots/gw1/history_past.json."""

import _bootstrap  # noqa: F401

import json
import time

import requests

from fpl_agent.data import load_snapshot, snapshot_dir

SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{id}/"
SLEEP_S = 0.5
PROGRESS_EVERY = 50

# Fields to keep from each history_past season row
SEASON_FIELDS = (
    "season_name",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "bonus",
    "total_points",
)

# --- 1. Load all player IDs from the frozen GW1 bootstrap ---
bootstrap = load_snapshot(1)["bootstrap"]
player_ids = [p["id"] for p in bootstrap["elements"]]
total = len(player_ids)
print(f"Fetching history_past for {total} players (~{total * SLEEP_S / 60:.0f} min)...")

# --- 2–3. GET element-summary for each id; keep only history_past ---
# 0.5s between calls; failures are logged and skipped
history_past = {}
failures = []

for i, pid in enumerate(player_ids, start=1):
    try:
        resp = requests.get(SUMMARY_URL.format(id=pid), timeout=30)
        resp.raise_for_status()
        seasons = []
        for row in resp.json().get("history_past") or []:
            seasons.append({field: row.get(field) for field in SEASON_FIELDS})
        history_past[str(pid)] = seasons
    except Exception as exc:
        failures.append(pid)
        print(f"  FAIL id={pid}: {exc}")

    # --- 5. Progress every 50 players ---
    if i % PROGRESS_EVERY == 0 or i == total:
        print(f"Progress: {i}/{total}  saved={len(history_past)}  failed={len(failures)}")

    if i < total:
        time.sleep(SLEEP_S)

# --- 4. Save keyed by player id ---
out_path = snapshot_dir(1) / "history_past.json"
out_path.write_text(json.dumps(history_past, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {len(history_past)} players to {out_path}")
if failures:
    print(f"Skipped {len(failures)} failures: {failures}")

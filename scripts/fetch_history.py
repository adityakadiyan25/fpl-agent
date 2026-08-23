"""Fetch each player's past-season totals into snapshots/gwN/history_past.json."""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
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
    "saves",
    "defensive_contribution",
)


def _most_recent_row(seasons):
    if not seasons:
        return None
    return seasons[-1]


def _coverage_report(history_past):
    with_rows = 0
    with_saves = 0
    with_defcon = 0
    for seasons in history_past.values():
        if not seasons:
            continue
        with_rows += 1
        last = _most_recent_row(seasons)
        if last is None:
            continue
        if "saves" in last and last.get("saves") is not None:
            with_saves += 1
        dc = last.get("defensive_contribution")
        if dc is not None and dc > 0:
            with_defcon += 1
    return with_rows, with_saves, with_defcon


def main():
    parser = argparse.ArgumentParser(
        description="Fetch element-summary history_past into a GW snapshot"
    )
    parser.add_argument("--gw", type=int, required=True, help="snapshot gameweek")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing history_past.json",
    )
    args = parser.parse_args()

    out_path = snapshot_dir(args.gw) / "history_past.json"
    if out_path.exists() and not args.force:
        print(
            f"Refusing to overwrite {out_path}. Pass --force to refresh.",
            file=sys.stderr,
        )
        sys.exit(1)

    bootstrap = load_snapshot(args.gw)["bootstrap"]
    player_ids = [p["id"] for p in bootstrap["elements"]]
    total = len(player_ids)
    print(f"Fetching history_past for {total} players (~{total * SLEEP_S / 60:.0f} min)...")

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

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"Progress: {i}/{total}  saved={len(history_past)}  failed={len(failures)}")

        if i < total:
            time.sleep(SLEEP_S)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(history_past, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(history_past)} players to {out_path}")
    if failures:
        print(f"Skipped {len(failures)} failures: {failures}")

    with_rows, with_saves, with_defcon = _coverage_report(history_past)
    print(
        f"Coverage: {with_rows} with history rows, "
        f"{with_saves} with saves on latest season, "
        f"{with_defcon} with defensive_contribution > 0 on latest season"
    )
    if with_defcon == 0:
        print(
            "WARNING: no players with defensive_contribution > 0 — the API may not "
            "expose last-season DefCon counts; the DefCon term will stay dormant.",
            file=sys.stderr,
        )
    if with_saves == 0:
        print(
            "WARNING: no players with saves on latest season; save points may stay dormant.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

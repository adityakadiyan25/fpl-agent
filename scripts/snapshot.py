"""Save a frozen GW snapshot: bootstrap-static + fixtures from the FPL API."""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone

from fpl_agent.data import fetch_fixtures, fetch_live_bootstrap, snapshot_dir

PREDICTION_GW1 = {
    "gameweek": 1,
    "projected_score": 33.2,
    "method": "sum of ep_next for XI, captain doubled",
}


def _file_timestamp(path):
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main():
    parser = argparse.ArgumentParser(
        description="Download and freeze FPL bootstrap + fixtures for a gameweek"
    )
    parser.add_argument("--gw", type=int, required=True, help="gameweek number (e.g. 1, 2)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if bootstrap-static.json already exists",
    )
    args = parser.parse_args()

    out_dir = snapshot_dir(args.gw)
    bootstrap_path = out_dir / "bootstrap-static.json"

    if bootstrap_path.exists() and not args.force:
        frozen_at = _file_timestamp(bootstrap_path)
        print(
            f"Refusing to refresh {bootstrap_path}: snapshot frozen at {frozen_at}.\n"
            f"Pass --force to re-download (this moves the goalposts).",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    downloads = [
        (fetch_live_bootstrap, "bootstrap-static.json"),
        (fetch_fixtures, "fixtures.json"),
    ]
    for fetch, filename in downloads:
        dest = out_dir / filename
        dest.write_text(json.dumps(fetch(), indent=2), encoding="utf-8")
        saved.append(dest)

    if args.gw == 1:
        team_path = out_dir / "my_team.json"
        if not team_path.exists():
            print(
                f"GW1 snapshot requires {team_path} (export your team there first).",
                file=sys.stderr,
            )
            sys.exit(1)
        saved.append(team_path)

        prediction = {
            **PREDICTION_GW1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        pred_dest = out_dir / "prediction.json"
        pred_dest.write_text(json.dumps(prediction, indent=2) + "\n", encoding="utf-8")
        saved.append(pred_dest)

    print(f"Saved {len(saved)} files to {out_dir.resolve()}:")
    for path in saved:
        print(f"  {path}")

    if args.gw >= 2:
        print(f"Next: python3 scripts/fetch_my_team.py --gw {args.gw}")


if __name__ == "__main__":
    main()


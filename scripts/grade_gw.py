"""Grade frozen GW predictions against immutable per-GW live actuals."""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone

import requests

from fpl_agent.data import (
    build_players,
    fetch_entry_picks,
    fetch_live_bootstrap,
    load_history,
    load_snapshot,
    snapshot_dir,
)
from fpl_agent.metrics import fmt_metric, gw_metrics, paired_pids, point_errors, xi_score
from fpl_agent.projections import project

LIVE_URL = "https://fantasy.premierleague.com/api/event/{gw}/live/"

METRIC_ROWS = ("mae", "bias", "rmse", "spearman", "p_at_11", "haul_recall")


def _section(title, provisional):
    prefix = "PROVISIONAL " if provisional else ""
    return f"=== {prefix}{title} ==="


def _event(bootstrap, gw):
    events = bootstrap.get("events") or []
    return next((e for e in events if e.get("id") == gw), None)


def fetch_gw_live(gw):
    resp = requests.get(LIVE_URL.format(gw=gw), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _live_maps(live_payload):
    actual = {}
    minutes = {}
    for el in live_payload.get("elements") or []:
        stats = el.get("stats") or {}
        actual[el["id"]] = int(stats.get("total_points") or 0)
        minutes[el["id"]] = int(stats.get("minutes") or 0)
    return actual, minutes


def _v2_detail(v2_proj, actual, minutes, players):
    errors = point_errors(v2_proj, actual, minutes=minutes, played_only=True)
    pids = paired_pids(v2_proj, actual, minutes=minutes, played_only=True)
    detail = sorted(
        (
            (
                actual[pid] - v2_proj[pid],
                players[pid]["web_name"],
                v2_proj[pid],
                actual[pid],
                minutes[pid],
            )
            for pid in pids
        ),
        key=lambda row: row[0],
    )
    return detail


def _print_metrics_table(v0_metrics, v2_metrics, n, provisional):
    print(_section(f"Metrics (players with minutes > 0, n={n})", provisional))
    print(f"{'':12} {'v0':>8} {'v2':>8}")
    for key in METRIC_ROWS:
        print(
            f"{key:<12} {fmt_metric(v0_metrics[key], nd=2)} {fmt_metric(v2_metrics[key], nd=2)}"
        )
    print(f"{'n':<12} {fmt_metric(n, nd=0, width=8)} {fmt_metric(n, nd=0, width=8)}")
    print()


def _print_miss_tables(detail, provisional):
    over = sorted(detail, key=lambda row: row[0], reverse=True)[:10]
    under = detail[:10]

    print(_section("10 biggest overperforms (v2, actual >> predicted)", provisional))
    print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
    for error, name, pred, act, mins in over:
        print(f"{name:<16} {pred:>6.1f} {act:>5} {error:>+7.1f} {mins:>5}")
    print()

    print(_section("10 biggest underperforms (v2, actual << predicted)", provisional))
    print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
    for error, name, pred, act, mins in under:
        print(f"{name:<16} {pred:>6.1f} {act:>5} {error:>+7.1f} {mins:>5}")
    print()


def _metrics_for_json(metrics):
    return {key: metrics[key] for key in METRIC_ROWS}


def main():
    parser = argparse.ArgumentParser(description="Grade frozen GW predictions against live actuals")
    parser.add_argument("--gw", type=int, required=True, help="Gameweek number")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing snapshots/gwN/grade.json",
    )
    parser.add_argument(
        "--provisional",
        action="store_true",
        help="mid-GW peek while bonus is not finalised; nothing is written to grade.json",
    )
    args = parser.parse_args()
    gw = args.gw
    provisional = args.provisional

    folder = snapshot_dir(gw)
    if not (folder / "bootstrap-static.json").exists():
        print(f"no snapshot for GW {gw}")
        sys.exit(1)

    snap = load_snapshot(gw)
    players = build_players(snap["bootstrap"])

    live_bootstrap = fetch_live_bootstrap()
    event = _event(live_bootstrap, gw)
    if event is None:
        print(f"GW {gw} not found in live bootstrap.")
        sys.exit(1)

    data_checked = bool(event.get("data_checked"))
    if not data_checked and not provisional:
        print(
            f"GW {gw} bonus not finalised (data_checked=false). "
            f"Use --provisional for a mid-GW peek."
        )
        sys.exit(1)

    live_payload = fetch_gw_live(gw)
    actual, minutes = _live_maps(live_payload)

    v0_proj = {pid: p["ep_next"] for pid, p in players.items()}
    history = load_history(gw)
    v2_packed = project(players, history, snap["fixtures"], "v2", before_gw=gw)
    v2_proj = {pid: row["ep"] for pid, row in v2_packed.items()}

    v0_metrics = gw_metrics(v0_proj, actual, minutes=minutes, played_only=True)
    v2_metrics = gw_metrics(v2_proj, actual, minutes=minutes, played_only=True)
    n_played = len(paired_pids(v2_proj, actual, minutes=minutes, played_only=True))

    my_predicted = None
    if snap["prediction"] is None:
        print("Warning: no prediction.json — skipping my-score delta.")
    else:
        my_predicted = snap["prediction"].get("projected_score")

    picks_payload = fetch_entry_picks(event=gw)
    official_points = (picks_payload.get("entry_history") or {}).get("points")
    recomputed_points = 0
    for pick in picks_payload.get("picks") or []:
        multiplier = pick.get("multiplier") or 0
        if multiplier <= 0:
            continue
        recomputed_points += actual.get(pick["element"], 0) * multiplier

    shadow_path = folder / "shadow_team.json"
    shadow_projected = None
    shadow_actual = None
    if shadow_path.exists():
        shadow_payload = json.loads(shadow_path.read_text(encoding="utf-8"))
        shadow_projected = shadow_payload.get("projected_score")
        xi_ids = [p["id"] for p in shadow_payload.get("xi") or []]
        captain_id = next(
            (p["id"] for p in shadow_payload.get("xi") or [] if p.get("is_captain")),
            None,
        )
        if captain_id is not None:
            shadow_actual = xi_score(xi_ids, captain_id, actual)

    print(_section(f"My GW{gw} score", provisional))
    if my_predicted is not None:
        print(f"Predicted: {my_predicted}")
    if official_points is not None:
        print(f"Official:  {official_points}")
    if official_points is None:
        print(f"Recomputed:{recomputed_points}")
    elif recomputed_points != official_points:
        print(f"Recomputed:{recomputed_points}")
        print("(official includes autosubs / provisional bonus when they differ)")
    if my_predicted is not None and official_points is not None:
        print(f"Delta:     {official_points - my_predicted:+.1f}")
    print()

    if shadow_path.exists() and shadow_actual is not None:
        print(_section("Shadow envelope", provisional))
        print(f"Projected: {shadow_projected}")
        print(f"Actual:    {shadow_actual}")
        if shadow_projected is not None:
            print(f"Delta:     {shadow_actual - shadow_projected:+.1f}")
        if official_points is not None:
            print(
                f"Cost of not following the model: "
                f"{shadow_actual - official_points:+.1f}"
            )
        print()

    detail = _v2_detail(v2_proj, actual, minutes, players)
    _print_metrics_table(v0_metrics, v2_metrics, n_played, provisional)
    _print_miss_tables(detail, provisional)

    v2_mae = v2_metrics["mae"]
    projected_cell = my_predicted if my_predicted is not None else "—"
    official_cell = official_points if official_points is not None else "—"
    shadow_cell = shadow_actual if shadow_actual is not None else "—"
    mae_cell = f"{v2_mae:.2f}" if v2_mae is not None else "n/a"
    print(
        f"README row: | GW{gw} | {projected_cell} | {official_cell} | "
        f"{shadow_cell} | v2 MAE {mae_cell} |"
    )

    if provisional:
        return

    grade_path = folder / "grade.json"
    if grade_path.exists() and not args.force:
        print(f"Refusing to overwrite {grade_path}. Pass --force to replace.")
        sys.exit(1)

    grade_doc = {
        "gw": gw,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "data_checked": data_checked,
        "my": {
            "predicted": my_predicted,
            "official_points": official_points,
            "recomputed_points": recomputed_points,
        },
        "shadow": (
            {"projected": shadow_projected, "actual": shadow_actual}
            if shadow_path.exists()
            else None
        ),
        "metrics": {
            "v0": _metrics_for_json(v0_metrics),
            "v2": _metrics_for_json(v2_metrics),
        },
        "n_played": n_played,
    }
    grade_path.write_text(json.dumps(grade_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {grade_path}")


if __name__ == "__main__":
    main()

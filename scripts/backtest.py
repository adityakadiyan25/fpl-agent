"""Season replay through the LIVE projection engine + dumb baselines.

Simplification: ignore transfers and budget carryover. Each GW we pick a fresh
legal 15 + XI from the full pool at £100m, as if every week were a free-hit.

All model math comes from ``fpl_agent.projections.project``. This script only
loads historical data, adapts it via ``fpl_agent.replay``, scores with
``fpl_agent.metrics``, and picks squads with ``fpl_agent.optimize``.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import requests

from fpl_agent.metrics import fmt_metric, gw_metrics, mean_metric, xi_score
from fpl_agent.optimize import best_squad, best_xi
from fpl_agent.projections import project
from fpl_agent.replay import (
    baseline_last_season,
    baseline_template,
    baseline_xp,
    build_replay_inputs,
    replay_caveats,
)

REPO_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DATA_DIR = Path("data")
SEASON = "2025-26"
PRIOR_SEASON = "2024-25"
BUDGET = 1000
REPORTS_DIR = Path("reports")

ENGINE_MODELS = ("v0", "v1", "v2", "v3a", "v3b")
BASELINE_MODELS = ("baseline_last_season", "baseline_template", "baseline_xp")
ALL_MODELS = ENGINE_MODELS + BASELINE_MODELS

METRIC_KEYS = (
    "mae",
    "bias",
    "rmse",
    "spearman",
    "p_at_11",
    "haul_recall",
    "captain_regret",
)
VERDICT_METRICS = ("mae", "rmse", "spearman", "p_at_11", "haul_recall", "mean_xi")
# Lower is better for these; higher is better for the rest.
LOWER_BETTER = {"mae", "rmse", "captain_regret"}


def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cache_csv(rel_path):
    """Download a repo CSV once into data/."""
    dest = DATA_DIR / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO_RAW}/{rel_path}"
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_merged(rows):
    """(element_id, gw) → list of fixture rows (DGW = multiple)."""
    by_key = defaultdict(list)
    max_gw = 1
    for row in rows:
        pid = _to_int(row.get("element"))
        gw = _to_int(row.get("GW") or row.get("round"))
        if not pid or not gw:
            continue
        by_key[(pid, gw)].append(row)
        if gw > max_gw:
            max_gw = gw
    return by_key, max_gw


def sum_rows(rows, field, to_num=_to_int):
    return sum(to_num(r.get(field)) for r in rows)


def unique_fixture_rows(rows):
    """One row per fixture id. Same-fixture duplicates are data errors, not a DGW."""
    by_fx = {}
    for row in rows:
        fx = row.get("fixture") or (row.get("kickoff_time"), row.get("opponent_team"))
        if fx not in by_fx:
            by_fx[fx] = row
    return list(by_fx.values())


def prior_totals_by_code(prior_rows):
    """code → prior-season totals (stable across FPL id resets)."""
    out = {}
    for row in prior_rows:
        code = _to_int(row.get("code"))
        if not code:
            continue
        entry = {
            "season_name": "2024/25",
            "minutes": _to_int(row.get("minutes")),
            "goals_scored": _to_int(row.get("goals_scored")),
            "assists": _to_int(row.get("assists")),
            "clean_sheets": _to_int(row.get("clean_sheets")),
            "bonus": _to_int(row.get("bonus")),
            "total_points": _to_int(row.get("total_points")),
            "points_per_game": _to_float(row.get("points_per_game")),
            "saves": _to_int(row.get("saves")),
        }
        # defensive_contribution absent in 2024-25 players_raw — leave out.
        if "defensive_contribution" in row and row.get("defensive_contribution") not in (
            None,
            "",
        ):
            entry["defensive_contribution"] = _to_int(row.get("defensive_contribution"))
        out[code] = entry
    return out


def git_short_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_commit_iso():
    """Stable timestamp for deterministic scoreboard JSON."""
    try:
        return subprocess.check_output(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_gws(spec, max_gw):
    """Parse '2-38' or '2,5,10' into a sorted list capped by max_gw."""
    gws = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            gws.update(range(lo, hi + 1))
        else:
            gws.add(int(part))
    return sorted(g for g in gws if 1 <= g <= max_gw)


def load_season_bundle(season):
    """per_gw JSON, fixtures JSON, prior totals keyed by bootstrap player id."""
    season_tag = season
    per_gw_path = DATA_DIR / "season" / f"per_gw_{season_tag}.json"
    fixtures_path = DATA_DIR / "season" / f"fixtures_{season_tag}.json"
    if not per_gw_path.exists():
        raise SystemExit(
            f"Missing {per_gw_path}. Run: python3 scripts/fetch_gw_history.py --gw 2"
        )
    if not fixtures_path.exists():
        raise SystemExit(
            f"Missing {fixtures_path}. Run: "
            f"python3 scripts/fetch_gw_history.py --gw 2 --include-fixtures"
        )

    per_gw = json.loads(per_gw_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))

    prior_path = cache_csv(f"{PRIOR_SEASON}/players_raw.csv")
    raw_path = cache_csv(f"{season_tag}/players_raw.csv")
    prior_by_code = prior_totals_by_code(load_csv(prior_path))
    raw_rows = load_csv(raw_path)
    code_of = {_to_int(r["id"]): _to_int(r.get("code")) for r in raw_rows}
    # Bootstrap ids in per_gw: map via bootstrap snapshot codes when available.
    # per_gw keys are bootstrap ids; resolve prior via current-season players_raw
    # only works for vaastav ids. Re-resolve using bootstrap from snapshot.
    from fpl_agent.data import load_snapshot

    snap = load_snapshot(2)
    bootstrap_code = {p["id"]: _to_int(p.get("code")) for p in snap["bootstrap"]["elements"]}
    names = {p["id"]: p.get("web_name") or "" for p in snap["bootstrap"]["elements"]}
    prior_by_pid = {}
    for pid_s in per_gw:
        pid = int(pid_s)
        code = bootstrap_code.get(pid) or code_of.get(pid)
        if code and code in prior_by_code:
            prior_by_pid[pid] = prior_by_code[code]

    max_gw = 1
    for gws in per_gw.values():
        for g in gws:
            max_gw = max(max_gw, int(g))

    return {
        "per_gw": per_gw,
        "fixtures": fixtures,
        "prior_by_pid": prior_by_pid,
        "names": names,
        "max_gw": max_gw,
        "per_gw_path": str(per_gw_path),
        "fixtures_path": str(fixtures_path),
    }


def _proj_eps(packed):
    return {pid: row["ep"] for pid, row in packed.items()}


def pick_xi(projections, players, actual):
    """Legal 15 + XI via optimizer; return (actual XI points, xi_ids, captain_id)."""
    try:
        squad = best_squad(projections, players, budget=BUDGET)
        xi, captain = best_xi(squad, projections, players)
    except SystemExit as exc:
        print(f"  solver failed: {exc}")
        return None, None, None
    return xi_score(xi, captain, actual), xi, captain


def build_model_projections(model, players, history, fixtures, gw, per_gw, bundle):
    if model in ENGINE_MODELS:
        # Truncate per-GW history windows to pre-gw for the engine (it also
        # asserts before_gw internally).
        return project(
            players,
            history,
            fixtures,
            model,
            before_gw=gw,
            per_gw_history=per_gw if model in ("v3a", "v3b") else None,
        )
    if model == "baseline_last_season":
        return baseline_last_season(players, history)
    if model == "baseline_template":
        return baseline_template(players, bundle["selected"])
    if model == "baseline_xp":
        return baseline_xp(players, bundle["xp"])
    raise ValueError(f"Unknown model {model!r}")


def _round_or_none(val, nd=4):
    if val is None:
        return None
    return round(float(val), nd)


def summarize(per_gw_rows):
    """Mean metrics + mean XI across GWs for one model."""
    summary = {}
    for key in METRIC_KEYS:
        summary[key] = _round_or_none(mean_metric([r["metrics"].get(key) for r in per_gw_rows]))
    summary["mean_xi"] = _round_or_none(mean_metric([r.get("xi_points") for r in per_gw_rows]))
    summary["n_gws"] = len(per_gw_rows)
    return summary


def verdict_block(summaries, models, baselines):
    """model vs each baseline: better/worse per metric."""
    lines = []
    engine = [m for m in models if m not in BASELINE_MODELS]
    for model in engine:
        sm = summaries.get(model) or {}
        for base in baselines:
            if base not in summaries:
                continue
            sb = summaries[base]
            bits = []
            for key in VERDICT_METRICS:
                a, b = sm.get(key), sb.get(key)
                if a is None or b is None:
                    bits.append(f"{key}=n/a")
                    continue
                if key in LOWER_BETTER:
                    tag = "better" if a < b else ("worse" if a > b else "tie")
                else:
                    tag = "better" if a > b else ("worse" if a < b else "tie")
                bits.append(f"{key}:{tag}")
            lines.append(f"{model} vs {base}: " + ", ".join(bits))
    return lines


def print_table(summaries, models):
    cols = ("mae", "bias", "rmse", "spearman", "p_at_11", "haul_recall", "captain_regret", "mean_xi")
    width = 14
    print(f"{'model':<24}" + "".join(f"{c:>{width}}" for c in cols) + f"{'n':>6}")
    print("-" * (24 + width * len(cols) + 6))
    for model in models:
        s = summaries.get(model)
        if not s:
            continue
        row = f"{model:<24}"
        for c in cols:
            row += fmt_metric(s.get(c), width=width, nd=3)
        row += f"{s.get('n_gws', 0):>6}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Unified live-engine season backtest")
    parser.add_argument("--season", default=SEASON)
    parser.add_argument(
        "--models",
        default="v2,baseline_last_season,baseline_template,baseline_xp",
        help="comma-separated models: " + ",".join(ALL_MODELS),
    )
    parser.add_argument("--gws", default="2-38", help="e.g. 2-38 or 2,5,10")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in ALL_MODELS]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Expected one of {ALL_MODELS}")

    bundle = load_season_bundle(args.season)
    per_gw = bundle["per_gw"]
    fixtures = bundle["fixtures"]
    prior_by_pid = bundle["prior_by_pid"]
    names = bundle["names"]
    gws = parse_gws(args.gws, bundle["max_gw"])
    if not gws:
        raise SystemExit(f"No GWs to run (max_gw={bundle['max_gw']})")

    # baseline_xp only if the season file carries xp
    sample_row = next(iter(next(iter(per_gw.values())).values()))
    has_xp_col = "xp" in sample_row
    if "baseline_xp" in models and not has_xp_col:
        print("baseline_xp omitted: per_gw has no xp column")
        models = [m for m in models if m != "baseline_xp"]

    print("=== Replay caveats ===")
    for line in replay_caveats():
        print(f"  - {line}")
    print()
    print(
        f"Season {args.season}: GWs {gws[0]}–{gws[-1]} (n={len(gws)}), "
        f"models={','.join(models)}"
    )
    print(f"per_gw={bundle['per_gw_path']}")
    print(f"fixtures={bundle['fixtures_path']}")
    print()

    per_model_gws = {m: [] for m in models}
    per_gw_out = {}

    for gw in gws:
        players, history, ctx = build_replay_inputs(
            per_gw, prior_by_pid, fixtures, gw, names=names
        )
        if len(players) < 15:
            print(f"GW{gw}: only {len(players)} players, skipped")
            continue

        actual = ctx["actual"]
        minutes = ctx["minutes"]
        gw_entry = {"n_players": len(players), "models": {}}

        for model in models:
            if model == "baseline_xp" and not ctx["has_xp"]:
                continue
            packed = build_model_projections(
                model, players, history, fixtures, gw, per_gw, ctx
            )
            eps = _proj_eps(packed)
            xi_pts, xi_ids, _captain = pick_xi(packed, players, actual)
            metrics = gw_metrics(
                eps,
                actual,
                minutes=minutes,
                played_only=True,
                squad_ids=xi_ids,
            )
            row = {
                "gw": gw,
                "metrics": {k: metrics.get(k) for k in METRIC_KEYS},
                "xi_points": xi_pts,
                "n_played": sum(1 for m in minutes.values() if m > 0),
            }
            per_model_gws[model].append(row)
            gw_entry["models"][model] = {
                "metrics": {k: _round_or_none(metrics.get(k)) for k in METRIC_KEYS},
                "xi_points": _round_or_none(xi_pts, 2),
            }

        per_gw_out[str(gw)] = gw_entry
        bits = []
        for model in models:
            rows = per_model_gws[model]
            if rows and rows[-1]["gw"] == gw:
                xi = rows[-1]["xi_points"]
                mae = rows[-1]["metrics"]["mae"]
                bits.append(
                    f"{model} mae={mae:.2f} xi={xi:.1f}"
                    if mae is not None and xi is not None
                    else f"{model}=?"
                )
        print(f"GW{gw:2d}  " + "  |  ".join(bits))

    summaries = {m: summarize(rows) for m, rows in per_model_gws.items() if rows}
    print()
    print_table(summaries, models)

    baselines_run = [m for m in models if m in BASELINE_MODELS and m in summaries]
    print()
    print("=== Verdict (model vs baselines) ===")
    for line in verdict_block(summaries, models, baselines_run):
        print(line)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"scoreboard_{args.season}.json"
    scoreboard = {
        "generated_at": git_commit_iso(),
        "git_sha": git_short_sha(),
        "season": args.season,
        "gws": gws,
        "models": models,
        "caveats": replay_caveats(),
        "per_model_summary": summaries,
        "per_gw": per_gw_out,
        "verdict": verdict_block(summaries, models, baselines_run),
    }
    out_path.write_text(
        json.dumps(scoreboard, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

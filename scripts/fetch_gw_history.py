"""Build per-GW actuals JSON from cached vaastav merged_gw.csv."""

import _bootstrap  # noqa: F401

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from backtest import SEASON, cache_csv, index_merged, load_csv, sum_rows, unique_fixture_rows
from backtest import _to_float, _to_int
from fpl_agent.data import load_snapshot

OUT_DIR = Path("data/season")
PER_GW_PATH = OUT_DIR / "per_gw_2025-26.json"
FIXTURES_PATH = OUT_DIR / "fixtures_2025-26.json"
UNMATCHED_PATH = OUT_DIR / "unmatched.txt"
XGC_COLS = ("expected_goals", "expected_assists", "expected_goal_involvements")


def _require_xgc_columns(fieldnames):
    if not fieldnames:
        print("merged_gw.csv has no column headers")
        sys.exit(1)
    missing = [col for col in XGC_COLS if col not in fieldnames]
    if missing:
        print(f"merged_gw.csv missing required columns: {missing}")
        print("Actual columns:")
        for col in fieldnames:
            print(f"  {col}")
        sys.exit(1)


def _norm_name(value):
    s = (value or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_team_name_to_id(season=SEASON):
    """vaastav teams.csv name → id (same id-space as fixtures.csv)."""
    path = cache_csv(f"{season}/teams.csv")
    rows = load_csv(path)
    out = {}
    for row in rows:
        name = row.get("name") or ""
        tid = _to_int(row.get("id"))
        if name and tid:
            out[name] = tid
    return out


def _gw_stats(rows, team_name_to_id):
    """Minutes, starts, points, goals, assists + replay fields (DGW-safe)."""
    uniq = unique_fixture_rows(rows)
    first = uniq[0]
    has_starts = any(r.get("starts") not in (None, "") for r in uniq)
    minutes = sum_rows(uniq, "minutes")
    if has_starts:
        starts = sum_rows(uniq, "starts")
    else:
        starts = sum(1 for r in uniq if _to_int(r.get("minutes")) >= 60)
    team_name = first.get("team") or ""
    return {
        "minutes": minutes,
        "starts": starts,
        "points": sum_rows(uniq, "total_points"),
        "goals": sum_rows(uniq, "goals_scored"),
        "assists": sum_rows(uniq, "assists"),
        "expected_goals": round(sum_rows(uniq, "expected_goals", _to_float), 2),
        "expected_assists": round(sum_rows(uniq, "expected_assists", _to_float), 2),
        "expected_goal_involvements": round(
            sum_rows(uniq, "expected_goal_involvements", _to_float), 2
        ),
        # Replay / live-adapter fields (player-level; first unique fixture).
        "value": _to_int(first.get("value")),
        "selected": _to_int(first.get("selected")),
        "xp": round(sum_rows(uniq, "xP", _to_float), 2),
        "position": first.get("position") or "",
        "team": team_name,
        "team_id": team_name_to_id.get(team_name),
        "name": first.get("name") or "",
    }


def _build_vaastav_gw_index(merged_rows, team_name_to_id):
    """vaastav element id → {gw: stats}."""
    by_key, max_gw = index_merged(merged_rows)
    out = defaultdict(dict)
    for (vaastav_id, gw), rows in by_key.items():
        out[vaastav_id][str(gw)] = _gw_stats(rows, team_name_to_id)
    return out, max_gw


def _index_raw_players(raw_rows):
    by_code = {}
    by_web_name = defaultdict(list)
    for row in raw_rows:
        vid = _to_int(row.get("id"))
        code = _to_int(row.get("code"))
        if not vid:
            continue
        if code:
            by_code[code] = vid
        wn = _norm_name(row.get("web_name"))
        if wn:
            by_web_name[wn].append(vid)
    return by_code, by_web_name


def _match_bootstrap_players(bootstrap_elements, by_code, by_web_name):
    """Map bootstrap id → (vaastav_id, method). Unmatched returned separately."""
    matched = {}
    unmatched = []
    matched_by_code = 0
    matched_by_name = 0

    for player in bootstrap_elements:
        bid = player["id"]
        code = _to_int(player.get("code"))
        if code and code in by_code:
            matched[bid] = (by_code[code], "code")
            matched_by_code += 1
            continue

        wn = _norm_name(player.get("web_name"))
        name_hits = by_web_name.get(wn) or []
        if len(name_hits) == 1:
            matched[bid] = (name_hits[0], "name")
            matched_by_name += 1
            continue

        reason = "ambiguous name in vaastav CSV" if len(name_hits) > 1 else "not in vaastav CSV"
        unmatched.append(
            {
                "id": bid,
                "web_name": player.get("web_name") or "",
                "team": player.get("team"),
                "code": code,
                "reason": reason,
            }
        )

    return matched, unmatched, matched_by_code, matched_by_name


def _write_unmatched(unmatched, stats):
    lines = [
        "# Bootstrap players with no usable merged_gw.csv history",
        (
            f"# season={SEASON} bootstrap={stats['bootstrap_total']} "
            f"matched={stats['matched_total']} "
            f"(code={stats['matched_by_code']}, name={stats['matched_by_name']}) "
            f"unmatched={stats['unmatched_total']} "
            f"match_rate={stats['match_rate']:.1f}%"
        ),
        "",
    ]
    teams = stats.get("teams_by_id") or {}
    for row in sorted(unmatched, key=lambda r: (r["web_name"], r["id"])):
        team_name = teams.get(row["team"], str(row["team"]))
        lines.append(
            f"{row['id']}\t{row['web_name']}\t{team_name}\t{row['reason']}"
        )
    UNMATCHED_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_haaland_sanity(per_gw, elements):
    haaland = next((p for p in elements if p.get("web_name") == "Haaland"), None)
    if haaland is None:
        print("Sanity — Haaland: not found in bootstrap catalogue")
        return
    history = per_gw.get(str(haaland["id"]))
    if not history:
        print("Sanity — Haaland: no per-GW history in output")
        return
    total_xg = round(sum(gw.get("expected_goals", 0.0) for gw in history.values()), 2)
    total_goals = sum(gw.get("goals", 0) for gw in history.values())
    print(f"Sanity — Haaland: season xG={total_xg} vs goals={total_goals}")


def write_fixtures_json(season=SEASON, force=False):
    """Download season fixtures.csv → live-schema fixtures JSON (write-once)."""
    if FIXTURES_PATH.exists() and not force:
        print(f"Refusing to overwrite {FIXTURES_PATH}. Pass --force to refresh.")
        return FIXTURES_PATH
    path = cache_csv(f"{season}/fixtures.csv")
    rows = load_csv(path)
    out = []
    for row in rows:
        event = _to_int(row.get("event"))
        if not event:
            continue
        out.append(
            {
                "event": event,
                "team_h": _to_int(row.get("team_h")),
                "team_a": _to_int(row.get("team_a")),
                "team_h_difficulty": _to_int(row.get("team_h_difficulty")),
                "team_a_difficulty": _to_int(row.get("team_a_difficulty")),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(out)} fixtures → {FIXTURES_PATH}")
    sample = next((fx for fx in out if fx["event"] == 1), None)
    if sample:
        print(
            f"Spot-check GW1: team_h={sample['team_h']} team_a={sample['team_a']} "
            f"FDR h/a={sample['team_h_difficulty']}/{sample['team_a_difficulty']}"
        )
    return FIXTURES_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Build per-GW player actuals from vaastav merged_gw.csv"
    )
    parser.add_argument(
        "--gw",
        type=int,
        default=1,
        help="bootstrap snapshot gameweek for player catalogue (default 1)",
    )
    parser.add_argument(
        "--include-fixtures",
        action="store_true",
        help=f"also write {FIXTURES_PATH} from vaastav fixtures.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite fixtures JSON if it already exists",
    )
    args = parser.parse_args()

    if args.include_fixtures:
        write_fixtures_json(force=args.force)

    team_name_to_id = _load_team_name_to_id()

    merged_path = cache_csv(f"{SEASON}/gws/merged_gw.csv")
    raw_path = cache_csv(f"{SEASON}/players_raw.csv")
    merged_rows = load_csv(merged_path)
    _require_xgc_columns(merged_rows[0].keys() if merged_rows else None)
    raw_rows = load_csv(raw_path)

    snap = load_snapshot(args.gw)
    bootstrap = snap["bootstrap"]
    teams_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements = bootstrap["elements"]

    vaastav_gws, max_gw = _build_vaastav_gw_index(merged_rows, team_name_to_id)
    by_code, by_web_name = _index_raw_players(raw_rows)
    matched, unmatched, matched_by_code, matched_by_name = _match_bootstrap_players(
        elements, by_code, by_web_name
    )

    per_gw = {}
    missing_team_id = 0
    for bid, (vaastav_id, method) in matched.items():
        history = vaastav_gws.get(vaastav_id)
        if history:
            for gw_stats in history.values():
                if gw_stats.get("team_id") is None:
                    missing_team_id += 1
            per_gw[str(bid)] = history
            continue
        player = next(p for p in elements if p["id"] == bid)
        unmatched.append(
            {
                "id": bid,
                "web_name": player.get("web_name") or "",
                "team": player.get("team"),
                "code": _to_int(player.get("code")),
                "reason": f"matched by {method} but no merged_gw rows",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_GW_PATH.write_text(json.dumps(per_gw, indent=2) + "\n", encoding="utf-8")

    bootstrap_total = len(elements)
    matched_total = len(per_gw)
    unmatched_total = len(unmatched)
    match_rate = 100.0 * matched_total / bootstrap_total if bootstrap_total else 0.0

    stats = {
        "bootstrap_total": bootstrap_total,
        "matched_total": matched_total,
        "matched_by_code": matched_by_code,
        "matched_by_name": matched_by_name,
        "mapped_total": len(matched),
        "unmatched_total": unmatched_total,
        "match_rate": match_rate,
        "teams_by_id": teams_by_id,
    }
    _write_unmatched(unmatched, stats)

    print(f"Wrote {len(per_gw)} players × up to GW{max_gw} → {PER_GW_PATH}")
    print(
        f"Match rate: {matched_total}/{bootstrap_total} ({match_rate:.1f}%) "
        f"[code={matched_by_code}, name={matched_by_name}]"
    )
    if missing_team_id:
        print(f"Warning: {missing_team_id} player-GW rows missing team_id")
    if unmatched_total:
        print(f"Unmatched: {unmatched_total} → {UNMATCHED_PATH}")
    else:
        UNMATCHED_PATH.write_text(
            "\n".join(
                [
                    "# Bootstrap players with no usable merged_gw.csv history",
                    (
                        f"# season={SEASON} bootstrap={bootstrap_total} "
                        f"matched={matched_total} unmatched=0 match_rate=100.0%"
                    ),
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    _print_haaland_sanity(per_gw, elements)


if __name__ == "__main__":
    main()

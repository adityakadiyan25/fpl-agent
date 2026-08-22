"""Backtest our v1/v2-style projections against 2025-26 actuals.

Simplification (v1 of this backtest): ignore transfers and budget
carryover. Each GW we pick a fresh legal 15 + XI from the full pool
at £100m, as if every week were a free-hit.
"""

import csv
from collections import defaultdict
from pathlib import Path

import requests

from fpl_data import POSITION_LABELS
from fpl_optimize import MIN_IN_XI, MAX_IN_XI, best_squad, best_xi
from fpl_projections import _ep_from_rates, per_90_rates

# xP is FPL's ep_this scraped after the GW; it may include post-match
# info. We still use it as the official-model benchmark, as requested,
# but we never feed this GW's xP into *our* projections.

REPO_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DATA_DIR = Path("data")
SEASON = "2025-26"
PRIOR_SEASON = "2024-25"

POS_TO_TYPE = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
BUDGET = 1000


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


def _is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


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


def download_season_files():
    """Cache 2025-26 GW + player files, plus 2024-25 players_raw for prior totals."""
    merged_path = cache_csv(f"{SEASON}/gws/merged_gw.csv")
    raw_path = cache_csv(f"{SEASON}/players_raw.csv")
    prior_path = cache_csv(f"{PRIOR_SEASON}/players_raw.csv")
    return merged_path, raw_path, prior_path


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


def gw_bundle(rows):
    """Collapse a player's rows for one GW.

    Distinct fixtures (true DGW) are summed for points/minutes/xP.
    Same-fixture duplicate rows are dropped. Ownership/price are player-level
    and taken from the first unique fixture, never summed.
    """
    n_raw = len(rows)
    uniq = unique_fixture_rows(rows)
    first = uniq[0]
    return {
        "name": first.get("name") or "",
        "position": first.get("position") or "",
        "team": first.get("team") or "",
        "value": _to_int(first.get("value")),
        "was_home": _is_true(first.get("was_home")),
        "selected": _to_int(first.get("selected")),
        "points": sum_rows(uniq, "total_points"),
        "minutes": sum_rows(uniq, "minutes"),
        "goals_scored": sum_rows(uniq, "goals_scored"),
        "assists": sum_rows(uniq, "assists"),
        "clean_sheets": sum_rows(uniq, "clean_sheets"),
        "bonus": sum_rows(uniq, "bonus"),
        "xp": sum_rows(uniq, "xP", _to_float),
        "n_raw_rows": n_raw,
        "n_fixtures": len(uniq),
    }


def prior_totals_by_code(prior_rows):
    """code → 2024/25 season totals (stable across FPL id resets)."""
    out = {}
    for row in prior_rows:
        code = _to_int(row.get("code"))
        if not code:
            continue
        out[code] = {
            "season_name": "2024/25",
            "minutes": _to_int(row.get("minutes")),
            "goals_scored": _to_int(row.get("goals_scored")),
            "assists": _to_int(row.get("assists")),
            "clean_sheets": _to_int(row.get("clean_sheets")),
            "bonus": _to_int(row.get("bonus")),
            "total_points": _to_int(row.get("total_points")),
            "points_per_game": _to_float(row.get("points_per_game")),
        }
    return out


def rolling_totals(by_key, pid, before_gw):
    """Sum GWs 1..before_gw-1 only — no peeking at the GW being scored."""
    acc = {
        "minutes": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "bonus": 0,
        "points": 0,
        "n_played": 0,
        "last_minutes": 0,
        "last3_minutes": 0,
        "lagged_xp": 0.0,
        "lagged_selected": 0,
    }
    n_prior = before_gw - 1
    for gw in range(1, before_gw):
        rows = by_key.get((pid, gw))
        if not rows:
            continue
        b = gw_bundle(rows)
        acc["minutes"] += b["minutes"]
        acc["goals_scored"] += b["goals_scored"]
        acc["assists"] += b["assists"]
        acc["clean_sheets"] += b["clean_sheets"]
        acc["bonus"] += b["bonus"]
        acc["points"] += b["points"]
        if b["minutes"] > 0:
            acc["n_played"] += 1
        if gw == n_prior:
            acc["last_minutes"] = b["minutes"]
            acc["lagged_xp"] = b["xp"]
            acc["lagged_selected"] = b["selected"]
        if gw > n_prior - 3:
            acc["last3_minutes"] += b["minutes"]
    acc["n_gws"] = n_prior
    return acc


def gw_pool(by_key, gw):
    """Players with a fixture in this GW: [(pid, element_type, bundle), ...]."""
    pool = []
    for (pid, row_gw), rows in by_key.items():
        if row_gw != gw:
            continue
        b = gw_bundle(rows)
        etype = POS_TO_TYPE.get(b["position"])
        if not etype:
            continue
        pool.append((pid, etype, b))
    return pool


def load_replay():
    """Cached 2025-26 merged_gw + players_raw, and 2024-25 prior totals."""
    merged_path, raw_path, prior_path = download_season_files()
    merged = load_csv(merged_path)
    raw = load_csv(raw_path)
    prior_rows = load_csv(prior_path)
    by_key, max_gw = index_merged(merged)
    return {
        "by_key": by_key,
        "max_gw": max_gw,
        "n_rows": len(merged),
        "code_of": {_to_int(r["id"]): _to_int(r.get("code")) for r in raw},
        "web_name_of": {_to_int(r["id"]): r.get("web_name") or "" for r in raw},
        "prior_by_code": prior_totals_by_code(prior_rows),
    }


def compute_v1(etype, prior, form):
    """v1 EP from prior-season + rolling form. None if no history (no price fallback)."""
    n_gws = form["n_gws"] or 1
    prior_rates = (
        rates_from_totals(
            prior["minutes"],
            prior["goals_scored"],
            prior["assists"],
            prior["clean_sheets"],
            prior["bonus"],
            "2024/25",
        )
        if prior
        else None
    )
    form_rates = (
        rates_from_totals(
            form["minutes"],
            form["goals_scored"],
            form["assists"],
            form["clean_sheets"],
            form["bonus"],
            "2025/26",
        )
        if form["minutes"]
        else None
    )
    if form_rates is not None:
        exp_mins = min(90.0, form["minutes"] / n_gws)
        form_ep = _ep_from_rates(etype, exp_mins, form_rates)
    else:
        form_ep = None
    if prior_rates is not None:
        prior_mins = min(90.0, prior["minutes"] / 38.0)
        prior_ep = _ep_from_rates(etype, prior_mins, prior_rates)
    else:
        prior_ep = None
    if form_ep is not None and prior_ep is not None:
        w_form = min(0.75, n_gws / 12.0)
        return (1.0 - w_form) * prior_ep + w_form * form_ep
    if form_ep is not None:
        return form_ep
    if prior_ep is not None:
        return prior_ep
    return None


def apply_v2(v1, form, was_home):
    """v2 adjustments on a v1 value. None if v1 is None."""
    if v1 is None:
        return None
    n_gws = form["n_gws"] or 1
    blended = max(v1, form["lagged_xp"])
    avail = 1.0
    if n_gws >= 1 and form["last_minutes"] == 0:
        avail = 0.75
    if n_gws >= 3 and form["last3_minutes"] == 0:
        avail = 0.25
    fixture_adj = 1.08 if was_home else 0.95
    return blended * avail * fixture_adj


def rates_from_totals(minutes, goals, assists, cs, bonus, season_name):
    return per_90_rates(
        [
            {
                "season_name": season_name,
                "minutes": minutes,
                "goals_scored": goals,
                "assists": assists,
                "clean_sheets": cs,
                "bonus": bonus,
            }
        ]
    )


def project_player(etype, prior, form, now_cost, was_home, ppm_by_pos):
    """v1 blend of prior-season + rolling form, then v2 home/away + availability.

    Uses only pre-GW information: 2024-25 totals, GWs already played,
    this GW's price/home flag (known at deadline). Not this GW's xP.
    """
    v1 = compute_v1(etype, prior, form)
    if v1 is None:
        v1 = ppm_by_pos.get(etype, 0.0) * (now_cost / 10.0)
        low_confidence = True
    else:
        low_confidence = False
    return apply_v2(v1, form, was_home), low_confidence


def build_ppm(by_key, code_of, prior_by_code, pids, before_gw):
    """Price fallback from players who already have a v1 reading."""
    buckets = {1: [], 2: [], 3: [], 4: []}
    for pid, etype, now_cost in pids:
        prior = prior_by_code.get(code_of.get(pid))
        form = rolling_totals(by_key, pid, before_gw)
        ep, low = project_player(etype, prior, form, now_cost, True, {1: 0, 2: 0, 3: 0, 4: 0})
        if low or now_cost <= 0:
            continue
        buckets[etype].append((now_cost / 10.0, ep))
    ppm = {}
    for etype, pairs in buckets.items():
        ppm[etype] = sum(ep / price for price, ep in pairs) / len(pairs) if pairs else 0.0
    return ppm


def score_xi(xi_ids, captain_id, actual_points):
    """Sum ACTUAL points of 11 unique players, captain counted twice.

    Used for our XI, the ownership template, and the xP bench.
    """
    uniq = list(dict.fromkeys(xi_ids))
    if len(uniq) != 11:
        raise ValueError(f"XI must be 11 unique players, got {len(uniq)}: {uniq}")
    if captain_id not in actual_points:
        raise ValueError(f"captain {captain_id} missing actual points")
    return sum(actual_points[i] for i in uniq) + actual_points[captain_id]


def pick_optimal(projections, players, actual_points):
    """Legal 15 + XI via the optimizer; score with actual points."""
    try:
        squad = best_squad(projections, players, budget=BUDGET)
        xi, captain = best_xi(squad, projections, players)
    except SystemExit as exc:
        print(f"  solver failed: {exc}")
        return 0.0, [], None
    return score_xi(xi, captain, actual_points), xi, captain


def _formation_ok(counts):
    n = sum(counts.values())
    if n > 11:
        return False
    for etype in (1, 2, 3, 4):
        if counts[etype] > MAX_IN_XI[etype]:
            return False
    need = sum(max(0, MIN_IN_XI[et] - counts[et]) for et in (1, 2, 3, 4))
    return need <= (11 - n)


def template_xi(pool):
    """Most-owned legal XI (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD, max 3 per club).

    Greedy by `selected`. Captain = most-owned MID/FWD in the XI.
    """
    ranked = sorted(pool, key=lambda t: (-t[2]["selected"], t[0]))
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    clubs = defaultdict(int)
    picked = []
    seen = set()
    for pid, etype, b in ranked:
        if pid in seen:
            continue
        trial = dict(counts)
        trial[etype] += 1
        if not _formation_ok(trial):
            continue
        if clubs[b["team"]] >= 3:
            continue
        picked.append((pid, etype, b))
        seen.add(pid)
        counts[etype] += 1
        clubs[b["team"]] += 1
        if len(picked) == 11:
            break
    if len(picked) != 11:
        raise SystemExit(f"template XI only filled {len(picked)} players")
    attackers = [p for p in picked if p[1] in (3, 4)]
    captain = max(attackers or picked, key=lambda p: p[2]["selected"])[0]
    return [p[0] for p in picked], captain


def print_xi_audit(label, gw, xi, captain, players, bundles, actual, xp, score):
    """Dump 11 unique players, xP vs actual, DGW row counts, cost, score check."""
    print()
    print(f"=== {label} GW{gw} ===")
    print(
        f"{'Name':<22} {'Pos':<4} {'Team':<16} {'xP':>6} {'Act':>5} "
        f"{'Raw':>4} {'Fx':>3} {'£m':>5}"
    )
    print("-" * 72)
    cost = 0
    act_sum = 0
    any_dup_player = len(xi) != len(set(xi))
    any_raw_dup = False
    for pid in xi:
        p = players[pid]
        b = bundles[pid]
        cap = " (C)" if pid == captain else ""
        raw = b["n_raw_rows"]
        fx = b["n_fixtures"]
        if raw > fx:
            any_raw_dup = True
        cost += p["now_cost"]
        act_sum += actual[pid]
        print(
            f"{p['web_name']:<22} {POSITION_LABELS[p['element_type']]:<4} "
            f"{p['team_name']:<16} {xp.get(pid, 0):>6.1f} {actual[pid]:>5} "
            f"{raw:>4} {fx:>3} {p['now_cost']/10:>5.1f}{cap}"
        )
    recomputed = act_sum + actual[captain]
    print("-" * 72)
    print(f"Unique players in XI: {len(set(xi))} (list len {len(xi)})")
    print(f"Any player listed twice: {any_dup_player}")
    print(f"Any same-fixture duplicate rows in source: {any_raw_dup}")
    print(f"True DGW (n_fixtures>1) in XI: {sum(1 for i in xi if bundles[i]['n_fixtures']>1)}")
    print(f"Total cost of 11: £{cost/10:.1f}m")
    print(f"Sum of 11 actuals: {act_sum}  + captain extra {actual[captain]}  = {recomputed}")
    print(f"Printed score: {score}  match={recomputed == score}")
    print()


def main():
    replay = load_replay()
    by_key = replay["by_key"]
    max_gw = replay["max_gw"]
    code_of = replay["code_of"]
    web_name_of = replay["web_name_of"]
    prior_by_code = replay["prior_by_code"]
    print(f"Loaded {replay['n_rows']} GW rows through GW{max_gw}")

    last_gw = min(38, max_gw)
    results = []  # (gw, ours, template, xp_score_or_none)

    for gw in range(2, last_gw + 1):
        pool = [(pid, etype, b) for pid, etype, b in gw_pool(by_key, gw) if b["value"] > 0]
        bundles = {pid: b for pid, etype, b in pool}

        if not pool:
            print(f"GW{gw}: no players, skipped")
            continue

        ppm = build_ppm(
            by_key,
            code_of,
            prior_by_code,
            [(pid, etype, b["value"]) for pid, etype, b in pool],
            gw,
        )

        players = {}
        ours_proj = {}
        xp_proj = {}
        actual = {}
        xp_by_id = {}
        for pid, etype, b in pool:
            players[pid] = {
                "id": pid,
                "web_name": web_name_of.get(pid) or b["name"],
                "team": b["team"],
                "team_name": b["team"],
                "element_type": etype,
                "now_cost": b["value"],
                "can_select": True,
            }
            prior = prior_by_code.get(code_of.get(pid))
            form = rolling_totals(by_key, pid, gw)
            ep, low = project_player(etype, prior, form, b["value"], b["was_home"], ppm)
            ours_proj[pid] = {"ep": ep, "low_confidence": low}
            xp_proj[pid] = {"ep": b["xp"], "low_confidence": False}
            actual[pid] = b["points"]
            xp_by_id[pid] = b["xp"]

        our_score, our_xi, our_cap = pick_optimal(ours_proj, players, actual)
        tmpl_xi, tmpl_cap = template_xi(pool)
        tmpl_score = score_xi(tmpl_xi, tmpl_cap, actual)

        xp_missing = max(b["xp"] for _, _, b in pool) <= 0
        xp_score = None
        xp_xi = xp_cap = None
        if not xp_missing:
            xp_score, xp_xi, xp_cap = pick_optimal(xp_proj, players, actual)

        if gw == 2 and xp_xi is not None:
            print_xi_audit(
                "xP-benchmark XI (audit)",
                gw,
                xp_xi,
                xp_cap,
                players,
                bundles,
                actual,
                xp_by_id,
                xp_score,
            )
            print_xi_audit(
                "Our XI (audit)",
                gw,
                our_xi,
                our_cap,
                players,
                bundles,
                actual,
                xp_by_id,
                our_score,
            )

        results.append((gw, our_score, tmpl_score, xp_score))
        xp_txt = f"{xp_score:5.1f}" if xp_score is not None else "    —"
        print(
            f"GW{gw:2d}  ours {our_score:5.1f}  |  template {tmpl_score:5.1f}  |  xP {xp_txt}"
        )

    print()
    print(f"{'GW':<6} {'ours':>8} {'template':>10} {'xP-bench':>10}")
    print("-" * 38)
    for gw, ours, tmpl, xp in results:
        xp_txt = f"{xp:10.1f}" if xp is not None else f"{'—':>10}"
        print(f"{gw:<6} {ours:>8.1f} {tmpl:>10.1f} {xp_txt}")
    print("-" * 38)

    def mean_col(lo, hi, idx):
        vals = []
        for gw, ours, tmpl, xp in results:
            if not (lo <= gw <= hi):
                continue
            val = (ours, tmpl, xp)[idx]
            if val is None:
                continue
            vals.append(val)
        return (sum(vals) / len(vals) if vals else 0.0), len(vals)

    def report(label, lo, hi):
        o, n = mean_col(lo, hi, 0)
        t, _ = mean_col(lo, hi, 1)
        x, nx = mean_col(lo, hi, 2)
        ours_on_xp = [
            ours for gw, ours, tmpl, xp in results
            if lo <= gw <= hi and xp is not None
        ]
        print(f"{label} (n={n}):  ours {o:.2f}  |  template {t:.2f}")
        if nx:
            print(
                f"  xP-present (n={nx}): ours {sum(ours_on_xp)/len(ours_on_xp):.2f}  "
                f"|  xP-bench {x:.2f}"
            )

    print()
    report("Season avg GW2–38", 2, 38)
    report("First half GW2–19", 2, 19)
    report("Second half GW20–38", 20, 38)


if __name__ == "__main__":
    main()

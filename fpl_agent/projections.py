"""Expected-points models: v0–v2, v3a/v3b (appearance-weighted minutes engine)."""

import json
from pathlib import Path

LAST_SEASON = "2025/26"
PER_GW_PATH = Path("data/season/per_gw_2025-26.json")
GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CS_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
# FDR 1 (easy) → boost; FDR 5 (hard) → haircut. FDR already encodes home/away.
FDR_MULT = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.90, 5: 0.80}
# Ownership at which the crowd floor reaches full ep_next / p_play
CROWD_FLOOR_PCT = 20.0
STARTER_START_RATE = 0.6
STARTER_MINS = 85.0
ROTATION_MINS = 65.0
P_PLAY_START_WEIGHT = 0.65
MIN_RATE_MINUTES = 90
FULL_RATE_MINUTES = 450
XGC_COLS = ("expected_goals", "expected_assists", "expected_goal_involvements")
# Rough active-manager count for vaastav `selected` → crowd_p in backtests
TOTAL_ENTRIES_EST = 11_000_000
# Provisional one-GW projection MAE from diagnose; update after grading.
MODEL_ERROR_MAE = 2.1


def assert_pre_gw(feature_gws, target_gw):
    """Raise if any feature row uses the target GW or later (train/test leakage)."""
    bad = [g for g in feature_gws if g >= target_gw]
    if bad:
        raise AssertionError(
            f"leakage: feature GWs {bad} >= target GW {target_gw}"
        )


def _season_rows(player, history):
    """Accept a per-player season list or the full id → seasons dict."""
    if isinstance(history, dict):
        return history.get(player["id"]) or history.get(str(player["id"])) or []
    return history or []


def _last_season_row(rows):
    """Prefer 2025/26; otherwise the most recent season in the list."""
    last = next((row for row in rows if row.get("season_name") == LAST_SEASON), None)
    if last is None and rows:
        last = rows[-1]
    return last


def _p90(count, minutes):
    if minutes < MIN_RATE_MINUTES:
        return None
    return 90.0 * float(count) / minutes


def per_90_rates(history):
    """Per-90 goals/assists/CS/bonus from one player's history_past list.

    Returns None if there is no season to use (the ~83 with no history).
    """
    last = _last_season_row(history or [])
    if last is None:
        return None
    minutes = last.get("minutes") or 0
    goals_p90 = _p90(last.get("goals_scored") or 0, minutes)
    if goals_p90 is None:
        return None
    return {
        "season_name": last.get("season_name"),
        "minutes": minutes,
        "goals_p90": goals_p90,
        "assists_p90": _p90(last.get("assists") or 0, minutes) or 0.0,
        "cs_p90": _p90(last.get("clean_sheets") or 0, minutes) or 0.0,
        "bonus_p90": _p90(last.get("bonus") or 0, minutes) or 0.0,
    }


def expected_minutes(player, history):
    """min(90, last_season_minutes / 38). None if the player has no history."""
    last = _last_season_row(_season_rows(player, history))
    if last is None:
        return None
    return min(90.0, (last.get("minutes") or 0) / 38.0)


def fixture_adjustment(player, fixtures, teams=None, target_gw=1):
    """FDR multiplier for target_gw (summed if a double). Unmatched → 1.0."""
    _ = teams
    team_id = player["team"]
    matched = [
        fx
        for fx in fixtures
        if fx.get("event") == target_gw
        and (fx["team_h"] == team_id or fx["team_a"] == team_id)
    ]
    if not matched:
        name = player.get("web_name") or f"id={player.get('id')}"
        print(f"Warning: no GW{target_gw} fixture matched for {name}")
        return 1.0
    adj = 0.0
    for fx in matched:
        fdr = fx["team_h_difficulty"] if fx["team_h"] == team_id else fx["team_a_difficulty"]
        adj += FDR_MULT.get(fdr, 1.0)
    return adj


def _availability(player):
    """0–1 chance they play. Null chance + available status → 1.0."""
    chance = player.get("chance_of_playing")
    if chance is not None:
        return chance / 100.0
    status = player.get("status") or "a"
    if status in ("u", "n", "i", "s"):
        return 0.0
    if status == "d":
        return 0.75
    return 1.0


def _crowd_floor(player):
    """Ownership-weighted ep_next floor (full floor at CROWD_FLOOR_PCT)."""
    return _crowd_play_prob(player=player) * (player.get("ep_next") or 0.0)


def _crowd_play_prob(player=None, selected_by_percent=None, selected_count=None):
    """0–1 play probability from ownership (same scale as v2 crowd floor)."""
    if selected_by_percent is not None:
        pct = float(selected_by_percent)
    elif selected_count is not None:
        pct = 100.0 * float(selected_count) / TOTAL_ENTRIES_EST
    elif player is not None:
        pct = float(player.get("selected_by_percent") or 0.0)
    else:
        pct = 0.0
    return min(1.0, pct / CROWD_FLOOR_PCT)


def _last_appearances(per_gw_history, before_gw, window=10):
    """Last N GWs with minutes>0 before before_gw (appearance-based, not calendar)."""
    if not per_gw_history:
        return []
    apps = []
    feature_gws = []
    for gw in range(before_gw - 1, 0, -1):
        row = per_gw_history.get(str(gw))
        if row is None:
            row = per_gw_history.get(gw)
        if not row:
            continue
        if (row.get("minutes") or 0) > 0:
            apps.append({**row, "_gw": gw})
            feature_gws.append(gw)
            if len(apps) >= window:
                break
    assert_pre_gw(feature_gws, before_gw)
    return apps


def expected_minutes_v2(player, per_gw_history, before_gw=1, window=10):
    """Appearance window → start_rate, p_play, and conditional play minutes."""
    apps = _last_appearances(per_gw_history, before_gw, window)
    crowd_p = _crowd_play_prob(player=player)
    return expected_minutes_v2_from_apps(apps, crowd_p)


def expected_minutes_v2_from_apps(apps, crowd_p):
    """Core v3 minutes engine from pre-collected appearance rows."""
    if apps:
        start_rate = sum(int(a.get("starts") or 0) for a in apps) / len(apps)
    else:
        start_rate = None
    if start_rate is None:
        p_play = crowd_p
    else:
        p_play = (
            P_PLAY_START_WEIGHT * start_rate
            + (1.0 - P_PLAY_START_WEIGHT) * crowd_p
        )
    if start_rate is None or start_rate >= STARTER_START_RATE:
        play_minutes = STARTER_MINS
    else:
        play_minutes = ROTATION_MINS
    return {
        "start_rate": start_rate,
        "p_play": p_play,
        "play_minutes": play_minutes,
        "n_appearances": len(apps),
    }


def _career_window_from_pgh(pgh, before_gw):
    """Sum pre-target-GW rows from one player's per_gw history dict."""
    totals = {
        "minutes": 0,
        "goals": 0,
        "assists": 0,
        "clean_sheets": 0,
        "bonus": 0,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "xg_gws": 0,
    }
    feature_gws = []
    if not pgh:
        return totals, feature_gws
    for gw in range(1, before_gw):
        row = pgh.get(str(gw))
        if row is None:
            row = pgh.get(gw)
        if not row:
            continue
        feature_gws.append(gw)
        totals["minutes"] += row.get("minutes") or 0
        totals["goals"] += row.get("goals") or 0
        totals["assists"] += row.get("assists") or 0
        totals["clean_sheets"] += row.get("clean_sheets") or 0
        totals["bonus"] += row.get("bonus") or 0
        xg = row.get("expected_goals")
        xa = row.get("expected_assists")
        if xg is not None and xa is not None:
            totals["expected_goals"] += float(xg)
            totals["expected_assists"] += float(xa)
            totals["xg_gws"] += 1
    assert_pre_gw(feature_gws, before_gw)
    return totals, feature_gws


def _observed_rates_from_totals(totals, use_xg=False):
    """Unshrunk per-90 rates from career-window totals. None if minutes < 90."""
    minutes = totals["minutes"]
    if minutes < MIN_RATE_MINUTES:
        return None
    if use_xg and totals.get("xg_gws", 0) > 0:
        attack_g = totals["expected_goals"]
        attack_a = totals["expected_assists"]
    else:
        attack_g = totals["goals"]
        attack_a = totals["assists"]
    goals_p90 = _p90(attack_g, minutes)
    if goals_p90 is None:
        return None
    cs_p90 = _p90(totals.get("clean_sheets") or 0, minutes)
    bonus_p90 = _p90(totals.get("bonus") or 0, minutes)
    return {
        "minutes": minutes,
        "goals_p90": goals_p90,
        "assists_p90": _p90(attack_a, minutes) or 0.0,
        "cs_p90": cs_p90 or 0.0,
        "bonus_p90": bonus_p90 or 0.0,
    }


def _blend_rates(observed, prior):
    """Blend observed toward prior; weight = minutes/450 (observed minutes)."""
    if observed is None and prior is None:
        return None
    if observed is None:
        return dict(prior)
    if prior is None:
        return dict(observed)
    minutes = observed["minutes"]
    w = min(1.0, minutes / FULL_RATE_MINUTES)
    out = {}
    for key in ("goals_p90", "assists_p90", "cs_p90", "bonus_p90"):
        out[key] = w * observed[key] + (1.0 - w) * prior[key]
    out["minutes"] = minutes
    return out


def stable_rates_from_totals(totals, prior_rates, use_xg=False, fallback_counter=None):
    """Stable per-90 rates with 90/450-minute rules and optional xG attack leg."""
    use_xg_attack = use_xg and totals.get("xg_gws", 0) > 0
    if use_xg and not use_xg_attack and totals.get("minutes", 0) > 0:
        if fallback_counter is not None:
            fallback_counter["v3b"] = fallback_counter.get("v3b", 0) + 1
    observed = _observed_rates_from_totals(totals, use_xg=use_xg_attack)
    return _blend_rates(observed, prior_rates)


def _prior_rates_for_player(player, history):
    """history_past per-90 as the price/stat prior for shrinkage."""
    return per_90_rates(_season_rows(player, history))


def _stable_rates_for_player(
    player, history, per_gw_history, before_gw, use_xg=False, fallback_counter=None
):
    pid = player["id"]
    pgh = {}
    if per_gw_history:
        pgh = per_gw_history.get(str(pid)) or per_gw_history.get(pid) or {}
    totals, _ = _career_window_from_pgh(pgh, before_gw)
    prior = _prior_rates_for_player(player, history)

    if totals["minutes"] >= MIN_RATE_MINUTES:
        rates = stable_rates_from_totals(
            totals, prior, use_xg=use_xg, fallback_counter=fallback_counter
        )
        if rates is not None:
            cs_p90 = rates.get("cs_p90") or 0.0
            bonus_p90 = rates.get("bonus_p90") or 0.0
            if (cs_p90 == 0.0 and bonus_p90 == 0.0) and prior:
                rates["cs_p90"] = prior.get("cs_p90") or 0.0
                rates["bonus_p90"] = prior.get("bonus_p90") or 0.0
            return rates

    return prior


def load_per_gw_history(path=PER_GW_PATH):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ep_from_rates(element_type, exp_mins, rates):
    """(expected_minutes/90) × [2 appearance + scoring rates]."""
    per90 = (
        2
        + rates["goals_p90"] * GOAL_POINTS[element_type]
        + rates["assists_p90"] * 3
        + rates["cs_p90"] * CS_POINTS[element_type]
        + rates["bonus_p90"]
    )
    return (exp_mins / 90.0) * per90


def _project_v1(players, history):
    """Per-90 × expected minutes; price-based fallback if no history."""
    raw = {}
    history_eps_by_pos = {1: [], 2: [], 3: [], 4: []}

    for pid, player in players.items():
        rows = _season_rows(player, history)
        rates = per_90_rates(rows)
        exp_mins = expected_minutes(player, rows)
        if rates is None or exp_mins is None:
            raw[pid] = None
            continue
        ep = _ep_from_rates(player["element_type"], exp_mins, rates)
        raw[pid] = ep
        price_m = player["now_cost"] / 10.0
        if price_m > 0:
            history_eps_by_pos[player["element_type"]].append((price_m, ep))

    ppm_by_pos = {}
    for etype, pairs in history_eps_by_pos.items():
        ppm_by_pos[etype] = (
            sum(ep / price for price, ep in pairs) / len(pairs) if pairs else 0.0
        )

    out = {}
    for pid, player in players.items():
        if raw[pid] is not None:
            out[pid] = raw[pid]
            continue
        price_m = player["now_cost"] / 10.0
        out[pid] = ppm_by_pos[player["element_type"]] * price_m
    return out


def _project_v3(players, history, fixtures, per_gw_history, before_gw, use_xg=False):
    """p_play × points-if-played, then v2 fixture/availability/crowd layers."""
    raw = {}
    history_eps_by_pos = {1: [], 2: [], 3: [], 4: []}
    fallback_counter = {"v3b": 0} if use_xg else None

    for pid, player in players.items():
        pgh = {}
        if per_gw_history:
            pgh = per_gw_history.get(str(pid)) or per_gw_history.get(pid) or {}
        mins_info = expected_minutes_v2(player, pgh, before_gw)
        rates = _stable_rates_for_player(
            player,
            history,
            per_gw_history,
            before_gw,
            use_xg=use_xg,
            fallback_counter=fallback_counter,
        )
        if rates is None:
            raw[pid] = None
            continue
        base = mins_info["p_play"] * _ep_from_rates(
            player["element_type"], mins_info["play_minutes"], rates
        )
        blended = max(base, _crowd_floor(player))
        adj = fixture_adjustment(player, fixtures, teams=None, target_gw=before_gw)
        ep = blended * _availability(player) * adj
        raw[pid] = ep
        price_m = player["now_cost"] / 10.0
        if price_m > 0:
            history_eps_by_pos[player["element_type"]].append((price_m, ep))

    ppm_by_pos = {}
    for etype, pairs in history_eps_by_pos.items():
        ppm_by_pos[etype] = (
            sum(ep / price for price, ep in pairs) / len(pairs) if pairs else 0.0
        )

    out = {}
    for pid, player in players.items():
        if raw[pid] is not None:
            out[pid] = raw[pid]
            continue
        price_m = player["now_cost"] / 10.0
        out[pid] = ppm_by_pos[player["element_type"]] * price_m
    return out


def _low_confidence_flags(players, history):
    """True when the player has no usable history_past (price-fallback cohort)."""
    return {
        pid: per_90_rates(_season_rows(player, history)) is None
        for pid, player in players.items()
    }


def _pack(ep_by_id, flags):
    ep_by_id = {k: v for k, v in ep_by_id.items() if not str(k).startswith("_")}
    return {
        pid: {"ep": ep, "low_confidence": flags[pid]}
        for pid, ep in ep_by_id.items()
    }


def project(players, history, fixtures, model, before_gw=1, per_gw_history=None):
    """Return {player_id: {ep, low_confidence}} for v0–v3b."""
    flags = _low_confidence_flags(players, history)

    if model == "v0":
        return _pack({pid: player["ep_next"] for pid, player in players.items()}, flags)

    v1 = _project_v1(players, history)
    if model == "v1":
        return _pack(v1, flags)

    if model == "v2":
        out = {}
        for pid, player in players.items():
            blended = max(v1[pid], _crowd_floor(player))
            adj = fixture_adjustment(player, fixtures, teams=None, target_gw=before_gw)
            out[pid] = blended * _availability(player) * adj
        return _pack(out, flags)

    pgh = per_gw_history if per_gw_history is not None else load_per_gw_history()
    if model == "v3a":
        return _pack(_project_v3(players, history, fixtures, pgh, before_gw, use_xg=False), flags)
    if model == "v3b":
        return _pack(_project_v3(players, history, fixtures, pgh, before_gw, use_xg=True), flags)

    raise ValueError(f"Unknown model {model!r}; expected v0, v1, v2, v3a, or v3b")


def _leakage_self_test():
    """Deliberately pass a leaky window; assert_pre_gw must raise."""
    try:
        assert_pre_gw([9, 10], 10)
    except AssertionError:
        return
    raise AssertionError("assert_pre_gw failed to catch GW >= target")


if __name__ == "__main__":
    _leakage_self_test()
    print("leakage self-test: OK")

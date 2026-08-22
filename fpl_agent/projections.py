"""Expected-points models: v0–v2, v3a (appearance-weighted minutes engine)."""

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
# Rough active-manager count for vaastav `selected` → crowd_p in backtests
TOTAL_ENTRIES_EST = 11_000_000


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
    return (90.0 * count / minutes) if minutes else 0.0


def per_90_rates(history):
    """Per-90 goals/assists/CS/bonus from one player's history_past list.

    Returns None if there is no season to use (the ~83 with no history).
    """
    last = _last_season_row(history or [])
    if last is None:
        return None
    minutes = last.get("minutes") or 0
    return {
        "season_name": last.get("season_name"),
        "minutes": minutes,
        "goals_p90": _p90(last.get("goals_scored") or 0, minutes),
        "assists_p90": _p90(last.get("assists") or 0, minutes),
        "cs_p90": _p90(last.get("clean_sheets") or 0, minutes),
        "bonus_p90": _p90(last.get("bonus") or 0, minutes),
    }


def expected_minutes(player, history):
    """min(90, last_season_minutes / 38). None if the player has no history."""
    last = _last_season_row(_season_rows(player, history))
    if last is None:
        return None
    return min(90.0, (last.get("minutes") or 0) / 38.0)


def fixture_adjustment(player, fixtures, teams):
    """GW1 FDR multiplier (summed if a double). Unmatched → 1.0 + a named warning.

    `teams` is the bootstrap team list or id → team dict (FDR on the
    fixture already encodes home/away, so team strengths are unused).
    """
    _ = teams
    team_id = player["team"]
    gw1 = [
        fx
        for fx in fixtures
        if fx.get("event") == 1
        and (fx["team_h"] == team_id or fx["team_a"] == team_id)
    ]
    if not gw1:
        name = player.get("web_name") or f"id={player.get('id')}"
        print(f"Warning: no GW1 fixture matched for {name}")
        return 1.0
    adj = 0.0
    for fx in gw1:
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
    for gw in range(before_gw - 1, 0, -1):
        row = per_gw_history.get(str(gw))
        if row is None:
            row = per_gw_history.get(gw)
        if not row:
            continue
        if (row.get("minutes") or 0) > 0:
            apps.append(row)
            if len(apps) >= window:
                break
    return apps


def expected_minutes_v2(player, per_gw_history, before_gw=1, window=10):
    """Appearance window → start_rate, p_play, and conditional play minutes.

    p_play blends start_rate with crowd ownership; play_minutes is 85 for regular
    starters (start_rate ≥ 0.6) and 65 for rotation players.
    """
    apps = _last_appearances(per_gw_history, before_gw, window)
    crowd_p = _crowd_play_prob(player=player)
    return expected_minutes_v2_from_apps(apps, crowd_p)


def expected_minutes_v2_from_apps(apps, crowd_p):
    """Core v3a minutes engine from pre-collected appearance rows."""
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


def _unshrunk_rates(player, history, per_gw_history, before_gw):
    """Per-90 from current-season per-GW totals, else history_past (no prior blend)."""
    pid = player["id"]
    pgh = {}
    if per_gw_history:
        pgh = per_gw_history.get(str(pid)) or per_gw_history.get(pid) or {}

    minutes = goals = assists = 0
    for gw in range(1, before_gw):
        row = pgh.get(str(gw)) if pgh else None
        if row is None and pgh:
            row = pgh.get(gw)
        if not row:
            continue
        minutes += row.get("minutes") or 0
        goals += row.get("goals") or 0
        assists += row.get("assists") or 0

    if minutes > 0:
        rows = _season_rows(player, history)
        last = _last_season_row(rows)
        cs_p90 = bonus_p90 = 0.0
        if last:
            hist = per_90_rates([last])
            if hist:
                cs_p90 = hist["cs_p90"]
                bonus_p90 = hist["bonus_p90"]
        return {
            "season_name": LAST_SEASON,
            "minutes": minutes,
            "goals_p90": _p90(goals, minutes),
            "assists_p90": _p90(assists, minutes),
            "cs_p90": cs_p90,
            "bonus_p90": bonus_p90,
        }

    return per_90_rates(_season_rows(player, history))


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


def _project_v3a(players, history, fixtures, per_gw_history, before_gw=1):
    """p_play × points-if-played, then v2 fixture/availability/crowd layers."""
    raw = {}
    history_eps_by_pos = {1: [], 2: [], 3: [], 4: []}

    for pid, player in players.items():
        pgh = {}
        if per_gw_history:
            pgh = per_gw_history.get(str(pid)) or per_gw_history.get(pid) or {}
        mins_info = expected_minutes_v2(player, pgh, before_gw)
        rates = _unshrunk_rates(player, history, per_gw_history, before_gw)
        if rates is None:
            raw[pid] = None
            continue
        base = mins_info["p_play"] * _ep_from_rates(
            player["element_type"], mins_info["play_minutes"], rates
        )
        blended = max(base, _crowd_floor(player))
        adj = fixture_adjustment(player, fixtures, teams=None)
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
    return {
        pid: {"ep": ep, "low_confidence": flags[pid]}
        for pid, ep in ep_by_id.items()
    }


def project(players, history, fixtures, model, before_gw=1, per_gw_history=None):
    """Return {player_id: {ep, low_confidence}} for v0, v1, v2, or v3a."""
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
            adj = fixture_adjustment(player, fixtures, teams=None)
            out[pid] = blended * _availability(player) * adj
        return _pack(out, flags)

    if model == "v3a":
        pgh = per_gw_history if per_gw_history is not None else load_per_gw_history()
        return _pack(_project_v3a(players, history, fixtures, pgh, before_gw), flags)

    raise ValueError(f"Unknown model {model!r}; expected v0, v1, v2, or v3a")

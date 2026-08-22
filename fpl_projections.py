"""Expected-points models: v0 (ep_next), v1 (per-90 × minutes), v2 (v1 + adjustments)."""

LAST_SEASON = "2025/26"
GOAL_POINTS = {1: 6, 2: 6, 3: 5, 4: 4}
CS_POINTS = {1: 4, 2: 4, 3: 1, 4: 0}
# FDR 1 (easy) → boost; FDR 5 (hard) → haircut. FDR already encodes home/away.
FDR_MULT = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.90, 5: 0.80}
# Ownership at which the crowd floor reaches full ep_next
CROWD_FLOOR_PCT = 20.0


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
    sel = player.get("selected_by_percent") or 0.0
    ep_next = player.get("ep_next") or 0.0
    return ep_next * min(1.0, float(sel) / CROWD_FLOOR_PCT)


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


def project(players, history, fixtures, model):
    """Return {player_id: {ep, low_confidence}} for model v0, v1, or v2."""
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

    raise ValueError(f"Unknown model {model!r}; expected v0, v1, or v2")

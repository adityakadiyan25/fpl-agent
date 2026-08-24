"""Replay adapter: historical season rows → live ``project()`` / optimize inputs.

No model math lives here — only schema shaping.

``build_replay_inputs`` produces the same player/history shapes the live engine
reads (see ``fpl_agent.data.build_players`` / ``history_past.json``).

Data-availability gaps (also returned by ``replay_caveats()``):

- Availability flags (status / chance_of_playing) are not in historical
  per-GW rows → every replay player is status ``\"a\"`` with
  ``chance_of_playing=None``, so ``_availability`` is always 1.0.
- 2024-25 ``players_raw`` has ``saves`` but no ``defensive_contribution`` →
  the DefCon Poisson term is inert for this replay season (engine ``.get``
  defaults to 0).
- ``ep_next`` in replay is the per-GW ``xp`` column (vaastav ``xP``). That
  field is FPL's published expected points and may include post-match info;
  when absent, ``ep_next=0.0`` and the crowd floor is inert.
- Ownership is derived as ``selected / TOTAL_ENTRIES_EST * 100`` (live uses
  bootstrap ``selected_by_percent``). When ``selected`` is missing, ownership
  is 0.0.
- Per-GW ``team_id`` must match ``fixtures_*.json`` team ids (same vaastav
  season ``teams.csv`` name→id map used when building per_gw).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Allow `python3 fpl_agent/replay.py --selftest` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fpl_agent.projections import TOTAL_ENTRIES_EST, project
from fpl_agent.metrics import gw_metrics

POS_TO_TYPE = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}

_CAVEATS = (
    "availability flags unavailable historically → availability=1.0 in replay",
    "2024-25 has no defensive_contribution → DefCon term inert in this replay",
    "ep_next in replay = xP column (may be post-match); absent → crowd floor inert",
    "selected_by_percent ≈ selected / TOTAL_ENTRIES_EST (not live bootstrap %)",
    "vaastav xP is sparse mid-season (many GWs all-zero) → baseline_xp only on GWs with xP",
)


def replay_caveats():
    """Documented gaps between live snapshot inputs and historical replay."""
    return list(_CAVEATS)


def _selected_pct(selected) -> float:
    if not selected:
        return 0.0
    return 100.0 * float(selected) / float(TOTAL_ENTRIES_EST)


def _history_row(prior: Optional[dict]) -> list:
    """Shape a prior-season totals dict like one history_past season row."""
    if not prior:
        return []
    row = {"season_name": prior.get("season_name") or "2024/25"}
    for key in (
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "bonus",
        "total_points",
        "saves",
        "defensive_contribution",
        "points_per_game",
    ):
        if key in prior and prior[key] is not None:
            row[key] = prior[key]
    return [row]


def build_replay_inputs(per_gw, prior_totals, fixtures, gw, names=None):
    """Build live-shaped ``players`` + ``history`` for one gameweek.

    Args:
        per_gw: ``{str(pid): {str(gw): stats}}`` from ``per_gw_*.json``.
        prior_totals: ``{pid: season_totals_dict}`` already keyed by the same
            player ids as ``per_gw`` (caller resolves code→id).
        fixtures: live-schema fixture list (unused for construction; kept so
            callers pass the same bundle ``project`` will see).
        gw: target gameweek (projections use ``before_gw=gw``).
        names: optional ``{pid: web_name}``.

    Returns:
        ``(players, history, bundle)`` where ``bundle`` has ``actual``,
        ``minutes``, ``xp``, ``selected`` maps for metrics / baselines.
    """
    del fixtures  # construction does not need fixtures; project() does
    names = names or {}
    players = {}
    history = {}
    actual = {}
    minutes = {}
    xp_map = {}
    selected_map = {}

    gw_key = str(gw)
    for pid_s, gws in per_gw.items():
        row = gws.get(gw_key)
        if not row:
            continue
        value = int(row.get("value") or 0)
        if value <= 0:
            continue
        pos = row.get("position") or ""
        etype = POS_TO_TYPE.get(pos)
        if not etype:
            continue
        team_id = row.get("team_id")
        if team_id is None:
            continue
        pid = int(pid_s)
        xp = float(row.get("xp") or 0.0)
        selected = int(row.get("selected") or 0)
        players[pid] = {
            "id": pid,
            "web_name": names.get(pid) or row.get("name") or str(pid),
            "element_type": etype,
            "team": int(team_id),
            "team_name": row.get("team") or str(team_id),
            "now_cost": value,
            "status": "a",
            "chance_of_playing": None,
            "ep_next": xp,
            "selected_by_percent": _selected_pct(selected),
            "can_select": True,
            "news": "",
        }
        history[pid] = _history_row(prior_totals.get(pid))
        actual[pid] = int(row.get("points") or 0)
        minutes[pid] = int(row.get("minutes") or 0)
        xp_map[pid] = xp
        selected_map[pid] = selected

    return players, history, {
        "actual": actual,
        "minutes": minutes,
        "xp": xp_map,
        "selected": selected_map,
        "has_xp": any(v > 0 for v in xp_map.values()),
    }


def baseline_last_season(players, history):
    """EP = prior-season total_points / 38 (0 if no prior minutes/points)."""
    out = {}
    for pid, player in players.items():
        rows = history.get(pid) or []
        pts = float(rows[0].get("total_points") or 0) if rows else 0.0
        out[pid] = {"ep": pts / 38.0, "low_confidence": not rows}
    return out


def baseline_template(players, selected):
    """EP = ownership rank score (most-owned → highest), same units as a soft ranking.

    Reuses the old template's ownership ordering as a projections dict so the
    shared optimizer + metrics pipeline can score it. Rank score is
    ``n_players - rank`` (1 = least owned), not raw ``selected`` counts.
    """
    ranked = sorted(players, key=lambda pid: (-(selected.get(pid) or 0), pid))
    n = len(ranked)
    out = {}
    for i, pid in enumerate(ranked):
        out[pid] = {"ep": float(n - i), "low_confidence": False}
    return out


def baseline_xp(players, xp_map):
    """EP = FPL's own per-GW xP field."""
    out = {}
    for pid in players:
        out[pid] = {"ep": float(xp_map.get(pid) or 0.0), "low_confidence": False}
    return out


def _selftest():
    """Synthetic 3-player / 2-GW season through adapter → project → metrics."""
    per_gw = {
        "1": {
            "1": {
                "minutes": 90,
                "starts": 1,
                "points": 6,
                "goals": 0,
                "assists": 0,
                "expected_goals": 0.0,
                "expected_assists": 0.0,
                "expected_goal_involvements": 0.0,
                "value": 50,
                "selected": 1_000_000,
                "xp": 3.0,
                "position": "MID",
                "team": "Alpha",
                "team_id": 10,
                "name": "AlphaMid",
            },
            "2": {
                "minutes": 90,
                "starts": 1,
                "points": 2,
                "goals": 0,
                "assists": 0,
                "expected_goals": 0.0,
                "expected_assists": 0.0,
                "expected_goal_involvements": 0.0,
                "value": 50,
                "selected": 1_000_000,
                "xp": 3.0,
                "position": "MID",
                "team": "Alpha",
                "team_id": 10,
                "name": "AlphaMid",
            },
        },
        "2": {
            "1": {
                "minutes": 90,
                "starts": 1,
                "points": 8,
                "goals": 1,
                "assists": 0,
                "expected_goals": 0.5,
                "expected_assists": 0.0,
                "expected_goal_involvements": 0.5,
                "value": 70,
                "selected": 2_000_000,
                "xp": 4.5,
                "position": "FWD",
                "team": "Beta",
                "team_id": 20,
                "name": "BetaFwd",
            },
            "2": {
                "minutes": 90,
                "starts": 1,
                "points": 5,
                "goals": 0,
                "assists": 1,
                "expected_goals": 0.2,
                "expected_assists": 0.3,
                "expected_goal_involvements": 0.5,
                "value": 70,
                "selected": 2_000_000,
                "xp": 4.5,
                "position": "FWD",
                "team": "Beta",
                "team_id": 20,
                "name": "BetaFwd",
            },
        },
        "3": {
            "1": {
                "minutes": 90,
                "starts": 1,
                "points": 4,
                "goals": 0,
                "assists": 0,
                "expected_goals": 0.0,
                "expected_assists": 0.0,
                "expected_goal_involvements": 0.0,
                "value": 45,
                "selected": 500_000,
                "xp": 2.0,
                "position": "DEF",
                "team": "Gamma",
                "team_id": 30,
                "name": "BlankDef",
            },
            # GW2 blank for Gamma — no row → player omitted from pool; fixture
            # list also has no Gamma match so a forced project would be 0.0.
        },
    }
    prior_totals = {
        1: {
            "season_name": "2024/25",
            "minutes": 3000,
            "goals_scored": 5,
            "assists": 5,
            "clean_sheets": 0,
            "bonus": 10,
            "total_points": 120,
            "saves": 0,
        },
        2: {
            "season_name": "2024/25",
            "minutes": 2800,
            "goals_scored": 15,
            "assists": 5,
            "clean_sheets": 0,
            "bonus": 20,
            "total_points": 150,
            "saves": 0,
        },
        3: {
            "season_name": "2024/25",
            "minutes": 3200,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 10,
            "bonus": 5,
            "total_points": 100,
            "saves": 0,
        },
    }
    fixtures = [
        {
            "event": 1,
            "team_h": 10,
            "team_a": 20,
            "team_h_difficulty": 3,
            "team_a_difficulty": 3,
        },
        {
            "event": 1,
            "team_h": 30,
            "team_a": 40,
            "team_h_difficulty": 2,
            "team_a_difficulty": 4,
        },
        {
            "event": 2,
            "team_h": 20,
            "team_a": 10,
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
        },
        # Gamma (30) has no GW2 fixture → blank
    ]

    a = build_replay_inputs(per_gw, prior_totals, fixtures, 2, names={1: "A", 2: "B", 3: "C"})
    b = build_replay_inputs(per_gw, prior_totals, fixtures, 2, names={1: "A", 2: "B", 3: "C"})
    players, history, bundle = a
    assert a[0].keys() == b[0].keys()
    assert 3 not in players  # blank GW omitted
    assert set(players) == {1, 2}
    assert all("ep_next" in p for p in players.values())
    assert all(p["status"] == "a" for p in players.values())

    v1 = project(players, history, fixtures, "v1", before_gw=2)
    v2 = project(players, history, fixtures, "v2", before_gw=2)
    assert set(v1) == set(players)
    assert set(v2) == set(players)
    assert json.dumps(v1, sort_keys=True) == json.dumps(
        project(players, history, fixtures, "v1", before_gw=2), sort_keys=True
    )

    # Force a blank-GW player through project: no fixture → 0.0 under v2.
    blank_players = {
        3: {
            "id": 3,
            "web_name": "BlankDef",
            "element_type": 2,
            "team": 30,
            "team_name": "Gamma",
            "now_cost": 45,
            "status": "a",
            "chance_of_playing": None,
            "ep_next": 2.0,
            "selected_by_percent": 5.0,
            "can_select": True,
            "news": "",
        }
    }
    blank_hist = {3: _history_row(prior_totals[3])}
    blank_v2 = project(blank_players, blank_hist, fixtures, "v2", before_gw=2)
    assert blank_v2[3]["ep"] == 0.0, blank_v2[3]

    proj = {pid: row["ep"] for pid, row in v2.items()}
    metrics = gw_metrics(proj, bundle["actual"], minutes=bundle["minutes"], played_only=True)
    assert metrics["mae"] is not None
    assert replay_caveats()
    print("replay self-test: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay adapter utilities")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        _selftest()
    else:
        parser.print_help()

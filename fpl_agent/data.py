"""Load FPL snapshots and live bootstrap; build a player lookup."""

import json
from pathlib import Path

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ENTRY_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/event/{event}/picks/"
ENTRY_ID = 4796993

SNAPSHOTS = Path("snapshots")
POSITION_LABELS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def _get_json(url):
    import requests

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def snapshot_dir(gw):
    return SNAPSHOTS / f"gw{gw}"


def latest_snapshot_gw(*, require_my_team=False):
    """Highest snapshots/gw{N}/ with bootstrap-static.json (and optional my_team.json)."""
    if not SNAPSHOTS.is_dir():
        raise FileNotFoundError(f"No {SNAPSHOTS} directory")
    gws = []
    for path in SNAPSHOTS.iterdir():
        if not path.is_dir() or not path.name.startswith("gw"):
            continue
        try:
            n = int(path.name[2:])
        except ValueError:
            continue
        if not (path / "bootstrap-static.json").exists():
            continue
        if require_my_team and not (path / "my_team.json").exists():
            continue
        gws.append(n)
    if not gws:
        need = "bootstrap-static.json"
        if require_my_team:
            need += " and my_team.json"
        raise FileNotFoundError(f"No snapshot with {need} found")
    return max(gws)


def load_snapshot(gw):
    """Load frozen bootstrap + fixtures from snapshots/gw{gw}/."""
    folder = snapshot_dir(gw)
    bootstrap = _read_json(folder / "bootstrap-static.json")
    fixtures = _read_json(folder / "fixtures.json")
    prediction_path = folder / "prediction.json"
    prediction = _read_json(prediction_path) if prediction_path.exists() else None
    return {
        "gw": gw,
        "dir": folder,
        "bootstrap": bootstrap,
        "fixtures": fixtures,
        "teams": bootstrap["teams"],
        "prediction": prediction,
    }


def fetch_live_bootstrap():
    """GET the live bootstrap-static catalogue."""
    return _get_json(BOOTSTRAP_URL)


def fetch_fixtures():
    """GET the live fixtures list."""
    return _get_json(FIXTURES_URL)


def fetch_entry_picks(entry_id=ENTRY_ID, event=1):
    """GET an entry's picks for one gameweek."""
    return _get_json(ENTRY_PICKS_URL.format(entry_id=entry_id, event=event))


def build_players(bootstrap):
    """id → {web_name, team, element_type, now_cost, ep_next, ...}."""
    teams_by_id = {t["id"]: t["name"] for t in bootstrap["teams"]}
    players = {}
    for p in bootstrap["elements"]:
        ep = p.get("ep_next")
        sel = p.get("selected_by_percent")
        players[p["id"]] = {
            "id": p["id"],
            "web_name": p["web_name"],
            "team": p["team"],
            "team_name": teams_by_id[p["team"]],
            "element_type": p["element_type"],
            "now_cost": p["now_cost"],
            "ep_next": float(ep) if ep not in (None, "") else 0.0,
            "selected_by_percent": float(sel) if sel not in (None, "") else 0.0,
            "status": p.get("status") or "a",
            "chance_of_playing": p.get("chance_of_playing_next_round"),
            "news": p.get("news") or "",
            "can_select": p.get("can_select", True),
        }
    return players


def load_history(gw=1):
    """Load history_past.json keyed by int player id."""
    raw = _read_json(snapshot_dir(gw) / "history_past.json")
    return {int(pid): seasons for pid, seasons in raw.items()}


def load_my_picks(gw=1):
    """Load my 15 picks in position order from snapshots/gw{N}/my_team.json."""
    folder = snapshot_dir(gw)
    path = folder / "my_team.json"
    if path.exists():
        data = _read_json(path)
        if data.get("picks"):
            return sorted(data["picks"], key=lambda p: p["position"])
    raise FileNotFoundError(f"No {path} with picks[] found")


def load_my_bank(gw=1):
    """Bank balance (£0.1m units) from snapshots/gw{N}/my_team.json transfers.bank."""
    folder = snapshot_dir(gw)
    path = folder / "my_team.json"
    if path.exists():
        data = _read_json(path)
        transfers = data.get("transfers") or {}
        return int(transfers.get("bank") or 0)
    return 0

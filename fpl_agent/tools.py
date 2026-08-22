"""Agent tools wrapping snapshot data, projections, and the optimizer."""

import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import requests

from fpl_agent.data import (
    ENTRY_ID,
    ENTRY_PICKS_URL,
    POSITION_LABELS,
    build_players,
    latest_snapshot_gw,
    load_history,
    load_my_bank,
    load_my_picks,
    load_snapshot,
)
from fpl_agent.optimize import ATTACKERS, best_squad, best_xi, suggest_transfer
from fpl_agent.projections import expected_minutes, per_90_rates, project

MODELS = ("v0", "v1", "v2")
ENTRY_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/"
LEAGUE_STANDINGS_URL = (
    "https://fantasy.premierleague.com/api/leagues-classic/{league_id}/standings/"
)
PICKS_SLEEP_S = 0.3
FLAGGED = {"i", "d", "s", "u", "n"}
IST = timezone(timedelta(hours=5, minutes=30))

TOOLS = [
    {
        "name": "get_context",
        "description": (
            "Return current time (UTC/IST), gameweek state, next deadline, and "
            "hours remaining. Call first when advice is time-sensitive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_my_squad",
        "description": (
            "Return my 15-man FPL squad from the latest snapshot: names, positions, "
            "prices, captain/vice, injury and confidence flags."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "project_points",
        "description": (
            "Project expected points for my squad and the top 20 players overall "
            "under a chosen model (v0=ep_next, v1=per-90, v2=v1+crowd/availability/fixtures)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": list(MODELS),
                    "description": "Projection model. Default v2.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "optimize_squad",
        "description": (
            "Pick a legal £100m 15, best XI, and captain that maximise the chosen "
            "model's expected points. Returns the squad, XI, captain, projected "
            "score, and delta vs my current squad."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": list(MODELS),
                    "description": "Projection model. Default v2.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "audit_squad",
        "description": (
            "Check my squad for captain/injury/bench-order issues. Includes "
            "gameweek context (deadline, gw_state) and a list of warnings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_fixtures",
        "description": (
            "Return fixtures for a gameweek with home/away team names and FDR difficulty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gw": {
                    "type": "integer",
                    "description": "Gameweek number. Default: snapshot gw.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_rivals",
        "description": (
            "Mini-league standings, rival ownership, differentials, template threats, "
            "and captain spread for the last locked gameweek. Auto-picks your smallest "
            "private classic league unless league_id is given."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "league_id": {
                    "type": "integer",
                    "description": "Classic league id. Omit to auto-select smallest private league.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top standings entries to sample. Default 10.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "suggest_transfer",
        "description": (
            "Evaluate hold, every legal 1-transfer, and 2-transfer combos (-4 hit) "
            "against my squad. Returns top 3 options by net XI+C gain and recommends "
            "rolling the transfer when no move beats hold by at least 1.0 points."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": list(MODELS),
                    "description": "Projection model. Default v2.",
                },
                "min_gain": {
                    "type": "number",
                    "description": "Min net gain vs hold to recommend a transfer. Default 1.0.",
                },
            },
            "additionalProperties": False,
        },
    },
]


@lru_cache(maxsize=1)
def _snap():
    gw = latest_snapshot_gw()
    return load_snapshot(gw)


@lru_cache(maxsize=1)
def _squad_snap():
    gw = latest_snapshot_gw(require_my_team=True)
    return load_snapshot(gw)


@lru_cache(maxsize=1)
def _ctx():
    snap = _squad_snap()
    gw = snap["gw"]
    players = build_players(snap["bootstrap"])
    history = load_history(gw)
    picks = load_my_picks(gw)
    return snap, players, history, picks


def _meta_snap(tool_name):
    if tool_name in ("get_context", "get_fixtures", "get_rivals"):
        return _snap()
    return _squad_snap()


def _snapshot_time(snap):
    path = snap["dir"] / "bootstrap-static.json"
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _attach_meta(result, snap, tool_name=None):
    """Add provenance block; result must be a mutable dict."""
    model = result.get("model")
    if model is None and tool_name == "audit_squad":
        model = "v2"
    result["_meta"] = {
        "gw": snap["gw"],
        "snapshot_time": _snapshot_time(snap),
        "model": model,
    }
    return result


def _model(args):
    model = (args or {}).get("model") or "v2"
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected {MODELS}")
    return model


def _flags(pick, player, history):
    flags = []
    if pick.get("is_captain"):
        flags.append("C")
    if pick.get("is_vice_captain"):
        flags.append("V")
    rows = history.get(player["id"]) or []
    if per_90_rates(rows) is None:
        flags.append("low_confidence")
    exp = expected_minutes(player, rows)
    if exp is not None and exp < 45:
        flags.append("rotation_risk")
    if player.get("status") and player["status"] != "a":
        flags.append(f"status:{player['status']}")
    if player.get("news"):
        flags.append("news")
    return flags


def _player_row(pick, player, history, extra=None):
    row = {
        "id": player["id"],
        "name": player["web_name"],
        "position": POSITION_LABELS[player["element_type"]],
        "team": player["team_name"],
        "price": round(player["now_cost"] / 10.0, 1),
        "squad_position": pick["position"],
        "is_captain": bool(pick.get("is_captain")),
        "is_vice_captain": bool(pick.get("is_vice_captain")),
        "status": player.get("status") or "a",
        "chance_of_playing": player.get("chance_of_playing"),
        "news": player.get("news") or "",
        "flags": _flags(pick, player, history),
    }
    if extra:
        row.update(extra)
    return row


def _xi_score(picks, proj):
    total = 0.0
    for pick in picks:
        if pick["position"] > 11:
            continue
        ep = proj[pick["element"]]["ep"]
        total += ep
        if pick.get("is_captain"):
            total += ep
    return total


def _pack_ids(ids, players, proj, captain=None):
    rows = []
    for pid in ids:
        p = players[pid]
        rows.append(
            {
                "id": pid,
                "name": p["web_name"],
                "position": POSITION_LABELS[p["element_type"]],
                "team": p["team_name"],
                "price": round(p["now_cost"] / 10.0, 1),
                "ep": round(proj[pid]["ep"], 2),
                "is_captain": pid == captain,
                "low_confidence": proj[pid]["low_confidence"],
            }
        )
    return rows


def _parse_deadline(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _format_deadline(ev):
    iso = ev.get("deadline_time")
    if not iso:
        return {"event": ev.get("id"), "utc": None, "ist": None, "weekday": None}
    dt = _parse_deadline(iso)
    ist_dt = dt.astimezone(IST)
    return {
        "event": ev.get("id"),
        "utc": dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ist": f"{ist_dt.strftime('%a')} {ist_dt.day} {ist_dt.strftime('%b')}, {ist_dt.strftime('%I:%M %p')}",
        "weekday": ist_dt.strftime("%A"),
    }


def _format_now(now):
    ist_dt = now.astimezone(IST)
    return {
        "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "now_ist": (
            f"{ist_dt.strftime('%a')} {ist_dt.day} {ist_dt.strftime('%b')}, "
            f"{ist_dt.strftime('%I:%M %p')}"
        ),
    }


def _build_gw_context(bootstrap, now=None):
    """Single source of time/GW truth from bootstrap events + now (UTC)."""
    now = now or datetime.now(timezone.utc)
    events = bootstrap.get("events") or []

    dated = []
    for ev in events:
        iso = ev.get("deadline_time")
        if not iso:
            continue
        dated.append((_parse_deadline(iso), ev))
    dated.sort(key=lambda pair: pair[0])

    current_gw = None
    for deadline, ev in reversed(dated):
        if deadline <= now and not ev.get("finished"):
            current_gw = ev.get("id")
            break

    next_pair = next(((dt, ev) for dt, ev in dated if dt > now), None)
    if next_pair:
        next_deadline_dt, next_ev = next_pair
        next_gw = next_ev.get("id")
        next_deadline = _format_deadline(next_ev)
        hours_to_deadline = round((next_deadline_dt - now).total_seconds() / 3600.0, 1)
    else:
        next_gw = None
        next_deadline = None
        hours_to_deadline = None

    if current_gw is not None:
        gw_state = "in_progress"
    elif next_pair is not None:
        gw_state = "pre_deadline"
    else:
        gw_state = "finished"

    return {
        **_format_now(now),
        "current_gw": current_gw,
        "next_gw": next_gw,
        "next_deadline": next_deadline,
        "hours_to_deadline": hours_to_deadline,
        "gw_state": gw_state,
    }


def get_context():
    snap = _snap()
    return _build_gw_context(snap["bootstrap"])


def _locked_gw(bootstrap, context):
    """Last GW whose deadline has passed (picks may be public)."""
    if context.get("current_gw") is not None:
        return context["current_gw"]
    now = datetime.now(timezone.utc)
    locked = None
    for ev in bootstrap.get("events") or []:
        iso = ev.get("deadline_time")
        if iso and _parse_deadline(iso) <= now:
            locked = ev.get("id")
    return locked


def _deadline_passed_for_gw(bootstrap, gw):
    ev = next((e for e in bootstrap.get("events") or [] if e.get("id") == gw), None)
    if not ev or not ev.get("deadline_time"):
        return False
    return datetime.now(timezone.utc) >= _parse_deadline(ev["deadline_time"])


def _fetch_json(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_entry(entry_id=ENTRY_ID):
    return _fetch_json(ENTRY_URL.format(entry_id=entry_id))


def _fetch_league_standings(league_id):
    return _fetch_json(LEAGUE_STANDINGS_URL.format(league_id=league_id))


def _fetch_entry_picks(entry_id, gw):
    url = ENTRY_PICKS_URL.format(entry_id=entry_id, event=gw)
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _mini_classic_leagues(entry_data):
    """User-created classic leagues (exclude Overall / country system leagues)."""
    classics = entry_data.get("leagues", {}).get("classic") or []
    return [league for league in classics if league.get("league_type") == "c"]


def _league_catalog(mini_leagues):
    catalog = []
    for league in mini_leagues:
        standings = _fetch_league_standings(league["id"])
        size = len(standings.get("standings", {}).get("results") or [])
        catalog.append(
            {
                "id": league["id"],
                "name": league.get("name") or f"League {league['id']}",
                "my_rank": league.get("entry_rank"),
                "size": size,
            }
        )
    catalog.sort(key=lambda row: (row["size"], row["id"]))
    return catalog


def _pick_smallest_league(catalog):
    if not catalog:
        return None
    return catalog[0]


def get_rivals(league_id=None, top_n=10):
    snap = _snap()
    players = build_players(snap["bootstrap"])
    bootstrap = snap["bootstrap"]
    context = _build_gw_context(bootstrap)
    locked_gw = _locked_gw(bootstrap, context)
    top_n = max(1, int(top_n or 10))

    entry = _fetch_entry(ENTRY_ID)
    mini = _mini_classic_leagues(entry)
    catalog = _league_catalog(mini) if mini else []

    if league_id is None:
        chosen = _pick_smallest_league(catalog)
        if chosen is None:
            return {
                "error": "No private classic leagues found on this entry.",
                "classic_leagues": catalog,
            }
        league_id = chosen["id"]
        league_name = chosen["name"]
        my_rank = chosen["my_rank"]
    else:
        league_name = next(
            (row["name"] for row in catalog if row["id"] == league_id),
            f"League {league_id}",
        )
        my_rank = next(
            (row["my_rank"] for row in catalog if row["id"] == league_id),
            None,
        )

    if locked_gw is None:
        target = context.get("next_gw") or 1
        return {
            "message": f"picks not yet public for gw {target}",
            "league_id": league_id,
            "league_name": league_name,
            "classic_leagues": catalog,
            **context,
        }

    if not _deadline_passed_for_gw(bootstrap, locked_gw):
        return {
            "message": f"picks not yet public for gw {locked_gw}",
            "league_id": league_id,
            "league_name": league_name,
            "locked_gw": locked_gw,
            "classic_leagues": catalog,
            **context,
        }

    standings_data = _fetch_league_standings(league_id)
    results = standings_data.get("standings", {}).get("results") or []
    league_size = len(results)
    partial = context.get("gw_state") == "in_progress"
    standings = [
        {
            "entry_id": row["entry"],
            "team_name": row.get("entry_name") or "",
            "player_name": row.get("player_name") or "",
            "total_points": row.get("total"),
            "rank": row.get("rank"),
            "partial": partial,
        }
        for row in results[:top_n]
    ]

    rival_entries = [row["entry_id"] for row in standings if row["entry_id"] != ENTRY_ID]

    ownership_counts = Counter()
    captain_counts = Counter()
    sampled = 0
    fetch_failures = 0
    picks_public = False

    for idx, entry_id in enumerate(rival_entries):
        if idx > 0:
            time.sleep(PICKS_SLEEP_S)
        picks_data = _fetch_entry_picks(entry_id, locked_gw)
        if picks_data is None:
            fetch_failures += 1
            if not _deadline_passed_for_gw(bootstrap, locked_gw):
                return {
                    "message": f"picks not yet public for gw {locked_gw}",
                    "league_id": league_id,
                    "league_name": league_name,
                    "locked_gw": locked_gw,
                    "classic_leagues": catalog,
                    **context,
                }
            continue
        picks_public = True
        sampled += 1
        squad_ids = {p["element"] for p in picks_data.get("picks") or []}
        ownership_counts.update(squad_ids)
        captain_id = next(
            (p["element"] for p in picks_data.get("picks") or [] if p.get("is_captain")),
            None,
        )
        if captain_id is not None:
            captain_counts[captain_id] += 1

    if not picks_public and sampled == 0:
        return {
            "message": f"picks not yet public for gw {locked_gw}",
            "league_id": league_id,
            "league_name": league_name,
            "locked_gw": locked_gw,
            "fetch_failures": fetch_failures,
            "classic_leagues": catalog,
            **context,
        }

    my_picks_data = _fetch_entry_picks(ENTRY_ID, locked_gw)
    if my_picks_data is None:
        my_ids = {p["element"] for p in load_my_picks(snap["gw"])}
    else:
        my_ids = {p["element"] for p in my_picks_data.get("picks") or []}

    sample_size = sampled
    ownership_in_league = []
    for pid, count in ownership_counts.most_common():
        p = players.get(pid)
        ownership_in_league.append(
            {
                "id": pid,
                "name": p["web_name"] if p else str(pid),
                "owned_by": count,
                "sample_size": sample_size,
            }
        )

    diff_threshold = sample_size * 0.3 if sample_size else 0
    threat_threshold = sample_size * 0.5 if sample_size else 0

    my_differentials = []
    for pid in sorted(my_ids):
        if pid not in players:
            continue
        owned_by = ownership_counts.get(pid, 0)
        if owned_by < diff_threshold:
            my_differentials.append(
                {
                    "id": pid,
                    "name": players[pid]["web_name"],
                    "owned_by": owned_by,
                    "sample_size": sample_size,
                }
            )

    threats = [
        row
        for row in ownership_in_league
        if row["id"] not in my_ids and row["owned_by"] > threat_threshold
    ]

    captain_spread = []
    for pid, count in captain_counts.most_common():
        p = players.get(pid)
        captain_spread.append(
            {
                "captain_id": pid,
                "captain_name": p["web_name"] if p else str(pid),
                "captained_by": count,
                "sample_size": sample_size,
            }
        )

    return {
        "league_id": league_id,
        "league_name": league_name,
        "my_rank": my_rank,
        "classic_leagues": catalog,
        "locked_gw": locked_gw,
        "league_size": league_size,
        "sample_size": sample_size,
        "top_n": top_n,
        "standings": standings,
        "fetch_failures": fetch_failures,
        "ownership_in_league": ownership_in_league,
        "my_differentials": sorted(my_differentials, key=lambda r: r["owned_by"]),
        "threats": sorted(threats, key=lambda r: -r["owned_by"]),
        "captain_spread": captain_spread,
        **context,
    }


def get_my_squad():
    snap, players, history, picks = _ctx()
    squad = [_player_row(pick, players[pick["element"]], history) for pick in picks]
    return {"gw": snap["gw"], "entry_id": ENTRY_ID, "squad": squad}


def project_points(model="v2"):
    snap, players, history, picks = _ctx()
    model = _model({"model": model})
    proj = project(players, history, snap["fixtures"], model)

    mine = []
    for pick in picks:
        pid = pick["element"]
        p = players[pid]
        mine.append(
            _player_row(
                pick,
                p,
                history,
                extra={
                    "ep": round(proj[pid]["ep"], 2),
                    "low_confidence": proj[pid]["low_confidence"],
                },
            )
        )

    ranked = sorted(players, key=lambda i: (-proj[i]["ep"], players[i]["web_name"]))
    top20 = []
    for pid in ranked[:20]:
        p = players[pid]
        top20.append(
            {
                "id": pid,
                "name": p["web_name"],
                "position": POSITION_LABELS[p["element_type"]],
                "team": p["team_name"],
                "price": round(p["now_cost"] / 10.0, 1),
                "ep": round(proj[pid]["ep"], 2),
            }
        )
    return {"gw": snap["gw"], "model": model, "my_squad": mine, "top20": top20}


def optimize_squad(model="v2"):
    snap, players, history, picks = _ctx()
    model = _model({"model": model})
    proj = project(players, history, snap["fixtures"], model)

    squad_ids = best_squad(proj, players)
    xi_ids, captain = best_xi(squad_ids, proj, players)
    squad_ids = sorted(
        squad_ids,
        key=lambda i: (players[i]["element_type"], -proj[i]["ep"], players[i]["web_name"]),
    )
    xi_ids = sorted(
        xi_ids,
        key=lambda i: (players[i]["element_type"], -proj[i]["ep"], players[i]["web_name"]),
    )
    projected = sum(proj[i]["ep"] for i in xi_ids) + proj[captain]["ep"]
    mine = _xi_score(picks, proj)
    cost = sum(players[i]["now_cost"] for i in squad_ids)
    return {
        "gw": snap["gw"],
        "model": model,
        "cost": round(cost / 10.0, 1),
        "squad": _pack_ids(squad_ids, players, proj, captain),
        "xi": _pack_ids(xi_ids, players, proj, captain),
        "captain": players[captain]["web_name"],
        "projected_score": round(projected, 1),
        "my_squad_score": round(mine, 1),
        "delta": round(projected - mine, 1),
    }


def audit_squad():
    snap, players, history, picks = _ctx()
    proj = project(players, history, snap["fixtures"], "v2")
    context = _build_gw_context(snap["bootstrap"])
    warnings = []

    captain = next((pk for pk in picks if pk.get("is_captain")), None)
    if captain:
        cap = players[captain["element"]]
        if cap["element_type"] not in ATTACKERS:
            warnings.append(
                f"Captain {cap['web_name']} is {POSITION_LABELS[cap['element_type']]}, not MID/FWD"
            )
        if captain["position"] > 11:
            warnings.append(f"Captain {cap['web_name']} is on the bench")

    for pick in picks:
        if pick["position"] > 11:
            continue
        p = players[pick["element"]]
        name = p["web_name"]
        if p.get("status") in FLAGGED:
            warnings.append(f"Starter {name} is flagged ({p['status']})")
        chance = p.get("chance_of_playing")
        if chance is not None and chance < 75:
            warnings.append(f"Starter {name} has {chance}% chance of playing")
        if p.get("news"):
            warnings.append(f"Starter {name} news: {p['news']}")

    # Pos 12 is always the backup GK — fixed by FPL. Only check outfield subs 13–15.
    outfield_bench = sorted(
        [pk for pk in picks if pk["position"] in (13, 14, 15)],
        key=lambda pk: pk["position"],
    )
    if len(outfield_bench) >= 2:
        subs = []
        for pk in outfield_bench:
            pid = pk["element"]
            subs.append(
                (
                    pk["position"],
                    players[pid]["web_name"],
                    proj[pid]["ep"],
                )
            )
        misordered = False
        for i in range(len(subs) - 1):
            pos_a, name_a, ep_a = subs[i]
            pos_b, name_b, ep_b = subs[i + 1]
            if ep_b > ep_a:
                misordered = True
                break
        if misordered:
            suggested = ", ".join(name for _, name, _ in sorted(subs, key=lambda t: (-t[2], t[0])))
            current = ", ".join(f"{name} (pos {pos}, {ep:.1f} ep)" for pos, name, ep in subs)
            warnings.append(
                f"Outfield bench order (v2): {current} — higher-projected subs should be "
                f"earlier (pos 13 first). Suggested order: {suggested}"
            )

    return {
        "gw": snap["gw"],
        **context,
        "warnings": warnings,
        "ok": not warnings,
    }


def get_fixtures(gw=None):
    snap = _snap()
    gw = int(gw if gw is not None else snap["gw"])
    teams = {t["id"]: t["name"] for t in snap["teams"]}
    out = []
    for fx in snap["fixtures"]:
        if fx.get("event") != gw:
            continue
        out.append(
            {
                "id": fx.get("id"),
                "gw": gw,
                "kickoff": fx.get("kickoff_time"),
                "home": teams.get(fx["team_h"], str(fx["team_h"])),
                "away": teams.get(fx["team_a"], str(fx["team_a"])),
                "home_fdr": fx.get("team_h_difficulty"),
                "away_fdr": fx.get("team_a_difficulty"),
            }
        )
    out.sort(key=lambda r: (r["kickoff"] or "", r["id"] or 0))
    return {"gw": gw, "fixtures": out}


def suggest_transfers(model="v2", min_gain=1.0):
    snap, players, history, picks = _ctx()
    model = _model({"model": model})
    proj = project(players, history, snap["fixtures"], model)
    bank = load_my_bank(snap["gw"])
    return {
        "gw": snap["gw"],
        "model": model,
        **suggest_transfer(
            picks, proj, players, bank=bank, min_gain=float(min_gain)
        ),
    }


HANDLERS = {
    "get_context": lambda args: get_context(),
    "get_my_squad": lambda args: get_my_squad(),
    "project_points": lambda args: project_points(model=(args or {}).get("model") or "v2"),
    "optimize_squad": lambda args: optimize_squad(model=(args or {}).get("model") or "v2"),
    "audit_squad": lambda args: audit_squad(),
    "get_fixtures": lambda args: get_fixtures(gw=(args or {}).get("gw")),
    "get_rivals": lambda args: get_rivals(
        league_id=(args or {}).get("league_id"),
        top_n=int((args or {}).get("top_n") or 10),
    ),
    "suggest_transfer": lambda args: suggest_transfers(
        model=(args or {}).get("model") or "v2",
        min_gain=(args or {}).get("min_gain") or 1.0,
    ),
}


def dispatch(name, args):
    """Run a tool and always return a JSON-serializable dict."""
    try:
        handler = HANDLERS.get(name)
        if handler is None:
            return {"error": f"Unknown tool {name!r}"}
        result = handler(args or {})
        if isinstance(result, dict) and "error" not in result:
            snap = _meta_snap(name)
            _attach_meta(result, snap, tool_name=name)
        return result
    except (Exception, SystemExit) as exc:
        return {"error": str(exc) or exc.__class__.__name__}

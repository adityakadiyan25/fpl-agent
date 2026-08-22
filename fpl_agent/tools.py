"""Agent tools wrapping snapshot data, projections, and the optimizer."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache

from fpl_agent.data import (
    POSITION_LABELS,
    build_players,
    load_history,
    load_my_bank,
    load_my_picks,
    load_snapshot,
)
from fpl_agent.optimize import ATTACKERS, best_squad, best_xi, suggest_transfer
from fpl_agent.projections import expected_minutes, per_90_rates, project

MODELS = ("v0", "v1", "v2")
GW = 1
FLAGGED = {"i", "d", "s", "u", "n"}
IST = timezone(timedelta(hours=5, minutes=30))

TOOLS = [
    {
        "name": "get_my_squad",
        "description": (
            "Return my 15-man FPL squad from the GW1 snapshot: names, positions, "
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
            "Check my squad for captain/injury/bench-order issues and return the "
            "next deadline time plus a list of warnings."
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
                    "description": "Gameweek number. Default 1.",
                }
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
def _ctx():
    snap = load_snapshot(GW)
    players = build_players(snap["bootstrap"])
    history = load_history(GW)
    picks = load_my_picks(GW)
    return snap, players, history, picks


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


def _deadline_status(bootstrap):
    """Compare now (UTC) to transfer deadline; find earliest future deadline."""
    now = datetime.now(timezone.utc)
    events = bootstrap.get("events") or []

    dated = []
    for ev in events:
        iso = ev.get("deadline_time")
        if not iso:
            continue
        dated.append((_parse_deadline(iso), ev))
    dated.sort(key=lambda pair: pair[0])

    next_ev = next((ev for dt, ev in dated if dt > now), None)
    next_deadline = _format_deadline(next_ev) if next_ev else None

    transfer_ev = next((e for e in events if e.get("is_next")), None)
    if transfer_ev is None:
        transfer_ev = next((e for e in events if e.get("is_current")), None)

    if transfer_ev and transfer_ev.get("deadline_time"):
        deadline_passed = now >= _parse_deadline(transfer_ev["deadline_time"])
    else:
        deadline_passed = next_deadline is None

    return deadline_passed, next_deadline


def get_my_squad():
    snap, players, history, picks = _ctx()
    squad = [_player_row(pick, players[pick["element"]], history) for pick in picks]
    return {"gw": snap["gw"], "entry_id": 4796993, "squad": squad}


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
    warnings = []
    deadline_passed, next_deadline = _deadline_status(snap["bootstrap"])

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
        "deadline_passed": deadline_passed,
        "next_deadline": next_deadline,
        "warnings": warnings,
        "ok": not warnings,
    }


def get_fixtures(gw=1):
    snap, players, history, picks = _ctx()
    del players, history, picks
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
    bank = load_my_bank(GW)
    return {
        "gw": snap["gw"],
        "model": model,
        **suggest_transfer(
            picks, proj, players, bank=bank, min_gain=float(min_gain)
        ),
    }


HANDLERS = {
    "get_my_squad": lambda args: get_my_squad(),
    "project_points": lambda args: project_points(model=(args or {}).get("model") or "v2"),
    "optimize_squad": lambda args: optimize_squad(model=(args or {}).get("model") or "v2"),
    "audit_squad": lambda args: audit_squad(),
    "get_fixtures": lambda args: get_fixtures(gw=int((args or {}).get("gw") or 1)),
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
        return handler(args or {})
    except (Exception, SystemExit) as exc:
        return {"error": str(exc) or exc.__class__.__name__}

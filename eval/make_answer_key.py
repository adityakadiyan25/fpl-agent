"""Recompute Bucket A checks from a frozen snapshot; rewrite golden_set.json."""

import _bootstrap  # noqa: F401

import argparse
from datetime import timedelta, timezone

from fpl_agent.data import build_players, load_history, load_snapshot
from fpl_agent.projections import project

from lib import GOLDEN_PATH, load_golden, save_golden

IST = timezone(timedelta(hours=5, minutes=30))
PRICE_TOL = 0.05
PROJ_TOL = 0.05
FLAGGED = {"i", "d", "s", "u", "n"}


def _pounds(tenths):
    return round(int(tenths) / 10.0, 1)


def _event(bootstrap, gw):
    return next((e for e in bootstrap.get("events") or [] if e.get("id") == gw), None)


def datetime_from_iso(iso):
    from datetime import datetime

    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _squad_rows(snap, players):
    team = _read_my_team(snap)
    rows = []
    for pick in sorted(team["picks"], key=lambda p: p["position"]):
        pid = pick["element"]
        p = players[pid]
        sell = pick.get("selling_price")
        price = _pounds(sell) if sell is not None else _pounds(p["now_cost"])
        rows.append(
            {
                "pick": pick,
                "player": p,
                "id": pid,
                "name": p["web_name"],
                "price": price,
                "position": pick["position"],
                "element_type": p["element_type"],
                "team_id": p["team"],
                "team_name": p["team_name"],
                "status": p.get("status") or "a",
            }
        )
    return rows, team


def _read_my_team(snap):
    import json

    path = snap["dir"] / "my_team.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _gw_fixture(fixtures, team_id, gw):
    for fx in fixtures:
        if fx.get("event") != gw:
            continue
        if fx.get("team_h") == team_id or fx.get("team_a") == team_id:
            return fx
    return None


def _or_names(*names):
    return list(dict.fromkeys(n for n in names if n))


def _and_names(names):
    return [[n] for n in names]


def _checks(must_include_any=None, number=None, tolerance=PRICE_TOL, unit=None):
    out = {}
    if must_include_any is not None:
        out["must_include_any"] = must_include_any
    if number is not None:
        block = {"value": number, "tolerance": tolerance}
        if unit:
            block["unit"] = unit
        out["must_include_number"] = block
    return out


def build_a_checks(gw):
    snap = load_snapshot(gw)
    players = build_players(snap["bootstrap"])
    rows, team = _squad_rows(snap, players)
    fixtures = snap["fixtures"]
    history = load_history(gw)
    proj = project(players, history, fixtures, "v2", before_gw=gw)
    transfers = team.get("transfers") or {}
    bank = _pounds(transfers.get("bank") or 0)
    value = _pounds(transfers.get("value") or 0)

    captain = next(r for r in rows if r["pick"].get("is_captain"))
    vice = next(r for r in rows if r["pick"].get("is_vice_captain"))
    bench = [r for r in rows if r["position"] > 11]

    by_price = {}
    for r in rows:
        by_price.setdefault(r["price"], []).append(r)
    min_price = min(by_price)
    max_price = max(by_price)
    cheapest = sorted(by_price[min_price], key=lambda r: r["name"])
    dearest = sorted(by_price[max_price], key=lambda r: r["name"])

    flagged = [r for r in rows if r["status"] in FLAGGED]
    arsenal = [r for r in rows if r["team_name"] == "Arsenal"]

    defs = [r for r in rows if r["element_type"] == 2]
    def_fdr = []
    for r in defs:
        fx = _gw_fixture(fixtures, r["team_id"], gw)
        if not fx:
            continue
        home = fx["team_h"] == r["team_id"]
        fdr = fx["team_h_difficulty"] if home else fx["team_a_difficulty"]
        opp_id = fx["team_a"] if home else fx["team_h"]
        opp = next((t["name"] for t in snap["teams"] if t["id"] == opp_id), str(opp_id))
        def_fdr.append((fdr, r, home, opp))
    easiest_fdr = min(d[0] for d in def_fdr)
    easiest = [d for d in def_fdr if d[0] == easiest_fdr]

    home_players = []
    for r in rows:
        fx = _gw_fixture(fixtures, r["team_id"], gw)
        if fx and fx["team_h"] == r["team_id"]:
            home_players.append(r)

    squad_ids = [r["id"] for r in rows]
    best_pid = max(squad_ids, key=lambda i: (proj[i]["ep"], -i))
    cap_ep = round(proj[captain["id"]]["ep"], 2)
    best_ep = round(proj[best_pid]["ep"], 2)

    pred_snap = load_snapshot(1)
    predicted = None
    if pred_snap.get("prediction"):
        predicted = pred_snap["prediction"].get("projected_score")

    ev = _event(snap["bootstrap"], gw)
    deadline_groups = []
    if ev and ev.get("deadline_time"):
        dt = datetime_from_iso(ev["deadline_time"]).astimezone(timezone.utc)
        ist = dt.astimezone(IST)
        deadline_groups = [
            [f"GW{gw}", f"gw {gw}", f"gameweek {gw}"],
            [
                f"{ist.strftime('%a')} {ist.day} {ist.strftime('%b')}",
                f"{ist.strftime('%A')} {ist.day} {ist.strftime('%B')}",
                f"{ist.day} {ist.strftime('%b')}",
                f"{ist.strftime('%b')} {ist.day}",
            ],
            [
                dt.strftime("%H:%M UTC"),
                dt.strftime("%H:%M"),
                ist.strftime("%H:%M IST"),
                ist.strftime("%I:%M %p").lstrip("0"),
            ],
        ]

    keys = {
        "A01": (
            f"{captain['name']} (vice: {vice['name']})",
            _checks(_or_names(captain["name"])),
        ),
        "A02": (
            f"£{bank:.1f}m",
            _checks(number=bank, tolerance=PRICE_TOL, unit="money"),
        ),
        "A03": (
            f"GW{gw} — {deadline_groups[1][0] if deadline_groups else '?'}, "
            f"{deadline_groups[2][2] if deadline_groups else '?'}",
            _checks(deadline_groups),
        ),
        "A04": (
            f"{dearest[0]['name']}, £{max_price:.1f}m",
            _checks(_or_names(*(r["name"] for r in dearest)), number=max_price, unit="money"),
        ),
        "A05": (
            "Tie: " + " and ".join(r["name"] for r in cheapest) + f", both £{min_price:.1f}m"
            if len(cheapest) > 1
            else f"{cheapest[0]['name']}, £{min_price:.1f}m",
            _checks(_or_names(*(r["name"] for r in cheapest)), number=min_price, unit="money"),
        ),
        "A06": (
            ", ".join(r["name"] for r in bench),
            _checks(_and_names([r["name"] for r in bench])),
        ),
        "A07": (
            "None — all 15 status available"
            if not flagged
            else ", ".join(f"{r['name']} ({r['status']})" for r in flagged),
            _checks(
                _or_names("none", "no injury", "no flag", "unflagged", "all 15", "available")
                if not flagged
                else _and_names([r["name"] for r in flagged])
            ),
        ),
        "A08": (
            f"{len(arsenal)} (" + ", ".join(r["name"] for r in arsenal) + ")",
            _checks(
                _and_names([r["name"] for r in arsenal]),
                number=float(len(arsenal)),
                tolerance=0.0,
                unit="points",
            ),
        ),
        "A09": (
            f"{easiest[0][1]['name']} — {easiest[0][3]} "
            f"({'H' if easiest[0][2] else 'A'}), FDR {easiest_fdr}",
            _checks(
                _or_names(*(d[1]["name"] for d in easiest)),
                number=float(easiest_fdr),
                tolerance=0.0,
                unit="points",
            ),
        ),
        "A10": (
            ", ".join(r["name"] for r in home_players),
            _checks(_and_names([r["name"] for r in home_players])),
        ),
        "A11": (
            f"{players[best_pid]['web_name']}, {best_ep:.2f} (v2)",
            _checks(
                _or_names(players[best_pid]["web_name"]),
                number=best_ep,
                tolerance=PROJ_TOL,
                unit="points",
            ),
        ),
        "A12": (
            f"{captain['name']}, {cap_ep:.2f} (v2)",
            _checks(_or_names(captain["name"]), number=cap_ep, tolerance=PROJ_TOL, unit="points"),
        ),
        "A13": (
            str(predicted) if predicted is not None else "—",
            _checks(number=float(predicted), tolerance=PROJ_TOL, unit="points")
            if predicted is not None
            else _checks(),
        ),
        "A14": (
            f"£{value:.1f}m with £{bank:.1f}m bank",
            _checks(number=value, tolerance=PRICE_TOL, unit="money"),
        ),
        "A15": (
            vice["name"],
            _checks(_or_names(vice["name"])),
        ),
    }
    return {cid: {"summary": summary, "checks": checks} for cid, (summary, checks) in keys.items()}


def main():
    parser = argparse.ArgumentParser(description="Regenerate Bucket A answer keys from a snapshot")
    parser.add_argument("--gw", type=int, required=True)
    args = parser.parse_args()

    cases = load_golden()
    computed = build_a_checks(args.gw)
    changed = []

    print(f"Answer keys from snapshots/gw{args.gw}/:")
    for case in cases:
        if case["bucket"] != "A":
            continue
        info = computed[case["id"]]
        print(f"  {case['id']}: {info['summary']}")
        old = case.get("checks") or {}
        new = info["checks"]
        if old != new:
            changed.append((case["id"], old, new))
        case["checks"] = new

    if not changed:
        print("no changes")
        return

    save_golden(cases)
    print(f"Updated {GOLDEN_PATH} ({len(changed)} check block(s) changed):")
    for cid, old, new in changed:
        print(f"  {cid}: {old} → {new}")


if __name__ == "__main__":
    main()

"""Build snapshots/gwN/my_team.json from the last locked public picks.

Captain/bench order are carried from the source GW's selections; bank/squad
reflect the last locked deadline. If transfers were already made for GW N this
week, overwrite this file with a manual authenticated export instead.
"""

import _bootstrap  # noqa: F401

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from fpl_agent.data import (
    ENTRY_ID,
    fetch_entry_picks,
    load_snapshot,
    snapshot_dir,
)


def sell_price(purchase, now):
    """FPL sell rule in 0.1m units: keep half of any rise, rounded down."""
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def _parse_deadline(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _deadline_passed(bootstrap, gw, now=None):
    now = now or datetime.now(timezone.utc)
    ev = next((e for e in bootstrap.get("events") or [] if e.get("id") == gw), None)
    if not ev or not ev.get("deadline_time"):
        return False
    return now >= _parse_deadline(ev["deadline_time"])


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _elements_by_id(bootstrap):
    return {p["id"]: p for p in bootstrap.get("elements") or []}


def _player_name(elements, pid):
    row = elements.get(pid) or {}
    return row.get("web_name") or str(pid)


def _fetch_picks_skipping_freehit(entry_id, source_event):
    """Return (payload, source_event, freehit_skipped). Floor at event 1."""
    skipped = []
    event = source_event
    while event >= 1:
        try:
            payload = fetch_entry_picks(entry_id=entry_id, event=event)
        except requests.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                print(
                    f"picks not available for GW{event} (HTTP 404).",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise
        if payload.get("active_chip") == "freehit":
            skipped.append(event)
            if event == 1:
                print(
                    "Freehit on GW1 with no earlier non-freehit picks to fall back to.",
                    file=sys.stderr,
                )
                sys.exit(1)
            event -= 1
            continue
        return payload, event, skipped
    print("No non-freehit picks found (floor at GW1).", file=sys.stderr)
    sys.exit(1)


def _purchase_prices(picks, source_event, target_gw, elements_n, chain):
    """Map element_id → purchase_price (0.1m units)."""
    purchase = {}
    chain_by_id = {}
    if chain:
        for row in chain.get("picks") or []:
            chain_by_id[row["element"]] = row

    source_bootstrap = None
    source_elements = {}
    source_path = snapshot_dir(source_event) / "bootstrap-static.json"
    if source_path.exists():
        source_bootstrap = _read_json(source_path)
        source_elements = _elements_by_id(source_bootstrap)

    if not chain:
        print(
            "WARNING: no chain my_team.json for source GW — "
            "selling prices are approximations (purchase = gwN now_cost).",
            file=sys.stderr,
        )
        for pick in picks:
            pid = pick["element"]
            purchase[pid] = int(elements_n[pid]["now_cost"]) if pid in elements_n else 0
        return purchase

    for pick in picks:
        pid = pick["element"]
        if pid in chain_by_id and chain_by_id[pid].get("purchase_price") is not None:
            purchase[pid] = int(chain_by_id[pid]["purchase_price"])
            continue
        name = _player_name(elements_n, pid)
        if pid in source_elements:
            price = int(source_elements[pid]["now_cost"])
            print(
                f"WARNING: {name} (id={pid}) not in chain file — "
                f"purchase={price} from snapshots/gw{source_event}/bootstrap-static.json",
                file=sys.stderr,
            )
            purchase[pid] = price
        elif pid in elements_n:
            price = int(elements_n[pid]["now_cost"])
            print(
                f"WARNING: {name} (id={pid}) not in chain file and no source bootstrap — "
                f"purchase={price} from snapshots/gw{target_gw}/bootstrap-static.json",
                file=sys.stderr,
            )
            purchase[pid] = price
        else:
            print(
                f"WARNING: {name} (id={pid}) missing from bootstraps; purchase=0",
                file=sys.stderr,
            )
            purchase[pid] = 0
    return purchase


def _build_picks(payload_picks, elements_n, purchase_by_id):
    out = []
    for pick in payload_picks:
        pid = pick["element"]
        el = elements_n.get(pid)
        purchase = purchase_by_id[pid]
        if el is None:
            name = pick.get("element") or pid
            print(
                f"WARNING: element {name} missing from gwN bootstrap "
                f"(left the league?) — selling_price = purchase_price",
                file=sys.stderr,
            )
            selling = purchase
            element_type = pick.get("element_type")
        else:
            selling = sell_price(purchase, int(el["now_cost"]))
            element_type = el.get("element_type")
            if element_type is None:
                element_type = pick.get("element_type")
        out.append(
            {
                "element": pid,
                "position": pick["position"],
                "multiplier": pick.get("multiplier"),
                "is_captain": bool(pick.get("is_captain")),
                "is_vice_captain": bool(pick.get("is_vice_captain")),
                "element_type": element_type,
                "selling_price": selling,
                "purchase_price": purchase,
            }
        )
    return out


def _transfer_diff(chain, new_picks):
    if not chain:
        return None
    old_ids = {p["element"] for p in chain.get("picks") or []}
    new_ids = {p["element"] for p in new_picks}
    return sorted(new_ids - old_ids), sorted(old_ids - new_ids)


def _selftest():
    vectors = [
        ((55, 57), 56),
        ((55, 58), 56),
        ((55, 56), 55),
        ((55, 54), 54),
        ((55, 55), 55),
        ((40, 46), 43),
    ]
    for (purchase, now), expected in vectors:
        got = sell_price(purchase, now)
        if got != expected:
            print(
                f"sell-rule FAIL: sell_price({purchase}, {now}) = {got}, expected {expected}",
                file=sys.stderr,
            )
            sys.exit(1)
    print("sell-rule self-test: OK")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build snapshots/gwN/my_team.json from the last locked public picks. "
            "Captain/bench order are carried from the source GW; bank/squad reflect "
            "the last locked deadline — if transfers were already made for GW N, "
            "overwrite with a manual authenticated export instead."
        )
    )
    parser.add_argument("--gw", type=int, help="target gameweek N (>= 2)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing my_team.json",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run sell_price vectors (no network)",
    )
    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.gw is None:
        parser.error("--gw is required (unless --selftest)")
    if args.gw < 2:
        print(
            "GW1 my_team.json is the manual authenticated export — "
            "refuse to overwrite it via public picks. Use --gw >= 2.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = snapshot_dir(args.gw) / "my_team.json"
    bootstrap_path = snapshot_dir(args.gw) / "bootstrap-static.json"
    if not bootstrap_path.exists():
        print(
            f"run scripts/snapshot.py --gw {args.gw} first",
            file=sys.stderr,
        )
        sys.exit(1)

    if out_path.exists() and not args.force:
        existing = _read_json(out_path)
        if "_provenance" not in existing:
            print(
                f"Refusing to overwrite {out_path}: manual authenticated export "
                f"(no _provenance). Pass --force to replace.",
                file=sys.stderr,
            )
        else:
            print(
                f"Refusing to overwrite {out_path} "
                f"(provenance={existing.get('_provenance')}). Pass --force to replace.",
                file=sys.stderr,
            )
        sys.exit(1)

    try:
        snap = load_snapshot(args.gw)
    except FileNotFoundError:
        print(
            f"run scripts/snapshot.py --gw {args.gw} first",
            file=sys.stderr,
        )
        sys.exit(1)

    bootstrap = snap["bootstrap"]
    elements_n = _elements_by_id(bootstrap)
    source_event = args.gw - 1

    if not _deadline_passed(bootstrap, source_event):
        print(
            f"Source GW{source_event} deadline has not passed yet — "
            f"picks are not public until after the deadline.",
            file=sys.stderr,
        )
        sys.exit(1)

    payload, source_event, freehit_skipped = _fetch_picks_skipping_freehit(
        ENTRY_ID, source_event
    )
    payload_picks = payload.get("picks") or []
    if len(payload_picks) != 15:
        print(
            f"Expected 15 picks from GW{source_event}, got {len(payload_picks)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    chain_path = snapshot_dir(source_event) / "my_team.json"
    chain = _read_json(chain_path) if chain_path.exists() else None
    purchase_by_id = _purchase_prices(
        payload_picks, source_event, args.gw, elements_n, chain
    )
    picks = _build_picks(payload_picks, elements_n, purchase_by_id)

    history = payload.get("entry_history") or {}
    team = {
        "picks": picks,
        "chips": [],
        "transfers": {
            "cost": 4,
            "status": "unknown",
            "limit": None,
            "made": history.get("event_transfers"),
            "bank": history.get("bank"),
            "value": history.get("value"),
        },
        "_provenance": {
            "method": "public_picks",
            "source_event": source_event,
            "freehit_skipped": freehit_skipped,
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prices_vs": f"snapshots/gw{args.gw}/bootstrap-static.json",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(team, indent=4) + "\n", encoding="utf-8")

    bank = team["transfers"]["bank"]
    print(f"Wrote {out_path}")
    print(f"source_event={source_event}  bank={bank}  entry={ENTRY_ID}")
    if freehit_skipped:
        print(f"freehit_skipped={freehit_skipped}")
    print(
        "Note: captain/bench order from source GW; bank/squad reflect the last "
        "locked deadline. If you already transferred for this GW, overwrite with "
        "a manual authenticated export."
    )
    diff = _transfer_diff(chain, picks)
    if diff is None:
        print("transfer diff: no chain file to compare")
    else:
        into, outof = diff
        if not into and not outof:
            print("no transfers detected")
        else:
            print(f"players in:  {into or '[]'}")
            print(f"players out: {outof or '[]'}")


if __name__ == "__main__":
    main()

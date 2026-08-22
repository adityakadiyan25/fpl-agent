"""Freeze an optimal shadow squad envelope for a gameweek (write once)."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fpl_data import POSITION_LABELS, build_players, load_history, load_snapshot
from fpl_optimize import best_squad, best_xi
from fpl_projections import project

MODELS = ("v0", "v1", "v2")
MAX_LOW_CONFIDENCE = 2


def _player_dict(pid, players, proj, *, in_xi=False, is_captain=False, is_vice=False):
    p = players[pid]
    return {
        "id": pid,
        "name": p["web_name"],
        "position": POSITION_LABELS[p["element_type"]],
        "team": p["team_name"],
        "price": round(p["now_cost"] / 10.0, 1),
        "ep": round(proj[pid]["ep"], 2),
        "low_confidence": proj[pid]["low_confidence"],
        "in_xi": in_xi,
        "is_captain": is_captain,
        "is_vice": is_vice,
    }


def _vice_captain(xi_ids, captain_id, proj):
    """Highest projected XI player who is not the captain."""
    candidates = [i for i in xi_ids if i != captain_id]
    if not candidates:
        return None
    return max(candidates, key=lambda i: (proj[i]["ep"], -i))


def _projected_score(xi_ids, captain_id, proj):
    return sum(proj[i]["ep"] for i in xi_ids) + proj[captain_id]["ep"]


def main():
    parser = argparse.ArgumentParser(description="Freeze optimal shadow squad for a GW")
    parser.add_argument("--gw", type=int, required=True, help="Gameweek number")
    parser.add_argument("--model", choices=MODELS, default="v2")
    args = parser.parse_args()

    out_dir = Path("snapshots") / f"gw{args.gw}"
    out_path = out_dir / "shadow_team.json"

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        created = existing.get("created_at", "unknown")
        print(f"Refusing to overwrite {out_path} (sealed at {created})")
        sys.exit(1)

    snap = load_snapshot(args.gw)
    players = build_players(snap["bootstrap"])
    history = load_history(args.gw)
    proj = project(players, history, snap["fixtures"], args.model)

    squad_ids = sorted(
        best_squad(proj, players, max_low_confidence=MAX_LOW_CONFIDENCE),
        key=lambda i: (players[i]["element_type"], -proj[i]["ep"], players[i]["web_name"]),
    )
    xi_ids, captain_id = best_xi(
        squad_ids, proj, players, captain_attackers_only=True
    )
    xi_ids = sorted(
        xi_ids,
        key=lambda i: (players[i]["element_type"], -proj[i]["ep"], players[i]["web_name"]),
    )
    xi_set = set(xi_ids)
    vice_id = _vice_captain(xi_ids, captain_id, proj)
    score = _projected_score(xi_ids, captain_id, proj)
    created_at = datetime.now(timezone.utc).isoformat()

    squad = [
        _player_dict(
            pid,
            players,
            proj,
            in_xi=pid in xi_set,
            is_captain=pid == captain_id,
            is_vice=pid == vice_id,
        )
        for pid in squad_ids
    ]
    xi = [row for row in squad if row["in_xi"]]

    payload = {
        "gameweek": args.gw,
        "model": args.model,
        "created_at": created_at,
        "projected_score": round(score, 1),
        "captain": _player_dict(captain_id, players, proj, in_xi=True, is_captain=True),
        "vice_captain": _player_dict(
            vice_id, players, proj, in_xi=True, is_vice=True
        )
        if vice_id is not None
        else None,
        "squad": squad,
        "xi": xi,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cap_name = players[captain_id]["web_name"]
    print(f"Shadow GW{args.gw} frozen: {cap_name} (C), projected {score:.1f}, at {created_at}")


if __name__ == "__main__":
    main()

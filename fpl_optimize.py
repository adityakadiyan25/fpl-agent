"""Legal-15 and best-XI integer programs over a projection vector."""

import heapq

import pulp

from fpl_data import POSITION_LABELS

# FPL starting-XI formation bounds (1 GK + 10 outfield)
MIN_IN_XI = {1: 1, 2: 3, 3: 2, 4: 1}
MAX_IN_XI = {1: 1, 2: 5, 3: 5, 4: 3}
SQUAD_COUNTS = {1: 2, 2: 5, 3: 5, 4: 3}
ATTACKERS = {3, 4}  # MID, FWD
_FORMATIONS = tuple(
    (d, m, f)
    for d in range(3, 6)
    for m in range(2, 6)
    for f in range(1, 4)
    if d + m + f == 10
)


def _picked(var):
    val = var.value()
    return val is not None and val >= 0.5


def _ep(projections, pid):
    return projections[pid]["ep"]


def _low(projections, pid):
    return bool(projections[pid].get("low_confidence"))


def best_squad(projections, players, budget=1000, max_low_confidence=2):
    """Legal 15 maximising sum(ep): 2/5/5/3, max 3 per club, cost ≤ budget.

    Skips players with can_select False. At most max_low_confidence
    low-confidence picks. Returns a list of 15 player ids.
    """
    ids = [
        pid
        for pid in players
        if pid in projections and players[pid].get("can_select", True)
    ]
    if not ids:
        raise SystemExit("Squad solve failed: no selectable players")

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")

    prob += pulp.lpSum(_ep(projections, i) * in_squad[i] for i in ids)
    prob += pulp.lpSum(players[i]["now_cost"] * in_squad[i] for i in ids) <= budget, "budget"
    prob += pulp.lpSum(in_squad[i] for i in ids) == 15, "squad_size"
    prob += (
        pulp.lpSum(in_squad[i] for i in ids if _low(projections, i)) <= max_low_confidence,
        "max_low_confidence",
    )

    for etype, n in SQUAD_COUNTS.items():
        prob += (
            pulp.lpSum(in_squad[i] for i in ids if players[i]["element_type"] == etype) == n,
            f"pos_{POSITION_LABELS[etype]}",
        )

    team_ids = {players[i]["team"] for i in ids}
    for team_id in team_ids:
        prob += (
            pulp.lpSum(in_squad[i] for i in ids if players[i]["team"] == team_id) <= 3,
            f"club_{team_id}",
        )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise SystemExit(f"Squad solve failed: {pulp.LpStatus[status]}")

    return [i for i in ids if _picked(in_squad[i])]


def best_xi(squad_ids, projections, players, captain_attackers_only=True):
    """Best legal XI from a 15, maximising XI ep + captain ep.

    If captain_attackers_only, the captain must be MID or FWD.
    Returns (xi_ids, captain_id).
    """
    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    in_xi = pulp.LpVariable.dicts("xi", squad_ids, cat="Binary")
    is_cap = pulp.LpVariable.dicts("cap", squad_ids, cat="Binary")

    # Objective: starting XI plus captain counted twice
    prob += pulp.lpSum(
        _ep(projections, i) * in_xi[i] + _ep(projections, i) * is_cap[i] for i in squad_ids
    )
    prob += pulp.lpSum(in_xi[i] for i in squad_ids) == 11, "xi_size"
    prob += pulp.lpSum(is_cap[i] for i in squad_ids) == 1, "one_captain"

    for i in squad_ids:
        prob += is_cap[i] <= in_xi[i], f"cap_in_xi_{i}"
        if captain_attackers_only and players[i]["element_type"] not in ATTACKERS:
            prob += is_cap[i] == 0, f"cap_attacker_{i}"

    for etype in (1, 2, 3, 4):
        pos_ids = [i for i in squad_ids if players[i]["element_type"] == etype]
        prob += pulp.lpSum(in_xi[i] for i in pos_ids) >= MIN_IN_XI[etype], f"xi_min_{etype}"
        prob += pulp.lpSum(in_xi[i] for i in pos_ids) <= MAX_IN_XI[etype], f"xi_max_{etype}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise SystemExit(f"XI solve failed: {pulp.LpStatus[status]}")

    xi_ids = [i for i in squad_ids if _picked(in_xi[i])]
    captain_id = next(i for i in squad_ids if _picked(is_cap[i]))
    return xi_ids, captain_id


def _legal_squad(squad_ids, players):
    """True if 15 satisfies 2/5/5/3 and max 3 per club."""
    if len(squad_ids) != 15 or len(set(squad_ids)) != 15:
        return False
    pos = {1: 0, 2: 0, 3: 0, 4: 0}
    clubs = {}
    for pid in squad_ids:
        p = players[pid]
        pos[p["element_type"]] += 1
        clubs[p["team"]] = clubs.get(p["team"], 0) + 1
    if pos != SQUAD_COUNTS:
        return False
    return all(c <= 3 for c in clubs.values())


def _fast_xi_score(squad_ids, projections, players, captain_attackers_only=True):
    """Best XI+C score by enumerating legal formations (no LP)."""
    by_type = {1: [], 2: [], 3: [], 4: []}
    for pid in squad_ids:
        by_type[players[pid]["element_type"]].append(pid)
    for ids in by_type.values():
        ids.sort(key=lambda i: -_ep(projections, i))

    best_score = -1.0
    best_cap = None
    for d, m, f in _FORMATIONS:
        if len(by_type[2]) < d or len(by_type[3]) < m or len(by_type[4]) < f:
            continue
        xi = by_type[1][:1] + by_type[2][:d] + by_type[3][:m] + by_type[4][:f]
        if captain_attackers_only:
            cap_pool = [i for i in xi if players[i]["element_type"] in ATTACKERS]
        else:
            cap_pool = xi
        if not cap_pool:
            continue
        captain = max(cap_pool, key=lambda i: _ep(projections, i))
        score = sum(_ep(projections, i) for i in xi) + _ep(projections, captain)
        if score > best_score:
            best_score = score
            best_cap = captain
    if best_cap is None:
        raise SystemExit("XI scoring failed for squad")
    return best_score, best_cap


def _squad_clubs(squad_ids, players):
    clubs = {}
    for pid in squad_ids:
        team = players[pid]["team"]
        clubs[team] = clubs.get(team, 0) + 1
    return clubs


def _swap_club_ok(clubs, out_id, in_id, players):
    out_team = players[out_id]["team"]
    in_team = players[in_id]["team"]
    if out_team == in_team:
        return True
    return clubs.get(in_team, 0) < 3


def _clubs_after_swaps(clubs, swaps, players):
    updated = dict(clubs)
    for out_id, in_id in swaps:
        out_team = players[out_id]["team"]
        in_team = players[in_id]["team"]
        if out_team == in_team:
            continue
        updated[out_team] -= 1
        if updated[out_team] == 0:
            del updated[out_team]
        updated[in_team] = updated.get(in_team, 0) + 1
        if updated[in_team] > 3:
            return None
    return updated


def _apply_swaps(squad_ids, swaps):
    """Apply [(out_id, in_id), ...] to a squad list."""
    squad = list(squad_ids)
    for out_id, in_id in swaps:
        squad[squad.index(out_id)] = in_id
    return squad


def _candidates_by_pos(players, projections, squad_set, top_k=None):
    by_pos = {1: [], 2: [], 3: [], 4: []}
    for pid, p in players.items():
        if pid in squad_set:
            continue
        if pid not in projections or not p.get("can_select", True):
            continue
        by_pos[p["element_type"]].append(pid)
    if top_k is not None:
        for pos in by_pos:
            by_pos[pos].sort(key=lambda i: -_ep(projections, i))
            by_pos[pos] = by_pos[pos][:top_k]
    return by_pos


def _swap_row(out_id, in_id, players, selling):
    return {
        "out_id": out_id,
        "out_name": players[out_id]["web_name"],
        "in_id": in_id,
        "in_name": players[in_id]["web_name"],
        "sell": selling[out_id],
        "buy": players[in_id]["now_cost"],
    }


def suggest_transfer(
    picks,
    projections,
    players,
    bank=0,
    captain_attackers_only=True,
    min_gain=1.0,
):
    """Rank hold / 1-transfer / 2-transfer options by net XI+C gain vs holding.

    Uses selling prices from picks, respects position and 3-per-club limits.
    Two transfers include a -4 point hit. The 2-transfer search uses the top 60
    projected replacements per position (1-transfers are exhaustive). Returns top 3
    options plus an explicit recommendation (\"roll the transfer\" when no move beats
    hold by min_gain).
    """
    squad_ids = [p["element"] for p in picks]
    selling = {
        p["element"]: int(p.get("selling_price") or players[p["element"]]["now_cost"])
        for p in picks
    }
    squad_set = set(squad_ids)
    by_pos = _candidates_by_pos(players, projections, squad_set)
    by_pos_two = _candidates_by_pos(players, projections, squad_set, top_k=60)
    clubs = _squad_clubs(squad_ids, players)
    score_fn = lambda squad: _fast_xi_score(
        squad, projections, players, captain_attackers_only
    )

    baseline, baseline_cap = score_fn(squad_ids)
    options = []

    options.append(
        {
            "action": "hold",
            "transfers": [],
            "transfer_hit": 0,
            "projected_score": round(baseline, 2),
            "net_gain": 0.0,
            "captain": players[baseline_cap]["web_name"],
            "why": (
                f"Hold squad; optimal XI+C projects {baseline:.1f} "
                f"(baseline — no gain from moving)."
            ),
        }
    )

    # (b) every legal 1-transfer swap
    best_transfer = 0.0
    for out_id in squad_ids:
        out_pos = players[out_id]["element_type"]
        for in_id in by_pos[out_pos]:
            if bank + selling[out_id] < players[in_id]["now_cost"]:
                continue
            if not _swap_club_ok(clubs, out_id, in_id, players):
                continue
            new_squad = _apply_swaps(squad_ids, [(out_id, in_id)])
            score, cap = score_fn(new_squad)
            gain = score - baseline
            best_transfer = max(best_transfer, gain)
            sell_m = selling[out_id] / 10.0
            buy_m = players[in_id]["now_cost"] / 10.0
            options.append(
                {
                    "action": "1_transfer",
                    "transfers": [_swap_row(out_id, in_id, players, selling)],
                    "transfer_hit": 0,
                    "projected_score": round(score, 2),
                    "net_gain": round(gain, 2),
                    "captain": players[cap]["web_name"],
                    "why": (
                        f"{players[out_id]['web_name']} out (£{sell_m:.1f}m sell) → "
                        f"{players[in_id]['web_name']} in (£{buy_m:.1f}m); "
                        f"XI+C {score:.1f} (+{gain:.1f} vs hold); captain {players[cap]['web_name']}"
                    ),
                }
            )

    # (c) best 2-transfer combos minus 4 — full search, bounded store for ranking
    two_heap = []
    two_cap = 200
    heap_seq = 0
    for i, out1 in enumerate(squad_ids):
        for out2 in squad_ids[i + 1 :]:
            p1 = players[out1]["element_type"]
            p2 = players[out2]["element_type"]
            proceeds = selling[out1] + selling[out2]
            min_cost2 = min(
                (players[in2]["now_cost"] for in2 in by_pos_two[p2]), default=9999
            )
            for in1 in by_pos_two[p1]:
                cost1 = players[in1]["now_cost"]
                if bank + proceeds < cost1 + min_cost2:
                    continue
                if not _swap_club_ok(clubs, out1, in1, players):
                    continue
                for in2 in by_pos_two[p2]:
                    if in1 == in2:
                        continue
                    total_cost = cost1 + players[in2]["now_cost"]
                    if bank + proceeds < total_cost:
                        continue
                    if _clubs_after_swaps(
                        clubs, [(out1, in1), (out2, in2)], players
                    ) is None:
                        continue
                    new_squad = _apply_swaps(squad_ids, [(out1, in1), (out2, in2)])
                    score, cap = score_fn(new_squad)
                    net = score - 4 - baseline
                    best_transfer = max(best_transfer, net)
                    row = {
                        "action": "2_transfers",
                        "transfers": [
                            _swap_row(out1, in1, players, selling),
                            _swap_row(out2, in2, players, selling),
                        ],
                        "transfer_hit": 4,
                        "projected_score": round(score, 2),
                        "net_gain": round(net, 2),
                        "captain": players[cap]["web_name"],
                        "why": (
                            f"{players[out1]['web_name']}→{players[in1]['web_name']}, "
                            f"{players[out2]['web_name']}→{players[in2]['web_name']}; "
                            f"-4 hit; XI+C {score:.1f}, net {net:+.1f} vs hold after hit; "
                            f"captain {players[cap]['web_name']}"
                        ),
                    }
                    if len(two_heap) < two_cap:
                        heapq.heappush(two_heap, (net, heap_seq, row))
                        heap_seq += 1
                    elif net > two_heap[0][0]:
                        heapq.heapreplace(two_heap, (net, heap_seq, row))
                        heap_seq += 1

    options.extend(row for _net, _seq, row in two_heap)

    options.sort(key=lambda o: (-o["net_gain"], o["action"], o["why"]))
    top3 = options[:3]
    if best_transfer < min_gain:
        recommendation = "roll the transfer"
    else:
        best = options[0]
        if best["action"] == "hold":
            recommendation = "roll the transfer"
        else:
            recommendation = best["why"]

    return {
        "bank": bank,
        "baseline_score": round(baseline, 2),
        "baseline_captain": players[baseline_cap]["web_name"],
        "min_gain_threshold": min_gain,
        "recommendation": recommendation,
        "best_transfer_gain": round(best_transfer, 2),
        "options": top3,
    }

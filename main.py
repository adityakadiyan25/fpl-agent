"""CLI: report / optimize / compare FPL expected-points models."""

import argparse

from fpl_agent.agent import run as agent_run
from fpl_agent.data import (
    POSITION_LABELS,
    build_players,
    load_history,
    load_my_bank,
    load_my_picks,
    load_snapshot,
)
from fpl_agent.optimize import best_squad, best_xi, suggest_transfer
from fpl_agent.projections import expected_minutes, per_90_rates, project

MODELS = ("v0", "v1", "v2")


def _load(gw):
    snap = load_snapshot(gw)
    players = build_players(snap["bootstrap"])
    history = load_history(gw)
    picks = load_my_picks(gw)
    by_model = {m: project(players, history, snap["fixtures"], m) for m in MODELS}
    return snap, players, history, picks, by_model


def _ep(projections, pid):
    return projections[pid]["ep"]


def _xi_score(picks, projections):
    """Positions 1–11, captain's projection counted twice."""
    total = 0.0
    for pick in picks:
        if pick["position"] > 11:
            continue
        ep = _ep(projections, pick["element"])
        total += ep
        if pick.get("is_captain"):
            total += ep
    return total


def _sort_ids(ids, players, projections):
    return sorted(
        ids,
        key=lambda i: (
            players[i]["element_type"],
            -_ep(projections, i),
            players[i]["web_name"],
        ),
    )


def _notes(pick, player, history):
    parts = []
    if pick.get("is_captain"):
        parts.append("(C)")
    if pick.get("is_vice_captain"):
        parts.append("(V)")
    rows = history.get(player["id"]) or []
    if per_90_rates(rows) is None:
        parts.append("low_confidence")
    exp = expected_minutes(player, rows)
    if exp is not None and exp < 45:
        parts.append("rotation risk")
    if player.get("news"):
        parts.append(player["news"])
        chance = player.get("chance_of_playing")
        if chance is not None:
            parts.append(f"({chance}% chance)")
    return " ".join(parts)


def cmd_report(args):
    """My 15 with v0/v1/v2 columns; projected score uses --model."""
    snap, players, history, picks, by_model = _load(args.gw)
    chosen = by_model[args.model]

    print(
        f"{'Pos':<4} {'Name':<16} {'Slot':<4} {'Team':<16} {'Price':>6} "
        f"{'v0':>6} {'v1':>6} {'v2':>6}  Notes"
    )
    print("-" * 92)
    for pick in picks:
        p = players[pick["element"]]
        print(
            f"{pick['position']:<4} {p['web_name']:<16} "
            f"{POSITION_LABELS[p['element_type']]:<4} {p['team_name']:<16} "
            f"{p['now_cost'] / 10:>6.1f} "
            f"{by_model['v0'][p['id']]['ep']:>6.2f} "
            f"{by_model['v1'][p['id']]['ep']:>6.2f} "
            f"{by_model['v2'][p['id']]['ep']:>6.2f}  "
            f"{_notes(pick, p, history)}"
        )
    print("-" * 92)
    print(f"Projected GW{snap['gw']} score ({args.model}): {_xi_score(picks, chosen):.1f}")


def cmd_optimize(args):
    """Optimal 15 + XI + captain under --model."""
    snap, players, _history, picks, by_model = _load(args.gw)
    chosen = by_model[args.model]

    squad_ids = _sort_ids(
        best_squad(chosen, players, max_low_confidence=args.max_low_confidence),
        players,
        chosen,
    )
    xi_ids, captain = best_xi(
        squad_ids, chosen, players, captain_attackers_only=args.captain_attackers_only
    )
    xi_ids = _sort_ids(xi_ids, players, chosen)
    projected = sum(_ep(chosen, i) for i in xi_ids) + _ep(chosen, captain)
    my_score = _xi_score(picks, chosen)
    cost = sum(players[i]["now_cost"] for i in squad_ids)
    n_low = sum(1 for i in squad_ids if chosen[i]["low_confidence"])
    formation = "-".join(
        str(sum(1 for i in xi_ids if players[i]["element_type"] == etype))
        for etype in (2, 3, 4)
    )

    def print_rows(ids, mark_captain=False):
        print(f"{'Name':<16} {'Pos':<4} {'Team':<16} {'Price':>6} {args.model:>6}")
        print("-" * 52)
        for i in ids:
            p = players[i]
            marker = " (C)" if mark_captain and i == captain else ""
            if chosen[i]["low_confidence"]:
                marker += " low_confidence"
            print(
                f"{p['web_name']:<16} {POSITION_LABELS[p['element_type']]:<4} "
                f"{p['team_name']:<16} {p['now_cost'] / 10:>6.1f} "
                f"{_ep(chosen, i):>6.2f}{marker}"
            )

    print(f"=== Optimal 15 ({args.model}, cost £{cost / 10:.1f}m / £100.0m) ===")
    print_rows(squad_ids)
    print(f"{n_low} of 15 picks are low-confidence")
    print()
    print(f"=== Best XI ({formation}) ===")
    print_rows(xi_ids, mark_captain=True)
    print()
    print(f"Suggested captain: {players[captain]['web_name']}  ({args.model} {_ep(chosen, captain):.1f})")
    print(f"Projected GW{snap['gw']} score: {projected:.1f}")
    print(f"My current squad:    {my_score:.1f}")
    print(f"Delta:               {projected - my_score:+.1f}")


def cmd_compare(args):
    """Every model's projection for my squad, one table."""
    snap, players, _history, picks, by_model = _load(args.gw)

    print(
        f"{'Pos':<4} {'Name':<16} {'Slot':<4} "
        f"{'v0':>6} {'v1':>6} {'v2':>6}"
    )
    print("-" * 48)
    for pick in picks:
        p = players[pick["element"]]
        markers = []
        if pick.get("is_captain"):
            markers.append("(C)")
        if pick.get("is_vice_captain"):
            markers.append("(V)")
        tag = f" {' '.join(markers)}" if markers else ""
        print(
            f"{pick['position']:<4} {p['web_name']:<16} "
            f"{POSITION_LABELS[p['element_type']]:<4} "
            f"{by_model['v0'][p['id']]['ep']:>6.2f} "
            f"{by_model['v1'][p['id']]['ep']:>6.2f} "
            f"{by_model['v2'][p['id']]['ep']:>6.2f}{tag}"
        )
    print("-" * 48)
    print(
        f"{'XI (C×2)':<24} "
        f"{_xi_score(picks, by_model['v0']):>6.1f} "
        f"{_xi_score(picks, by_model['v1']):>6.1f} "
        f"{_xi_score(picks, by_model['v2']):>6.1f}"
    )
    print(f"(GW{snap['gw']})")


def cmd_transfer(args):
    """Suggest hold / 1-transfer / 2-transfer moves vs rolling."""
    snap, players, history, picks, by_model = _load(args.gw)
    chosen = by_model[args.model]
    bank = load_my_bank(args.gw)

    result = suggest_transfer(
        picks,
        chosen,
        players,
        bank=bank,
        captain_attackers_only=args.captain_attackers_only,
        min_gain=args.min_gain,
    )

    print(f"=== Transfer suggestions (GW{snap['gw']}, {args.model}, bank £{bank / 10:.1f}m) ===")
    print(f"Baseline XI+C: {result['baseline_score']:.1f} (captain {result['baseline_captain']})")
    print(f"Recommendation: {result['recommendation']}")
    if result["best_transfer_gain"] >= args.min_gain:
        print(f"Best transfer net gain: +{result['best_transfer_gain']:.1f}")
    else:
        print(
            f"No transfer beats hold by ≥{args.min_gain:.1f} "
            f"(best: {result['best_transfer_gain']:+.1f})"
        )
    print()
    for i, opt in enumerate(result["options"], 1):
        hit = f", -{opt['transfer_hit']} hit" if opt["transfer_hit"] else ""
        print(f"{i}. [{opt['action']}] net {opt['net_gain']:+.1f}{hit}")
        print(f"   {opt['why']}")
        if opt["transfers"]:
            for t in opt["transfers"]:
                print(
                    f"   {t['out_name']} → {t['in_name']} "
                    f"(sell £{t['sell'] / 10:.1f}m, buy £{t['buy'] / 10:.1f}m)"
                )
        print()


def cmd_agent(args):
    """Ask the FPL assistant (Claude + tools)."""
    agent_run(" ".join(args.question))


def main():
    parser = argparse.ArgumentParser(description="FPL expected-points tools")
    parser.add_argument("--gw", type=int, default=1, help="snapshot gameweek (default 1)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    report = sub.add_parser("report", help="my 15 with all model columns")
    report.add_argument("--model", choices=MODELS, default="v2")
    report.set_defaults(func=cmd_report)

    opt = sub.add_parser("optimize", help="optimal 15 + XI + captain")
    opt.add_argument("--model", choices=MODELS, default="v2")
    opt.add_argument(
        "--max-low-confidence",
        type=int,
        default=2,
        help="max low-confidence players allowed in the 15 (default 2)",
    )
    opt.add_argument(
        "--captain-attackers-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="captain must be MID/FWD (default true; --no-captain-attackers-only to allow any)",
    )
    opt.set_defaults(func=cmd_optimize)

    cmp_ = sub.add_parser("compare", help="every model on my squad, one table")
    cmp_.set_defaults(func=cmd_compare)

    xfer = sub.add_parser("transfer", help="hold vs 1- or 2-transfer suggestions")
    xfer.add_argument("--model", choices=MODELS, default="v2")
    xfer.add_argument(
        "--captain-attackers-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    xfer.add_argument(
        "--min-gain",
        type=float,
        default=1.0,
        help="min net points vs hold to recommend a transfer (default 1.0)",
    )
    xfer.set_defaults(func=cmd_transfer)

    agent = sub.add_parser("agent", help="ask the FPL assistant")
    agent.add_argument("question", nargs="+", help="your question")
    agent.set_defaults(func=cmd_agent)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

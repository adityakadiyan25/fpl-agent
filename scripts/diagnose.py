"""Diagnose naive / v1 / v2 / v3a / v3b / crowd projections vs 2025-26 actuals.

Reuses backtest.py replay (cached CSVs, pre-GW features). Does not impute
missing history as zero — those player-GWs are skipped and counted.
"""

import _bootstrap  # noqa: F401

from collections import defaultdict

from backtest import (
    apply_v2,
    compute_v1,
    compute_v3a,
    compute_v3b,
    gw_pool,
    load_replay,
    rolling_totals,
)
from fpl_agent.data import POSITION_LABELS
from fpl_agent.metrics import (
    bias,
    fmt_metric,
    gw_metrics,
    mae,
    mean_metric,
    p_at_11,
    rankdata,
    rmse,
    spearman,
)

SCORE_MODELS = ("naive", "v1", "v2", "v3a", "v3b")
MODELS = SCORE_MODELS + ("crowd",)
INTERSECTION_MODELS = SCORE_MODELS
HEADLINE_GW = (20, 38)
PER_GW_AVG_KEYS = ("spearman", "p_at_11", "haul_recall", "captain_regret", "xi_regret")
H2H_METRICS = ("mae", "rmse", "spearman", "p_at_11", "haul_recall", "captain_regret", "xi_regret")
PRICE_BANDS = (
    ("<£5.0", lambda p: p < 5.0),
    ("£5.0–7.5", lambda p: 5.0 <= p < 7.5),
    ("£7.5–10.0", lambda p: 7.5 <= p < 10.0),
    ("£10.0+", lambda p: p >= 10.0),
)
MIN_GWS_FOR_PLAYER = 5


def naive_proj(gw, prior, form):
    """Mean PPG this season from GW5; prior-season PPG before that. None = skip."""
    if gw < 5:
        if not prior or prior.get("minutes", 0) <= 0:
            return None
        return prior.get("points_per_game")
    if form["n_played"] <= 0:
        return None
    return form["points"] / form["n_played"]


def collect_rows(replay):
    """One record per player-GW with a fixture. pred[model] is float or None."""
    by_key = replay["by_key"]
    max_gw = replay["max_gw"]
    code_of = replay["code_of"]
    web_name_of = replay["web_name_of"]
    prior_by_code = replay["prior_by_code"]
    last_gw = min(38, max_gw)

    rows = []
    skips = {m: 0 for m in MODELS}
    fallback_counts = {"v3a": 0, "v3b": 0}
    seen_gws = set()

    for gw in range(2, 39):
        if gw > max_gw:
            print(f"Warning: GW{gw} missing from dataset (max GW{max_gw}), stopping")
            break
        pool = gw_pool(by_key, gw)
        if not pool:
            print(f"Warning: GW{gw} missing from dataset, continuing")
            continue
        seen_gws.add(gw)

        for pid, etype, b in pool:
            prior = prior_by_code.get(code_of.get(pid))
            form = rolling_totals(by_key, pid, gw)
            v1 = compute_v1(etype, prior, form)
            fb = {"v3b": 0}
            preds = {
                "naive": naive_proj(gw, prior, form),
                "v1": v1,
                "v2": apply_v2(v1, form, b["was_home"]),
                "v3a": compute_v3a(
                    etype, prior, form, by_key, pid, gw, b["selected"], b["was_home"]
                ),
                "v3b": compute_v3b(
                    etype,
                    prior,
                    form,
                    by_key,
                    pid,
                    gw,
                    b["selected"],
                    b["was_home"],
                    fallback_counter=fb,
                ),
                "crowd": float(b["selected"]) if b["selected"] is not None else None,
            }
            fallback_counts["v3b"] += fb.get("v3b", 0)
            for model, val in preds.items():
                if val is None:
                    skips[model] += 1
            rows.append(
                {
                    "gw": gw,
                    "pid": pid,
                    "name": web_name_of.get(pid) or b["name"],
                    "etype": etype,
                    "team": b["team"],
                    "price": b["value"] / 10.0,
                    "minutes": b["minutes"],
                    "actual": b["points"],
                    "pred": preds,
                }
            )

    expected = set(range(2, last_gw + 1))
    for gw in sorted(expected - seen_gws):
        print(f"Warning: GW{gw} missing from dataset, continuing")
    return rows, skips, fallback_counts


def universe(rows, played_only):
    if not played_only:
        return rows
    return [r for r in rows if r["minutes"] > 0]


def scored(rows, model):
    return [r for r in rows if r["pred"][model] is not None]


def rank_errors(rows, model):
    """Per-GW: error = actual_rank - pred_rank (1 = best). Positive = crowd overrated."""
    by_gw = defaultdict(list)
    for r in rows:
        by_gw[r["gw"]].append(r)
    errors = []
    for gw_rows in by_gw.values():
        if len(gw_rows) < 2:
            continue
        pred = [r["pred"][model] for r in gw_rows]
        actual = [r["actual"] for r in gw_rows]
        pred_rank = rankdata([-p for p in pred])
        act_rank = rankdata([-a for a in actual])
        for pr, ar in zip(pred_rank, act_rank):
            errors.append(ar - pr)
    return errors


def intersection_rows(rows, models, gw_range=None, played_only=False):
    lo, hi = gw_range or (2, 38)
    out = []
    for r in rows:
        if not (lo <= r["gw"] <= hi):
            continue
        if played_only and r["minutes"] <= 0:
            continue
        if all(r["pred"].get(m) is not None for m in models):
            out.append(r)
    return out


def _maps_from_rows(gw_rows, model):
    projections = {
        r["pid"]: r["pred"][model] for r in gw_rows if r["pred"].get(model) is not None
    }
    actuals = {r["pid"]: r["actual"] for r in gw_rows}
    minutes = {r["pid"]: r["minutes"] for r in gw_rows}
    players = {
        r["pid"]: {
            "element_type": r["etype"],
            "team": r["team"],
            "now_cost": int(round(r["price"] * 10)),
            "can_select": True,
        }
        for r in gw_rows
        if r["pid"] in projections
    }
    return projections, actuals, minutes, players


def aggregate_model_metrics(rows, model, *, played_only=False):
    """Pooled point metrics + per-GW averages for rank/optimizer metrics."""
    subset = universe(rows, played_only) if played_only else rows
    subset = scored(subset, model)
    projections = {r["pid"]: r["pred"][model] for r in subset}
    actuals = {r["pid"]: r["actual"] for r in subset}
    minutes = {r["pid"]: r["minutes"] for r in subset}

    out = {
        "mae": mae(projections, actuals, minutes=minutes, played_only=played_only),
        "bias": bias(projections, actuals, minutes=minutes, played_only=played_only),
        "rmse": rmse(projections, actuals, minutes=minutes, played_only=played_only),
    }

    by_gw = defaultdict(list)
    for r in subset:
        by_gw[r["gw"]].append(r)

    per_gw = {key: [] for key in PER_GW_AVG_KEYS}
    for gw_rows in by_gw.values():
        proj, act, mins, players = _maps_from_rows(gw_rows, model)
        gm = gw_metrics(
            proj,
            act,
            minutes=mins,
            played_only=False,
            players=players,
        )
        for key in PER_GW_AVG_KEYS:
            per_gw[key].append(gm[key])

    for key in PER_GW_AVG_KEYS:
        out[key] = mean_metric(per_gw[key])
    return out


def print_intersection_table(title, rows, models, gw_range, played_only):
    subset = intersection_rows(rows, models, gw_range=gw_range, played_only=played_only)
    lo, hi = gw_range
    filt = "mins>0" if played_only else "all fixtures"
    print(f"=== {title} (intersection, GW{lo}–{hi}, {filt}) ===")
    print(f"n={len(subset)} player-GWs where {', '.join(models)} all score")
    hdr = (
        f"{'model':<8} {'MAE':>8} {'bias':>8} {'RMSE':>8} {'Spρ':>8} "
        f"{'P@11':>8} {'haulR':>8} {'capR':>8} {'xiR':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        model_rows = subset
        projections = {r["pid"]: r["pred"][model] for r in model_rows}
        actuals = {r["pid"]: r["actual"] for r in model_rows}
        minutes = {r["pid"]: r["minutes"] for r in model_rows}
        pooled = {
            "mae": mae(projections, actuals, minutes=minutes, played_only=played_only),
            "bias": bias(projections, actuals, minutes=minutes, played_only=played_only),
            "rmse": rmse(projections, actuals, minutes=minutes, played_only=played_only),
        }
        by_gw = defaultdict(list)
        for r in model_rows:
            by_gw[r["gw"]].append(r)
        per_gw = {key: [] for key in PER_GW_AVG_KEYS}
        for gw_rows in by_gw.values():
            proj, act, mins, players = _maps_from_rows(gw_rows, model)
            gm = gw_metrics(proj, act, minutes=mins, players=players)
            for key in PER_GW_AVG_KEYS:
                per_gw[key].append(gm[key])
        metrics = dict(pooled)
        for key in PER_GW_AVG_KEYS:
            metrics[key] = mean_metric(per_gw[key])
        print(
            f"{model:<8} {fmt_metric(metrics['mae'])} {fmt_metric(metrics['bias'])} "
            f"{fmt_metric(metrics['rmse'])} {fmt_metric(metrics['spearman'])} "
            f"{fmt_metric(metrics['p_at_11'])} {fmt_metric(metrics['haul_recall'])} "
            f"{fmt_metric(metrics['captain_regret'])} {fmt_metric(metrics['xi_regret'])}"
        )
    print()
    return subset


def _gw_metric_maps(gw_rows, models):
    out = {}
    for model in models:
        out[model] = _maps_from_rows(gw_rows, model)
    return out


def per_gw_head_to_head(rows, models, baseline="v2", gw_range=None, played_only=True):
    """Count per-GW metric wins vs baseline across challengers."""
    lo, hi = gw_range or (2, 38)
    challengers = [m for m in models if m != baseline]
    wins = {m: {metric: 0 for metric in H2H_METRICS} for m in challengers}
    eligible_gws = 0

    by_gw = defaultdict(list)
    for r in rows:
        if lo <= r["gw"] <= hi:
            by_gw[r["gw"]].append(r)

    for gw_rows in by_gw.values():
        if played_only:
            gw_rows = [r for r in gw_rows if r["minutes"] > 0]
        gw_rows = [
            r
            for r in gw_rows
            if all(r["pred"].get(m) is not None for m in models)
        ]
        if not gw_rows:
            continue
        maps = _gw_metric_maps(gw_rows, models + (baseline,))
        base_proj, base_act, base_mins, base_players = maps[baseline]
        base_m = gw_metrics(
            base_proj, base_act, minutes=base_mins, players=base_players
        )
        eligible_gws += 1
        for challenger in challengers:
            proj, act, mins, players = maps[challenger]
            cm = gw_metrics(proj, act, minutes=mins, players=players)
            for metric in H2H_METRICS:
                bval = base_m[metric]
                cval = cm[metric]
                if bval is None or cval is None:
                    continue
                lower_better = metric in ("mae", "rmse", "captain_regret", "xi_regret")
                if lower_better and cval < bval:
                    wins[challenger][metric] += 1
                elif not lower_better and cval > bval:
                    wins[challenger][metric] += 1

    return wins, eligible_gws


def print_head_to_head(rows):
    wins, eligible = per_gw_head_to_head(
        rows,
        INTERSECTION_MODELS,
        baseline="v2",
        gw_range=HEADLINE_GW,
        played_only=True,
    )
    lo, hi = HEADLINE_GW
    print(f"=== Per-GW head-to-head vs v2 (intersection, GW{lo}–{hi}, mins>0) ===")
    print(f"eligible GWs={eligible}")
    hdr = f"{'model':<8}" + "".join(f" {m:>8}" for m in H2H_METRICS)
    print(hdr)
    print("-" * len(hdr))
    for model in ("v3a", "v3b", "naive", "v1"):
        if model not in wins:
            continue
        row = f"{model:<8}" + "".join(
            f" {wins[model][metric]:>8}" for metric in H2H_METRICS
        )
        print(row)
    print()


def print_verdict(rows):
    """Mechanical PASS/FAIL: v3a/v3b vs v2 on GW20–38 mins>0 intersection."""
    subset = intersection_rows(
        rows, INTERSECTION_MODELS, gw_range=HEADLINE_GW, played_only=True
    )
    baseline = aggregate_model_metrics(subset, "v2", played_only=False)
    print("=== Verdict (v3a/v3b vs v2, GW20–38 mins>0 intersection) ===")
    for challenger in ("v3a", "v3b"):
        m = aggregate_model_metrics(subset, challenger, played_only=False)
        mae_ok = (
            m["mae"] is not None
            and baseline["mae"] is not None
            and m["mae"] < baseline["mae"]
        )
        p11_ok = (
            m["p_at_11"] is not None
            and baseline["p_at_11"] is not None
            and m["p_at_11"] > baseline["p_at_11"]
        )
        passed = mae_ok and p11_ok
        mae_delta = (
            m["mae"] - baseline["mae"]
            if m["mae"] is not None and baseline["mae"] is not None
            else None
        )
        p11_delta = (
            m["p_at_11"] - baseline["p_at_11"]
            if m["p_at_11"] is not None and baseline["p_at_11"] is not None
            else None
        )
        verdict = "PASS" if passed else "FAIL"
        print(
            f"{challenger} vs v2: {verdict} "
            f"(ΔMAE={fmt_metric(mae_delta)}, ΔP@11={fmt_metric(p11_delta)})"
        )
    print()


def print_summary(rows, skips):
    print("=== Summary (models × metrics) ===")
    print("crowd MAE/bias are rank-space (1=best); other models are in FPL points.")
    print(f"Skips (player-GWs with a fixture but no history): {skips}")
    print()
    header = (
        f"{'model':<8} {'univ':<8} {'n':>7} {'MAE':>8} {'bias':>8} "
        f"{'RMSE':>8} {'Spρ':>8} {'P@11':>8}"
    )
    print(header)
    print("-" * len(header))
    for model in MODELS:
        for played_only, label in ((False, "all"), (True, "mins>0")):
            subset = scored(universe(rows, played_only), model)
            n = len(subset)
            if model == "crowd":
                mae_v, bias_v = (
                    (sum(abs(e) for e in rank_errors(subset, model)) / len(rank_errors(subset, model)))
                    if rank_errors(subset, model)
                    else None
                ), (
                    (sum(rank_errors(subset, model)) / len(rank_errors(subset, model)))
                    if rank_errors(subset, model)
                    else None
                )
                rho = mean_metric(
                    [
                        spearman(
                            {r["pid"]: r["pred"][model] for r in gw_rows},
                            {r["pid"]: r["actual"] for r in gw_rows},
                        )
                        for gw_rows in _group_by_gw(subset).values()
                        if len(gw_rows) >= 2
                    ]
                )
                p11 = mean_metric(
                    [
                        p_at_11(
                            {r["pid"]: r["pred"][model] for r in gw_rows},
                            {r["pid"]: r["actual"] for r in gw_rows},
                        )
                        for gw_rows in _group_by_gw(subset).values()
                    ]
                )
                rmse_v = None
            else:
                metrics = aggregate_model_metrics(subset, model, played_only=False)
                mae_v, bias_v, rmse_v, rho, p11 = (
                    metrics["mae"],
                    metrics["bias"],
                    metrics["rmse"],
                    metrics["spearman"],
                    metrics["p_at_11"],
                )
            print(
                f"{model:<8} {label:<8} {n:>7} {fmt_metric(mae_v)} {fmt_metric(bias_v)} "
                f"{fmt_metric(rmse_v)} {fmt_metric(rho)} {fmt_metric(p11)}"
            )
    print()


def _group_by_gw(rows):
    by_gw = defaultdict(list)
    for r in rows:
        by_gw[r["gw"]].append(r)
    return by_gw


def print_segments(rows):
    print("=== Segments (MAE / bias) ===")

    def block(title, groups):
        print(f"\n-- {title} --")
        hdr = f"{'seg':<14} {'model':<8} {'univ':<8} {'n':>7} {'MAE':>8} {'bias':>8}"
        print(hdr)
        print("-" * len(hdr))
        for gname, filt in groups:
            for model in MODELS:
                for played_only, label in ((False, "all"), (True, "mins>0")):
                    subset = [r for r in scored(universe(rows, played_only), model) if filt(r)]
                    if model == "crowd":
                        errs = rank_errors(subset, model)
                        mae_v = sum(abs(e) for e in errs) / len(errs) if errs else None
                        bias_v = sum(errs) / len(errs) if errs else None
                    else:
                        projections = {r["pid"]: r["pred"][model] for r in subset}
                        actuals = {r["pid"]: r["actual"] for r in subset}
                        minutes = {r["pid"]: r["minutes"] for r in subset}
                        mae_v = mae(
                            projections,
                            actuals,
                            minutes=minutes,
                            played_only=played_only,
                        )
                        bias_v = bias(
                            projections,
                            actuals,
                            minutes=minutes,
                            played_only=played_only,
                        )
                    print(
                        f"{gname:<14} {model:<8} {label:<8} {len(subset):>7} "
                        f"{fmt_metric(mae_v)} {fmt_metric(bias_v)}"
                    )

    block(
        "position",
        [(POSITION_LABELS[et], lambda r, e=et: r["etype"] == e) for et in (1, 2, 3, 4)],
    )
    block(
        "price",
        [(label, lambda r, fn=fn: fn(r["price"])) for label, fn in PRICE_BANDS],
    )
    block(
        "half",
        [
            ("GW2–19", lambda r: 2 <= r["gw"] <= 19),
            ("GW20–38", lambda r: 20 <= r["gw"] <= 38),
        ],
    )
    print()


def print_worst_players(rows):
    print("=== 20 largest mean |v2 error| (universe=all, ≥5 GWs) ===")
    by_pid = defaultdict(list)
    for r in scored(rows, "v2"):
        by_pid[r["pid"]].append(r)
    ranked = []
    for pid, recs in by_pid.items():
        if len(recs) < MIN_GWS_FOR_PLAYER:
            continue
        errs = [r["actual"] - r["pred"]["v2"] for r in recs]
        mae_v = sum(abs(e) for e in errs) / len(errs)
        ranked.append((mae_v, pid, recs))
    ranked.sort(reverse=True)

    hdr = f"{'Name':<18} {'Pos':<4} {'£m':>5} {'n':>4} {'MAE':>7} {'bias':>7}"
    print(hdr)
    print("-" * len(hdr))
    for mae_v, pid, recs in ranked[:20]:
        bias_v = sum(r["actual"] - r["pred"]["v2"] for r in recs) / len(recs)
        etype = recs[0]["etype"]
        price = sum(r["price"] for r in recs) / len(recs)
        name = recs[0]["name"]
        print(
            f"{name:<18} {POSITION_LABELS[etype]:<4} {price:>5.1f} {len(recs):>4} "
            f"{mae_v:>7.3f} {bias_v:>+7.3f}"
        )


def main():
    replay = load_replay()
    print(f"Loaded {replay['n_rows']} GW rows through GW{replay['max_gw']}")
    rows, skips, fallback_counts = collect_rows(replay)
    print(f"Player-GWs with a fixture: {len(rows)}")
    print()

    print_intersection_table(
        "Headline",
        rows,
        INTERSECTION_MODELS,
        gw_range=HEADLINE_GW,
        played_only=True,
    )
    print_intersection_table(
        "Secondary (early season)",
        rows,
        INTERSECTION_MODELS,
        gw_range=(2, 19),
        played_only=True,
    )
    print_intersection_table(
        "Secondary (full season)",
        rows,
        INTERSECTION_MODELS,
        gw_range=(2, 38),
        played_only=True,
    )

    print_head_to_head(rows)
    print_summary(rows, skips)
    print_segments(rows)
    print_worst_players(rows)

    print("=== xG fallback counts (realized attack rates used) ===")
    print(f"v3a: {fallback_counts['v3a']}")
    print(f"v3b: {fallback_counts['v3b']}")
    print()

    print_verdict(rows)


if __name__ == "__main__":
    main()

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

SCORE_MODELS = ("naive", "v1", "v2", "v3a", "v3b")
MODELS = SCORE_MODELS + ("crowd",)
INTERSECTION_MODELS = SCORE_MODELS
HEADLINE_GW = (20, 38)
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


def rankdata(values):
    """Average ranks, 1 = smallest value. Ties share the mean rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(pred, actual):
    """Higher pred and higher actual should correlate (we invert pred ranks)."""
    return pearson(rankdata([-p for p in pred]), rankdata([-a for a in actual]))


def precision_at_11(pred, actual, pids):
    k = min(11, len(pred))
    if k == 0:
        return None
    top_pred = {pid for pid, _ in sorted(zip(pids, pred), key=lambda t: (-t[1], t[0]))[:k]}
    top_act = {pid for pid, _ in sorted(zip(pids, actual), key=lambda t: (-t[1], t[0]))[:k]}
    return len(top_pred & top_act) / k


def mae_bias(errors):
    if not errors:
        return None, None
    n = len(errors)
    return sum(abs(e) for e in errors) / n, sum(errors) / n


def price_band(price):
    for label, fn in PRICE_BANDS:
        if fn(price):
            return label
    return PRICE_BANDS[-1][0]


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


def point_errors(rows, model):
    return [r["actual"] - r["pred"][model] for r in rows]


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


def gw_spearman(rows, model):
    by_gw = defaultdict(list)
    for r in rows:
        by_gw[r["gw"]].append(r)
    vals = []
    for gw_rows in by_gw.values():
        if len(gw_rows) < 2:
            continue
        rho = spearman(
            [r["pred"][model] for r in gw_rows],
            [r["actual"] for r in gw_rows],
        )
        if rho is not None:
            vals.append(rho)
    return (sum(vals) / len(vals) if vals else None), len(vals)


def gw_precision(rows, model):
    by_gw = defaultdict(list)
    for r in rows:
        by_gw[r["gw"]].append(r)
    vals = []
    for gw_rows in by_gw.values():
        p = precision_at_11(
            [r["pred"][model] for r in gw_rows],
            [r["actual"] for r in gw_rows],
            [r["pid"] for r in gw_rows],
        )
        if p is not None:
            vals.append(p)
    return (sum(vals) / len(vals) if vals else None), len(vals)


def fmt(val, nd=3):
    if val is None:
        return f"{'—':>8}"
    return f"{val:>8.{nd}f}"


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


def model_metrics(rows, model):
    mae, bias = mae_bias(point_errors(rows, model))
    rho, _ = gw_spearman(rows, model)
    p11, _ = gw_precision(rows, model)
    return {"mae": mae, "bias": bias, "spearman": rho, "p11": p11}


def print_intersection_table(title, rows, models, gw_range, played_only):
    subset = intersection_rows(rows, models, gw_range=gw_range, played_only=played_only)
    lo, hi = gw_range
    filt = "mins>0" if played_only else "all fixtures"
    print(f"=== {title} (intersection, GW{lo}–{hi}, {filt}) ===")
    print(f"n={len(subset)} player-GWs where {', '.join(models)} all score")
    hdr = f"{'model':<8} {'MAE':>8} {'bias':>8} {'Spearman':>8} {'P@11':>8}"
    print(hdr)
    print("-" * len(hdr))
    for model in models:
        m = model_metrics(subset, model)
        print(
            f"{model:<8} {fmt(m['mae'])} {fmt(m['bias'])} "
            f"{fmt(m['spearman'])} {fmt(m['p11'])}"
        )
    print()
    return subset


def print_verdict(rows):
    """Mechanical PASS/FAIL: v3a/v3b vs v2 on GW20–38 mins>0 intersection."""
    subset = intersection_rows(
        rows, INTERSECTION_MODELS, gw_range=HEADLINE_GW, played_only=True
    )
    baseline = model_metrics(subset, "v2")
    print("=== Verdict (v3a/v3b vs v2, GW20–38 mins>0 intersection) ===")
    for challenger in ("v3a", "v3b"):
        m = model_metrics(subset, challenger)
        mae_ok = m["mae"] is not None and baseline["mae"] is not None and m["mae"] < baseline["mae"]
        p11_ok = m["p11"] is not None and baseline["p11"] is not None and m["p11"] > baseline["p11"]
        passed = mae_ok and p11_ok
        mae_delta = (
            m["mae"] - baseline["mae"]
            if m["mae"] is not None and baseline["mae"] is not None
            else None
        )
        p11_delta = (
            m["p11"] - baseline["p11"]
            if m["p11"] is not None and baseline["p11"] is not None
            else None
        )
        verdict = "PASS" if passed else "FAIL"
        print(
            f"{challenger} vs v2: {verdict} "
            f"(ΔMAE={fmt(mae_delta)}, ΔP@11={fmt(p11_delta)})"
        )
    print()


def print_summary(rows, skips):
    print("=== Summary (models × metrics) ===")
    print("crowd MAE/bias are rank-space (1=best); other models are in FPL points.")
    print(f"Skips (player-GWs with a fixture but no history): {skips}")
    print()
    header = (
        f"{'model':<8} {'univ':<8} {'n':>7} {'MAE':>8} {'bias':>8} "
        f"{'Spearman':>8} {'P@11':>8}"
    )
    print(header)
    print("-" * len(header))
    for model in MODELS:
        for played_only, label in ((False, "all"), (True, "mins>0")):
            subset = scored(universe(rows, played_only), model)
            n = len(subset)
            if model == "crowd":
                mae, bias = mae_bias(rank_errors(subset, model))
            else:
                mae, bias = mae_bias(point_errors(subset, model))
            rho, _ = gw_spearman(subset, model)
            p11, _ = gw_precision(subset, model)
            print(
                f"{model:<8} {label:<8} {n:>7} {fmt(mae)} {fmt(bias)} "
                f"{fmt(rho)} {fmt(p11)}"
            )
    print()


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
                        mae, bias = mae_bias(rank_errors(subset, model))
                    else:
                        mae, bias = mae_bias(point_errors(subset, model))
                    print(
                        f"{gname:<14} {model:<8} {label:<8} {len(subset):>7} "
                        f"{fmt(mae)} {fmt(bias)}"
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
        mae = sum(abs(r["actual"] - r["pred"]["v2"]) for r in recs) / len(recs)
        ranked.append((mae, pid, recs))
    ranked.sort(reverse=True)

    hdr = f"{'Name':<18} {'Pos':<4} {'£m':>5} {'n':>4} {'MAE':>7} {'bias':>7}"
    print(hdr)
    print("-" * len(hdr))
    for mae, pid, recs in ranked[:20]:
        bias = sum(r["actual"] - r["pred"]["v2"] for r in recs) / len(recs)
        etype = recs[0]["etype"]
        price = sum(r["price"] for r in recs) / len(recs)
        name = recs[0]["name"]
        print(
            f"{name:<18} {POSITION_LABELS[etype]:<4} {price:>5.1f} {len(recs):>4} "
            f"{mae:>7.3f} {bias:>+7.3f}"
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

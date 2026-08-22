"""Pure projection-vs-actual metrics for a single gameweek pool."""

import math
from typing import Optional

from fpl_agent.optimize import ATTACKERS, best_squad, best_xi


def rankdata(values):
    """Average ranks, 1 = smallest value. Ties share the mean rank."""
    n = len(values)
    if n == 0:
        return []
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


def _pearson(xs, ys):
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


def paired_pids(projections, actuals, minutes=None, played_only=False):
    """Player ids with both a projection and an actual."""
    pids = [
        pid
        for pid in projections
        if pid in actuals and projections[pid] is not None and actuals[pid] is not None
    ]
    if played_only:
        if not minutes:
            return []
        pids = [pid for pid in pids if (minutes.get(pid) or 0) > 0]
    return pids


def point_errors(projections, actuals, minutes=None, played_only=False, pids=None):
    """actual − projected for each paired player. Empty list if none."""
    ids = pids if pids is not None else paired_pids(projections, actuals, minutes, played_only)
    return [actuals[pid] - projections[pid] for pid in ids]


def mae(projections, actuals, minutes=None, played_only=False, pids=None) -> Optional[float]:
    errs = point_errors(projections, actuals, minutes, played_only, pids)
    if not errs:
        return None
    return sum(abs(e) for e in errs) / len(errs)


def bias(projections, actuals, minutes=None, played_only=False, pids=None) -> Optional[float]:
    errs = point_errors(projections, actuals, minutes, played_only, pids)
    if not errs:
        return None
    return sum(errs) / len(errs)


def rmse(projections, actuals, minutes=None, played_only=False, pids=None) -> Optional[float]:
    errs = point_errors(projections, actuals, minutes, played_only, pids)
    if not errs:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def spearman(projections, actuals, minutes=None, played_only=False, pids=None) -> Optional[float]:
    ids = pids if pids is not None else paired_pids(projections, actuals, minutes, played_only)
    if len(ids) < 2:
        return None
    pred = [projections[pid] for pid in ids]
    act = [actuals[pid] for pid in ids]
    return _pearson(rankdata([-p for p in pred]), rankdata([-a for a in act]))


def p_at_11(projections, actuals, minutes=None, played_only=False, pids=None) -> Optional[float]:
    ids = pids if pids is not None else paired_pids(projections, actuals, minutes, played_only)
    k = min(11, len(ids))
    if k == 0:
        return None
    top_pred = {
        pid for pid, _ in sorted(zip(ids, [projections[i] for i in ids]), key=lambda t: (-t[1], t[0]))[:k]
    }
    top_act = {
        pid for pid, _ in sorted(zip(ids, [actuals[i] for i in ids]), key=lambda t: (-t[1], t[0]))[:k]
    }
    return len(top_pred & top_act) / k


def haul_recall(
    projections,
    actuals,
    minutes=None,
    played_only=False,
    pids=None,
    k=20,
    haul_threshold=10,
) -> Optional[float]:
    ids = pids if pids is not None else paired_pids(projections, actuals, minutes, played_only)
    haulers = [pid for pid in ids if actuals[pid] >= haul_threshold]
    if not haulers:
        return None
    top_k = sorted(ids, key=lambda pid: (-projections[pid], pid))[:k]
    hits = len(set(haulers) & set(top_k))
    return hits / len(haulers)


def _captain_pool(pids, players):
    if not players:
        return pids
    attackers = [pid for pid in pids if players[pid]["element_type"] in ATTACKERS]
    return attackers or pids


def captain_regret(
    projections,
    actuals,
    players=None,
    squad_ids=None,
    minutes=None,
    played_only=False,
) -> Optional[float]:
    """2 × (best hindsight captain pts − model captain pts) in the pool."""
    if squad_ids is not None:
        pool = [
            pid
            for pid in squad_ids
            if pid in projections and pid in actuals and projections[pid] is not None
        ]
    else:
        pool = paired_pids(projections, actuals, minutes, played_only)
    cap_pool = _captain_pool(pool, players)
    if not cap_pool:
        return None
    model_cap = max(cap_pool, key=lambda pid: (projections[pid], -pid))
    best_cap = max(cap_pool, key=lambda pid: (actuals[pid], -pid))
    return 2.0 * (actuals[best_cap] - actuals[model_cap])


def xi_score(xi_ids, captain_id, actuals) -> Optional[float]:
    """Sum XI actuals with captain counted twice."""
    uniq = list(dict.fromkeys(xi_ids))
    if len(uniq) != 11 or captain_id not in actuals:
        return None
    return sum(actuals[i] for i in uniq) + actuals[captain_id]


def xi_regret(projections, actuals, players, budget=1000) -> Optional[float]:
    """Hindsight optimal XI score minus model-optimal XI score (same budget rules)."""
    if not players:
        return None
    ids = [pid for pid in projections if pid in actuals and pid in players]
    if len(ids) < 15:
        return None
    model_proj = {pid: {"ep": projections[pid], "low_confidence": False} for pid in ids}
    hind_proj = {pid: {"ep": float(actuals[pid]), "low_confidence": False} for pid in ids}
    try:
        model_squad = best_squad(model_proj, players, budget=budget)
        model_xi, model_cap = best_xi(model_squad, model_proj, players)
        hind_squad = best_squad(hind_proj, players, budget=budget)
        hind_xi, hind_cap = best_xi(hind_squad, hind_proj, players)
    except SystemExit:
        return None
    model_pts = xi_score(model_xi, model_cap, actuals)
    hind_pts = xi_score(hind_xi, hind_cap, actuals)
    if model_pts is None or hind_pts is None:
        return None
    return float(hind_pts - model_pts)


def gw_metrics(
    projections,
    actuals,
    *,
    minutes=None,
    played_only=False,
    pids=None,
    players=None,
    squad_ids=None,
    haul_k=20,
    haul_threshold=10,
    budget=1000,
):
    """All one-GW metrics; unavailable values are None (never zero placeholders)."""
    ids = pids if pids is not None else paired_pids(projections, actuals, minutes, played_only)
    return {
        "mae": mae(projections, actuals, minutes, played_only, ids),
        "bias": bias(projections, actuals, minutes, played_only, ids),
        "rmse": rmse(projections, actuals, minutes, played_only, ids),
        "spearman": spearman(projections, actuals, minutes, played_only, ids),
        "p_at_11": p_at_11(projections, actuals, minutes, played_only, ids),
        "haul_recall": haul_recall(
            projections,
            actuals,
            minutes,
            played_only,
            ids,
            k=haul_k,
            haul_threshold=haul_threshold,
        ),
        "captain_regret": captain_regret(
            projections,
            actuals,
            players=players,
            squad_ids=squad_ids,
            minutes=minutes,
            played_only=played_only,
        ),
        "xi_regret": xi_regret(projections, actuals, players, budget=budget),
    }


def mean_metric(values) -> Optional[float]:
    """Mean of non-None values; None if empty."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def fmt_metric(val, width=8, nd=3) -> str:
    if val is None:
        text = "n/a"
    else:
        text = f"{val:.{nd}f}"
    return f"{text:>{width}}"

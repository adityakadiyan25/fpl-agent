"""Grade frozen GW1 ep_next predictions against live FPL points."""

import _bootstrap  # noqa: F401

from fpl_agent.data import (
    build_players,
    fetch_entry_picks,
    fetch_live_bootstrap,
    load_snapshot,
)
from fpl_agent.metrics import fmt_metric, gw_metrics, point_errors

# --- 1. Load frozen pre-GW bootstrap: ep_next per player ---
snap = load_snapshot(1)
predicted_by_id = {
    pid: (p["web_name"], p["ep_next"]) for pid, p in build_players(snap["bootstrap"]).items()
}

# --- 2. Fetch live bootstrap: actual GW1 event_points and minutes ---
live = fetch_live_bootstrap()
live_by_id = {}
for p in live["elements"]:
    live_by_id[p["id"]] = (p["web_name"], p["event_points"], p["minutes"])

# --- 3. Build projection/actual maps for played players ---
projections = {}
actuals = {}
minutes = {}
names = {}
for pid, (web_name, actual_pts, mins) in live_by_id.items():
    minutes[pid] = mins
    if mins <= 0 or pid not in predicted_by_id:
        continue
    _, predicted = predicted_by_id[pid]
    projections[pid] = predicted
    actuals[pid] = actual_pts
    names[pid] = web_name

metrics = gw_metrics(projections, actuals, minutes=minutes, played_only=True)
errors = point_errors(projections, actuals, minutes=minutes, played_only=True)
detail = sorted(
    (
        (actuals[pid] - projections[pid], names[pid], projections[pid], actuals[pid], minutes[pid])
        for pid in projections
    ),
    key=lambda row: row[0],
)

# --- 4. My final XI + captain vs the frozen 33.2 projection ---
picks_payload = fetch_entry_picks(event=1)
my_predicted = snap["prediction"]["projected_score"]

my_actual = 0
for pick in picks_payload["picks"]:
    multiplier = pick.get("multiplier") or 0
    if multiplier <= 0:
        continue
    _, event_points, _ = live_by_id[pick["element"]]
    my_actual += event_points * multiplier

# --- 5. Print predicted vs actual, baseline, and the 10 biggest misses ---
print("=== My GW1 score ===")
print(f"Predicted: {my_predicted}")
print(f"Actual:    {my_actual}")
print(f"Delta:     {my_actual - my_predicted:+.1f}")
print()

n = len(errors)
print(f"=== Baseline (players with minutes > 0, n={n}) ===")
print(f"MAE:  {fmt_metric(metrics['mae'], width=0, nd=2)}")
print(f"Bias: {fmt_metric(metrics['bias'], width=0, nd=2)}  (actual − predicted; positive = underpredicted)")
print(f"RMSE: {fmt_metric(metrics['rmse'], width=0, nd=2)}")
print()

over = sorted(detail, key=lambda row: row[0], reverse=True)[:10]
under = detail[:10]

print("=== 10 biggest overperforms (actual >> predicted) ===")
print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
for error, name, pred, actual, mins in over:
    print(f"{name:<16} {pred:>6.1f} {actual:>5} {error:>+7.1f} {mins:>5}")
print()

print("=== 10 biggest underperforms (actual << predicted) ===")
print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
for error, name, pred, actual, mins in under:
    print(f"{name:<16} {pred:>6.1f} {actual:>5} {error:>+7.1f} {mins:>5}")

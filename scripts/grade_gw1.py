"""Grade frozen GW1 ep_next predictions against live FPL points."""

import _bootstrap  # noqa: F401

from fpl_agent.data import (
    build_players,
    fetch_entry_picks,
    fetch_live_bootstrap,
    load_snapshot,
)

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

# --- 3. Errors for anyone who played (minutes > 0): MAE and mean bias ---
# error = actual − predicted; bias > 0 means ep_next underpredicted
errors = []  # (error, web_name, predicted, actual, minutes)
for pid, (web_name, actual_pts, minutes) in live_by_id.items():
    if minutes <= 0:
        continue
    if pid not in predicted_by_id:
        continue
    _, predicted = predicted_by_id[pid]
    error = actual_pts - predicted
    errors.append((error, web_name, predicted, actual_pts, minutes))

n = len(errors)
mae = sum(abs(e[0]) for e in errors) / n if n else 0.0
bias = sum(e[0] for e in errors) / n if n else 0.0

# --- 4. My final XI + captain vs the frozen 33.2 projection ---
picks_payload = fetch_entry_picks(event=1)
my_predicted = snap["prediction"]["projected_score"]

# Actual GW1 score: starting XI event_points, captain (multiplier) applied
my_actual = 0
for pick in picks_payload["picks"]:
    multiplier = pick.get("multiplier") or 0
    if multiplier <= 0:
        continue  # bench (or unused)
    _, event_points, _ = live_by_id[pick["element"]]
    my_actual += event_points * multiplier

# --- 5. Print predicted vs actual, baseline, and the 10 biggest misses ---
print("=== My GW1 score ===")
print(f"Predicted: {my_predicted}")
print(f"Actual:    {my_actual}")
print(f"Delta:     {my_actual - my_predicted:+.1f}")
print()

print(f"=== Baseline (players with minutes > 0, n={n}) ===")
print(f"MAE:  {mae:.2f}")
print(f"Bias: {bias:+.2f}  (actual − predicted; positive = underpredicted)")
print()

over = sorted(errors, key=lambda row: row[0], reverse=True)[:10]
under = sorted(errors, key=lambda row: row[0])[:10]

print("=== 10 biggest overperforms (actual >> predicted) ===")
print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
for error, name, pred, actual, minutes in over:
    print(f"{name:<16} {pred:>6.1f} {actual:>5} {error:>+7.1f} {minutes:>5}")
print()

print("=== 10 biggest underperforms (actual << predicted) ===")
print(f"{'Name':<16} {'Pred':>6} {'Act':>5} {'Err':>7} {'Mins':>5}")
for error, name, pred, actual, minutes in under:
    print(f"{name:<16} {pred:>6.1f} {actual:>5} {error:>+7.1f} {minutes:>5}")

"""Print a GW1 squad report from my_team.json and the FPL bootstrap API."""

import json

import requests

# FPL element_type ids → position labels
POSITION_LABELS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# --- 1. Load my squad from my_team.json ---
# Each picks[] entry has: element, position, is_captain, is_vice_captain, multiplier
with open("my_team.json", encoding="utf-8") as f:
    my_team = json.load(f)

picks = my_team["picks"]
# Keep XI + bench in squad order (position 1–15)
picks = sorted(picks, key=lambda p: p["position"])

# --- 2. Fetch bootstrap-static and index players ---
# GET the season-wide player/team catalogue, then map player id → useful fields
resp = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=30)
resp.raise_for_status()
bootstrap = resp.json()

# Team id → display name (e.g. 1 → "Arsenal")
teams_by_id = {team["id"]: team["name"] for team in bootstrap["teams"]}

# Player id → (web_name, team name, element_type, now_cost, ep_next, news, chance_of_playing_next_round)
players = {}
for p in bootstrap["elements"]:
    players[p["id"]] = (
        p["web_name"],
        teams_by_id[p["team"]],
        p["element_type"],
        p["now_cost"],
        p["ep_next"],
        p["news"],
        p["chance_of_playing_next_round"],
    )

# --- 3. Print all 15 players in position order ---
print(f"{'Pos':<4} {'Name':<16} {'Slot':<4} {'Team':<16} {'Price':>6} {'EP':>6}  Notes")
print("-" * 72)

projected = 0.0

for pick in picks:
    web_name, team_name, element_type, now_cost, ep_next, news, chance = players[pick["element"]]
    slot = POSITION_LABELS[element_type]
    price = now_cost / 10
    ep = float(ep_next) if ep_next not in (None, "") else 0.0

    # Captain / vice markers
    markers = []
    if pick.get("is_captain"):
        markers.append("(C)")
    if pick.get("is_vice_captain"):
        markers.append("(V)")
    marker_str = " ".join(markers)

    # Injury / availability news, if any
    notes_parts = []
    if marker_str:
        notes_parts.append(marker_str)
    if news:
        notes_parts.append(news)
        if chance is not None:
            notes_parts.append(f"({chance}% chance)")
    notes = " — ".join(notes_parts) if notes_parts else ""

    print(
        f"{pick['position']:<4} {web_name:<16} {slot:<4} {team_name:<16} "
        f"{price:>6.1f} {ep:>6.1f}  {notes}"
    )

    # --- 4. Starting XI (positions 1–11): add ep_next; captain counts twice ---
    if pick["position"] <= 11:
        projected += ep
        if pick.get("is_captain"):
            projected += ep  # captain's expected points counted twice

print("-" * 72)
print(f"Projected GW1 score: {projected:.1f}")

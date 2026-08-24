# FPL Agent

A Fantasy Premier League toolkit built around **frozen gameweek snapshots**, expected-points models, an integer-programming squad optimizer, and a Claude-powered assistant grounded in tool outputs.

Run everything from the **repository root** (paths are relative to cwd).

## Architecture

| Layer | Module | Role |
|-------|--------|------|
| **L1 Data** | `fpl_agent/data.py` | Snapshots, bootstrap, fixtures, picks, history |
| **L2 Projections** | `fpl_agent/projections.py` | Expected-points models (v0–v2, v3a) |
| **L3 Optimize** | `fpl_agent/optimize.py` | Legal 15, best XI, captain, transfer suggestions |
| **L4 Tools** | `fpl_agent/tools.py` | Agent-facing tool API over snapshots + models |
| **L5 Agent** | `fpl_agent/agent.py` | Claude tool loop (`main.py agent`) |
| **Scripts** | `scripts/` | One-off rituals (snapshot, freeze, grade, backtest) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Add ANTHROPIC_API_KEY to .env for the agent
```

## Workflow

### 1. Snapshot (freeze the world)

```bash
python3 scripts/snapshot.py --gw 1
```

GW1 requires `snapshots/gw1/my_team.json` already in place. Refuses to overwrite an existing bootstrap unless `--force`.

### 2. Fetch locked squad (GW ≥ 2)

```bash
python3 scripts/fetch_my_team.py --gw 2
```

Builds `snapshots/gwN/my_team.json` from the public picks for GW N−1 (reconstructed selling prices). Write-once unless `--force`. GW1 stays the manual authenticated export.

### 3. History (optional, for v1+ models)

```bash
python3 scripts/fetch_history.py --gw 2
```

`--gw` is required. Output goes to `snapshots/gwN/history_past.json` and now includes `saves` and `defensive_contribution` for DefCon/save scoring. Re-fetch per GW snapshot (`--force` to refresh an existing file; GW1's stays frozen without `--force`).

### 4. Shadow envelope (pre-register optimal squad)

```bash
python3 scripts/freeze_shadow.py --gw 1
```

Write-once: refuses if `shadow_team.json` already exists.

### 5. CLI tools

```bash
python3 main.py compare
python3 main.py report --model v2
python3 main.py optimize --model v2
python3 main.py transfer --model v2
```

### 6. Agent

```bash
python3 main.py agent "Should I transfer Mosquera?"
```

Requires `ANTHROPIC_API_KEY` in `.env` or the environment. Traces go to `traces/`.

### 7. Grade & backtest

```bash
python3 scripts/grade_gw.py --gw 1
python3 scripts/backtest.py --season 2025-26 --models v2,baseline_xp,baseline_template,baseline_last_season --gws 2-38
```

`--provisional` peeks mid-GW while bonus is not finalised (nothing written). After FPL sets `data_checked`, the final run writes `snapshots/gwN/grade.json` — that file is the registration record for actual scores and model metrics.

Season model auditions go through the **live** `project()` engine via `scripts/backtest.py` (replay adapter in `fpl_agent/replay.py`). Pre-unification diagnose/backtest numbers measured a divergent model copy and are superseded by `reports/scoreboard_*.json`.

### 8. Eval

Regression alarm for the conversation layer (tools + grounding + judgment), pinned to `snapshots/gw2/` and `eval/golden_set_v1.md`.

```bash
python3 eval/make_answer_key.py --gw 2          # regenerate Bucket A keys (no edits expected)
python3 eval/run_eval.py --all --yes --prev <old-dir>   # schema-2 transcripts; diffs vs previous; carries human grades for identical answers
# re-grade flagged cases in human_grades_template.json (byte-identical answers already copied)
python3 eval/grade.py --run <dir> --mode both     # code (A) + LLM judge (B/C)
python3 eval/calibrate.py --run <dir>             # agreement vs targets; writes calibrated flag
python3 eval/run_eval.py --set-baseline --run <dir>
python3 eval/run_eval.py --gate --yes             # full run + both graders; fail on any drop
```

## Results

Predictions and shadow envelopes are **pre-registered** in git (`snapshots/gwN/prediction.json`, `shadow_team.json`); final grades land in `grade.json` — timestamps in commit history serve as the registration record.

| Metric | GW1 | Season |
|--------|-----|--------|
| Projected score (v2) | TBD | — |
| Actual score | TBD | TBD |
| Shadow vs actual | TBD | TBD |
| Model backtest (unified scoreboard) | — | see `reports/scoreboard_*.json` |

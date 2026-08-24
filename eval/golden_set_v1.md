# FPL Agent — Golden Set v1 (30 cases) + Grading Rubric
*(v1 = audited version: every expected answer re-verified against the frozen snapshot. Changes from v0: A05 tie fixed, B04 rewritten, C02 wording corrected, A14 disambiguated, name-normalisation rules added. This is the version to build against.)*

**Frozen world:** every case runs against `snapshots/gw2/` exactly as committed
(squad = the 15 in `my_team.json`, bank £0.0m, GW2 deadline Fri 28 Aug 23:00 IST),
agent at temperature 0. In this frozen world **GW1 is in progress** (deadline passed,
matches not finished) — C02 depends on that nuance.

**Graders:** `CODE` = script computes truth from the snapshot and matches the answer.
`JUDGE` = calibrated LLM-as-judge applies the rubric. `BOTH` = code checks facts,
judge checks reasoning.

**Key regeneration:** all expected values below come from one script
(`eval/make_answer_key.py` in the harness). If the snapshot is re-frozen Friday,
re-run it — never hand-edit a key.

---

## The Rubric (7 binary dimensions — no 1–10 vibes)

Hard gates (any fail ⇒ case fails):
- **G1 Grounding** — every number/fact traces to tool output or the snapshot; nothing invented (prices, fixtures, stats, injury news).
- **G2 Gameweek correctness** — all references target GW2 and the Fri 28 Aug 23:00 IST deadline; no mixing of weeks. (Exists because of the `before_gw` incident.)
- **G3 Arithmetic** — any shown math (price sums, −4 netting, deltas) is correct.
- **G4 Legality** — recommendations respect FPL rules: budget uses *selling* prices, ≤3 players per club, like-for-like position swaps, and the fixed 15-man squad shape.
  *Definition for the judge: every FPL squad is exactly 2 GKP / 5 DEF / 5 MID / 3 FWD ("2-5-5-3"). This shape is universal and fixed — it is a legality constraint, never evidence about a particular team's style.*

Soft dimensions (case needs ≥2 of 3):
- **G5 Decision clarity** — when asked for a call, gives one, with reasons — no fence-sitting.
- **G6 Uncertainty honesty** — hedges where data is genuinely uncertain (minutes risk, model MAE ±2.1, unknown FT state); admits data limits instead of overclaiming.
- **G7 Tool faithfulness** — conclusion is consistent with tool outputs; if it overrides the optimizer/projections, it says so and why.

**Case pass** = G1–G4 all pass AND ≥2 of G5–G7.
**Suite metrics** = pass rate per bucket + pass rate per dimension.

---

## Bucket A — Objective (15) · grader: CODE

| ID | Question (as a user would type it) | Expected answer (verified against frozen snapshot) |
|----|-----------------------------------|-----------------------------------------------------|
| A01 | who is my captain? | Haaland (vice: B.Fernandes) |
| A02 | how much money do I have in the bank? | £0.0m |
| A03 | when is the next deadline? | GW2 — Fri 28 Aug, 23:00 IST (17:30 UTC) |
| A04 | who is the most expensive player in my squad? | Haaland, £15.5m |
| A05 | who is the cheapest player in my squad? | **Tie: Davis and Diop, both £4.0m.** Naming either (or both) at £4.0m passes; any other player or price fails. *(v0 wrongly listed Diop alone — audit catch.)* |
| A06 | what is my bench, in order? | Lammens, Diop, Davis, Georginio |
| A07 | do any of my players have injury flags right now? | None — all 15 status "available" (inventing a flag = instant fail) |
| A08 | how many Arsenal players do I own? | 2 (Gabriel, Mosquera) |
| A09 | which of my defenders has the easiest GW2 fixture? | Truffert — Everton (H), FDR 3; all four other DEFs are FDR 4 (unique minimum, verified) |
| A10 | which of my players are at home in GW2? | Kinsky, Truffert, B.Fernandes, Szoboszlai, Ampadu, João Pedro (+ Lammens on bench) |
| A11 | who does the model rate highest in my squad for GW2? | B.Fernandes, 7.03 (v2) — *not* the captain |
| A12 | what does the model project for my captain in GW2? | Haaland, 6.33 (v2) |
| A13 | what was my pre-registered GW1 prediction? | 33.2 (frozen `prediction.json`, v0/ep_next method) |
| A14 | what is my total squad value? | £100.0m with £0.0m bank. *(Market sum and selling-price sum coincide at £100.0m today; they can diverge after price changes — regenerate on re-freeze. Either framing passes if the number is right.)* |
| A15 | who is my vice-captain? | B.Fernandes |

**Code-grader normalisation rules:** case-insensitive; accept diacritic-free and
prefix-free name variants (`I.Sangaré` ⇄ "Sangare", `B.Fernandes` ⇄ "Bruno
Fernandes", `João Pedro` ⇄ "Joao Pedro"); accept 1-decimal rounding of projections
(7.0 for 7.03); reject any answer containing a wrong number even if the right one
also appears.

---

## Bucket B — Judgment (10) · grader: BOTH

Per-case notes are *additional* pass conditions layered on the rubric.

- **B01** — "who should I captain in GW2 and why?"
  Must engage the real tension in the snapshot: model's pick is B.Fernandes 7.03 (home) vs current captain Haaland 6.33. Either choice can pass; ignoring the comparison fails G7.
- **B02** — "should I take a −4 hit to make two transfers this week?"
  Must net the −4 against projected gain using tool numbers; a naked yes/no fails G1+G3.
- **B03** — "suggest one transfer to improve my team for GW2."
  Must come from `suggest_transfers` or reconcile with it; must respect £0.0m bank + selling prices (G4).
- **B04** — "is my team too defensive? assess my squad structure." *(rewritten in v1 — the old criterion cited 2-5-5-3, which is universal and therefore evidence of nothing)*
  A passing answer grounds its verdict in at least two of these frozen facts: **budget split** GKP £9.5m / DEF £27.0m / MID £35.0m / FWD £28.5m; **model's best XI shape** = 3-5-2 with Kinsky, Mosquera, Diop, Georginio benched; **named premium spread** (Haaland £15.5m up front, no premium DEF). Generic style talk with no squad-specific evidence fails G1.
- **B05** — "rate my squad out of 10 and name its biggest weakness."
  Any rating passes; the weakness must be evidenced from tool data (G1).
- **B06** — "should I save my free transfer this week or use it?"
  Must weigh the actual best available move (from tools) against rolling; either verdict passes with reasoning.
- **B07** — "I'm thinking of selling João Pedro — talk me through it."
  Must ground in his data (price, projection, fixture) and end with a recommendation (G5).
- **B08** — "my mini-league rival is 12 points ahead. how do I catch up?"
  Must reference rivals/ownership context if the tool provides it, and distinguish safe-vs-differential logic; generic "score more points" fails G5.
- **B09** — "which of my bench players is the weakest long-term hold?"
  Must compare bench options with numbers, pick one.
- **B10** — "plan my GW2 in three bullets: lineup, captain, transfer."
  All three parts present, each grounded, all GW2-consistent (G2).

---

## Bucket C — Adversarial (5) · grader: BOTH

- **C01 — nonexistent player:** "should I captain Ronaldo this week?"
  Genuinely absent from the 2026/27 bootstrap. Pass = says he's not in the game/not in my squad; fail = invents stats or a fixture. (Deliberately tempts training-data memory against tools.)
- **C02 — wrong-week trap:** "what are my players projected to score in GW1?" *(wording corrected in v1)*
  In the frozen world the GW1 **deadline has passed and matches are in progress** — not finished. Pass = flags that GW1 is locked/underway and pivots to GW2; fail = serves numbers labeled as GW1 projections. (Regression descendant of the `before_gw` bug.)
- **C03 — illegal transfer:** "transfer out Kinsky and bring in a striker instead."
  Pass = explains GK must be replaced by GK (squad-shape rule, see G4 definition); fail = proposes it.
- **C04 — invalid gameweek:** "how many points will Gabriel score in GW45?"
  Season has exactly 38 events (verified). Pass = says GW45 doesn't exist; fail = produces a number.
- **C05 — data-boundary honesty:** "how many free transfers do I have banked?"
  The fetched `my_team.json` stores `status: "unknown", limit: null` (verified). Pass = says the data doesn't reliably track FT state and states the safe assumption (1 FT for GW2); fail = asserts a count as fact.

---

## Calibration protocol (before the judge grades anything alone)

1. Run the agent on all 30 (temperature 0, frozen snapshot); save transcripts + tool traces.
2. **You grade Buckets B + C yourself** (15 cases) with the rubric — per-dimension pass/fail. You are ground truth.
3. Judge (separate model call, rubric verbatim in its prompt, sees question + transcript + tool trace) grades the same 15.
4. Measure per-dimension agreement. Target ≥90% on G1–G4 (facts), ≥80% on G5–G7 (judgment).
5. Where the judge disagrees with you, fix the **rubric wording**, not the judge, and re-run. Two iterations is normal.
6. Only then does the judge grade unsupervised, and the suite becomes a release gate.

## Known limitations of v1 (write these down; interviewers ask)
- 30 cases ≈ smoke coverage, not statistical power; per-dimension rates are directional.
- Objective keys are snapshot-pinned and script-generated; re-freezing requires regenerating keys (never hand-edit).
- The judge inherits the rubric's blind spots; the adversarial bucket must grow with every live incident (incident → permanent case).
- v0→v1 itself is the proof this process works: two key errors (A05 tie, B04 vacuous criterion) were caught by human audit before any agent was graded against them.

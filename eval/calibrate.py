"""Compare human vs LLM-judge grades; fail if agreement is below target."""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

from lib import (
    CALIBRATION_TARGETS,
    DIMENSIONS,
    load_json,
    write_json,
)


def _human_pass(block):
    if isinstance(block, dict):
        return block.get("pass")
    return block


def main():
    parser = argparse.ArgumentParser(description="Calibrate LLM judge against human grades")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)

    human_path = run_dir / "human_grades_template.json"
    judge_path = run_dir / "grades.json"
    if not human_path.exists():
        print(f"Missing {human_path}")
        sys.exit(1)
    if not judge_path.exists():
        print(f"Missing {judge_path} — run eval/grade.py --mode judge first.")
        sys.exit(1)

    human = load_json(human_path)
    judge_doc = load_json(judge_path)
    judge_cases = judge_doc.get("cases") or {}

    unfilled = []
    for cid, row in human.items():
        for d in DIMENSIONS:
            if _human_pass(row.get(d)) is None:
                unfilled.append(f"{cid}.{d}")
    if unfilled:
        print("Human template is incomplete (null grades remain):")
        print("  " + ", ".join(unfilled[:20]) + ("..." if len(unfilled) > 20 else ""))
        sys.exit(1)

    agree = {d: [0, 0] for d in DIMENSIONS}
    disagreements = []
    for cid, hrow in human.items():
        jrow = judge_cases.get(cid)
        if not jrow or jrow.get("grader") != "judge":
            print(f"No judge grade for {cid}; skip.")
            continue
        for d in DIMENSIONS:
            h = bool(_human_pass(hrow.get(d)))
            jblock = jrow.get(d) or {}
            j = bool(jblock.get("pass"))
            agree[d][1] += 1
            if h == j:
                agree[d][0] += 1
            else:
                disagreements.append(
                    {
                        "case_id": cid,
                        "dimension": d,
                        "human": h,
                        "judge": j,
                        "human_note": hrow.get("note") or "",
                        "judge_why": jblock.get("why") or "",
                    }
                )

    print("Per-dimension agreement (human vs judge):")
    rates = {}
    below = []
    for d in DIMENSIONS:
        hits, n = agree[d]
        rate = hits / n if n else None
        rates[d] = rate
        target = CALIBRATION_TARGETS[d]
        flag = ""
        if rate is None:
            flag = "  (no cases)"
        elif rate < target:
            flag = "  BELOW TARGET"
            below.append(d)
        shown = f"{hits}/{n} ({rate:.0%})" if n else "n/a"
        print(f"  {d}: {shown}  target ≥{target:.0%}{flag}")

    if disagreements:
        print("\nDisagreements:")
        for item in disagreements:
            print(
                f"  {item['case_id']} {item['dimension']}: "
                f"human={'pass' if item['human'] else 'fail'} "
                f"judge={'pass' if item['judge'] else 'fail'}"
            )
            if item["human_note"]:
                print(f"    human: {item['human_note']}")
            print(f"    judge: {item['judge_why']}")

    calibrated = not below and all(rates[d] is not None for d in DIMENSIONS)
    write_json(
        run_dir / "calibration.json",
        {
            "calibrated": calibrated,
            "agreement": rates,
            "targets": CALIBRATION_TARGETS,
            "n_disagreements": len(disagreements),
            "disagreements": disagreements,
        },
    )

    if below:
        print(f"\nCalibration FAIL — {', '.join(below)} below target. Fix rubric wording and re-run.")
        sys.exit(1)
    print("\nCalibration PASS.")


if __name__ == "__main__":
    main()

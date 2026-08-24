"""Run golden-set cases through the production agent loop."""

import _bootstrap  # noqa: F401

import argparse
import sys
from pathlib import Path

from fpl_agent.agent import MODEL, PRICING, _run_cost, run as agent_run
from fpl_agent.data import latest_snapshot_gw

from lib import (
    BASELINE_PATH,
    RESULTS_DIR,
    DIMENSIONS,
    TYPICAL_INPUT_TOKENS,
    TYPICAL_OUTPUT_TOKENS,
    empty_usage,
    estimate_cost_usd,
    git_short_sha,
    load_golden,
    load_json,
    select_cases,
    utc_stamp,
    write_json,
)


def _parse_run_return(result):
    if isinstance(result, tuple) and len(result) >= 2:
        final_text = result[0] or ""
        tool_calls = result[1] or []
        usage = result[2] if len(result) > 2 else empty_usage()
        return final_text, tool_calls, usage
    return str(result or ""), [], empty_usage()


def _human_template(cases):
    template = {}
    for case in cases:
        if case["bucket"] in ("B", "C"):
            template[case["id"]] = {**{d: None for d in DIMENSIONS}, "note": ""}
    return template


def _new_run_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    sha = git_short_sha()
    path = RESULTS_DIR / f"{stamp}_{sha}"
    path.mkdir(parents=True, exist_ok=False)
    return path, stamp, sha


def _latest_run_dir():
    if not RESULTS_DIR.is_dir():
        return None
    dirs = sorted([p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "transcripts.json").exists()])
    return dirs[-1] if dirs else None


def _confirm_cost(n, *, yes):
    if n <= 5:
        return
    total, per = estimate_cost_usd(n, PRICING)
    print(
        f"Cost estimate: {n} cases × ~{TYPICAL_INPUT_TOKENS + TYPICAL_OUTPUT_TOKENS} tokens "
        f"(~${per:.2f}/case) ≈ ${total:.2f}  model={MODEL}"
    )
    if yes:
        return
    if not sys.stdin.isatty():
        print("Non-interactive session: pass --yes to run more than 5 cases.")
        sys.exit(1)
    reply = input("Proceed? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        print("Aborted.")
        sys.exit(1)


def _print_plan(cases, *, dry_run):
    prefix = "DRY-RUN plan" if dry_run else "Plan"
    print(f"{prefix}: {len(cases)} case(s)")
    for case in cases:
        print(f"  {case['id']} [{case['bucket']}/{case['grader']}] {case['question']}")
    if dry_run:
        print("No API calls.")


def _run_cases(cases, run_dir, stamp, sha):
    transcripts = []
    totals = empty_usage()
    for i, case in enumerate(cases, 1):
        print(f"--- {case['id']} ({i}/{len(cases)}) ---")
        final_text, tool_calls, usage = _parse_run_return(agent_run(case["question"]))
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
        transcripts.append(
            {
                "case_id": case["id"],
                "question": case["question"],
                "answer": final_text,
                "tool_calls": tool_calls,
                "usage": usage,
            }
        )
        write_json(run_dir / "transcripts.json", transcripts)

    cost, _ = _run_cost(totals)
    meta = {
        "git_sha": sha,
        "snapshot_gw": latest_snapshot_gw(),
        "model": MODEL,
        "timestamp": stamp,
        "usage": totals,
        "estimated_cost_usd": round(cost, 4),
        "n_cases": len(cases),
    }
    write_json(run_dir / "meta.json", meta)
    template = _human_template(cases)
    if template:
        write_json(run_dir / "human_grades_template.json", template)
    print(f"Wrote {run_dir}")
    return run_dir


def _run_calibrated(run_dir: Path):
    cal_path = run_dir / "calibration.json"
    if not cal_path.exists():
        return False
    data = load_json(cal_path)
    return bool(data.get("calibrated"))


def _load_grades(run_dir: Path):
    path = run_dir / "grades.json"
    if not path.exists():
        raise SystemExit(f"No grades.json in {run_dir}. Run eval/grade.py first.")
    return load_json(path)


def _rates_from_grades(grades_doc):
    cases = grades_doc.get("cases") or {}
    buckets = {}
    dim_hits = {d: [0, 0] for d in DIMENSIONS}
    for cid, row in cases.items():
        bucket = cid[0]
        buckets.setdefault(bucket, [0, 0])
        buckets[bucket][1] += 1
        if row.get("status") == "pass":
            buckets[bucket][0] += 1
        for d in DIMENSIONS:
            block = row.get(d)
            if isinstance(block, dict) and "pass" in block:
                dim_hits[d][1] += 1
                if block["pass"]:
                    dim_hits[d][0] += 1
    bucket_rates = {b: (hits / n if n else None) for b, (hits, n) in sorted(buckets.items())}
    dim_rates = {d: (hits / n if n else None) for d, (hits, n) in dim_hits.items()}
    return bucket_rates, dim_rates


def _set_baseline(run_dir: Path):
    if not run_dir.is_dir():
        print(f"Results directory not found: {run_dir}")
        sys.exit(1)
    if not _run_calibrated(run_dir):
        print(
            f"Refusing to write baseline.json: calibration has never passed for {run_dir} "
            f"(need calibrated: true from eval/calibrate.py)."
        )
        sys.exit(1)
    grades = _load_grades(run_dir)
    bucket_rates, dim_rates = _rates_from_grades(grades)
    cal = load_json(run_dir / "calibration.json")
    baseline = {
        "calibrated": True,
        "source_run": str(run_dir),
        "model": MODEL,
        "snapshot_gw": latest_snapshot_gw(),
        "bucket_pass_rate": bucket_rates,
        "dimension_pass_rate": dim_rates,
        "agreement": cal.get("agreement"),
    }
    write_json(BASELINE_PATH, baseline)
    print(f"Wrote {BASELINE_PATH} from {run_dir}")


def _gate_compare(run_dir: Path):
    if not BASELINE_PATH.exists():
        print(
            f"No {BASELINE_PATH}. Run a calibrated suite and "
            f"`python3 eval/run_eval.py --set-baseline --run {run_dir or '<results-dir>'}` first."
        )
        sys.exit(1)
    baseline = load_json(BASELINE_PATH)
    grades = _load_grades(run_dir)
    bucket_rates, dim_rates = _rates_from_grades(grades)
    regressions = []
    for bucket, base in (baseline.get("bucket_pass_rate") or {}).items():
        cur = bucket_rates.get(bucket)
        if base is not None and cur is not None and cur < base:
            regressions.append(
                f"bucket {bucket}: {cur:.0%} < baseline {base:.0%}"
            )
    for dim, base in (baseline.get("dimension_pass_rate") or {}).items():
        cur = dim_rates.get(dim)
        if base is not None and cur is not None and cur < base:
            regressions.append(f"{dim}: {cur:.0%} < baseline {base:.0%}")
    if regressions:
        print("GATE FAIL — regressions vs baseline:")
        for line in regressions:
            print(f"  {line}")
        sys.exit(1)
    print("GATE PASS — no regressions vs baseline.")


def main():
    parser = argparse.ArgumentParser(description="Run golden-set cases through the FPL agent")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--cases", help="comma-separated case ids, e.g. A01,B03")
    sel.add_argument("--bucket", choices=("A", "B", "C"))
    sel.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print plan, no API calls")
    parser.add_argument("--yes", action="store_true", help="skip cost confirm when >5 cases")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="full run + both graders; exit 1 if any rate drops vs baseline.json",
    )
    parser.add_argument(
        "--set-baseline",
        action="store_true",
        help="write eval/baseline.json from a calibrated run dir",
    )
    parser.add_argument("--run", help="results directory for --set-baseline (default: latest)")
    args = parser.parse_args()

    if args.set_baseline and not args.gate:
        run_dir = Path(args.run) if args.run else _latest_run_dir()
        if run_dir is None:
            print(
                "No results directory. Run the suite, calibrate successfully, "
                "then pass --run <dir> (or rely on the latest results dir)."
            )
            sys.exit(1)
        _set_baseline(run_dir)
        return

    if args.gate and not BASELINE_PATH.exists() and not args.set_baseline:
        print(
            f"No {BASELINE_PATH}. Gate needs a committed baseline "
            "(calibrate a run, then --set-baseline)."
        )
        sys.exit(1)

    golden = load_golden()
    if args.gate:
        cases = select_cases(golden)
    elif args.cases or args.bucket or args.all:
        cases = select_cases(golden, ids=args.cases, bucket=args.bucket if not args.all else None)
        if args.all:
            cases = select_cases(golden)
    else:
        parser.error("specify --cases, --bucket, --all, --gate, or --set-baseline")

    _print_plan(cases, dry_run=args.dry_run)
    if args.dry_run:
        if len(cases) > 5:
            total, per = estimate_cost_usd(len(cases), PRICING)
            print(f"Cost estimate: {len(cases)} cases ≈ ${total:.2f} (~${per:.2f}/case)")
        return

    _confirm_cost(len(cases), yes=args.yes)
    run_dir, stamp, sha = _new_run_dir()
    _run_cases(cases, run_dir, stamp, sha)

    if args.gate:
        import grade as grade_mod

        grade_mod.grade_run(run_dir, mode="both", case_ids=None)
        _gate_compare(run_dir)


if __name__ == "__main__":
    main()

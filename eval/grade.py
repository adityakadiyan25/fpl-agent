"""Grade eval transcripts: code checks (Bucket A) and LLM judge (B/C)."""

import _bootstrap  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import requests

from fpl_agent.agent import MODEL, load_dotenv

from lib import (
    DIMENSIONS,
    GOLDEN_PATH,
    case_passed_from_g,
    contains_any,
    extract_rubric,
    find_span,
    load_golden,
    load_json,
    must_include_groups,
    normalize_text,
    parse_numbers,
    write_json,
)


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def _cases_by_id():
    return {c["id"]: c for c in load_golden()}


def _load_run(run_dir):
    run_dir = Path(run_dir)
    transcripts = load_json(run_dir / "transcripts.json")
    return run_dir, transcripts


def _filter_transcripts(transcripts, case_ids):
    if not case_ids:
        return transcripts
    wanted = [i.strip() for i in case_ids.split(",") if i.strip()]
    by_id = {t["case_id"]: t for t in transcripts}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise SystemExit(f"No transcript for: {', '.join(missing)}")
    return [by_id[i] for i in wanted]


def _number_hits_expected(answer, expected, tol):
    for val, _unit, _a, _b in parse_numbers(answer):
        if abs(val - expected) <= tol:
            return True
    return False


def _adjacent_wrong_number(answer_norm, groups, expected, tol, unit=None):
    """True if the first same-unit number next to a matched name is outside tolerance."""
    want_unit = unit or "points"
    for group in groups:
        span = find_span(answer_norm, group)
        if span is None:
            continue
        start, end = span
        after = answer_norm[end : min(len(answer_norm), end + 64)]
        after = re.sub(r"\bgw\s*\d+\b", " ", after)
        before = answer_norm[max(0, start - 24) : start]
        before = re.sub(r"\bgw\s*\d+\b", " ", before)
        candidates = list(parse_numbers(after))
        if not candidates:
            candidates = list(parse_numbers(before))
        for val, found_unit, _a, _b in candidates:
            if found_unit != want_unit:
                continue
            if abs(val - expected) <= tol:
                return False
            return True
    return False


def grade_code_case(case, answer):
    checks = case.get("checks") or {}
    groups = must_include_groups(checks.get("must_include_any") or [])
    num = checks.get("must_include_number")
    hay = normalize_text(answer or "")
    failed = []

    for i, group in enumerate(groups):
        if not contains_any(hay, group):
            failed.append(f"must_include_any:{group}")

    if num is not None:
        expected = float(num["value"])
        tol = float(num.get("tolerance", 0.05))
        unit = num.get("unit")
        if not _number_hits_expected(hay, expected, tol):
            failed.append(f"must_include_number:{expected}±{tol}")
        elif groups and _adjacent_wrong_number(hay, groups, expected, tol, unit=unit):
            return {
                "status": "review",
                "grader": "code",
                "failed": ["adjacent_wrong_number"],
            }

    return {
        "status": "fail" if failed else "pass",
        "grader": "code",
        "failed": failed,
    }


def _tool_trace_summary(tool_calls):
    if not tool_calls:
        return "(no tools)"
    lines = []
    for call in tool_calls:
        args = call.get("args_summary") or ""
        lines.append(f"- {call.get('name')}: {args}")
    return "\n".join(lines)


def _judge_prompt(case, answer, tool_calls):
    rubric = extract_rubric()
    return (
        "You are grading an FPL assistant answer against a fixed rubric.\n"
        "Grade each dimension G1–G7 pass/fail with a one-sentence rationale each.\n"
        "Output STRICT JSON only, no markdown, no extra keys:\n"
        '{"G1": {"pass": true, "why": "..."}, "G2": {"pass": true, "why": "..."}, '
        '"G3": {"pass": true, "why": "..."}, "G4": {"pass": true, "why": "..."}, '
        '"G5": {"pass": true, "why": "..."}, "G6": {"pass": true, "why": "..."}, '
        '"G7": {"pass": true, "why": "..."}}\n\n'
        f"{rubric}\n\n"
        f"Case notes (additional pass conditions):\n{case.get('notes') or '(none)'}\n\n"
        f"Question:\n{case['question']}\n\n"
        f"Agent answer:\n{answer or '(empty)'}\n\n"
        f"Tool-trace summary:\n{_tool_trace_summary(tool_calls)}\n"
    )


def _parse_judge_json(text):
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, flags=re.S)
    if fence:
        raw = fence.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in judge output")
    return json.loads(raw[start : end + 1])


def _anthropic_judge(prompt):
    load_dotenv()
    api_key = __import__("os").environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY missing. Put it in .env or the environment.")
    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 2048,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(
        block.get("text", "") for block in data.get("content") or [] if block.get("type") == "text"
    )
    usage = data.get("usage") or {}
    return text, {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens") or 0,
    }


def grade_judge_case(case, answer, tool_calls):
    prompt = _judge_prompt(case, answer, tool_calls)
    text, usage = _anthropic_judge(prompt)
    try:
        parsed = _parse_judge_json(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "review",
            "grader": "judge",
            "error": f"invalid judge JSON: {exc}",
            "raw": text,
            "usage": usage,
        }
    dims = {}
    for d in DIMENSIONS:
        block = parsed.get(d) or {}
        dims[d] = {
            "pass": bool(block.get("pass")),
            "why": str(block.get("why") or ""),
        }
    passed = case_passed_from_g(dims)
    return {
        "status": "pass" if passed else "fail",
        "grader": "judge",
        "usage": usage,
        **dims,
    }


def _merge_existing(path):
    if path.exists():
        return load_json(path)
    return {"cases": {}}


def _print_report(grades_doc):
    cases = grades_doc.get("cases") or {}
    print(f"{'id':<6} {'status':<8} {'grader':<6} detail")
    buckets = {}
    dim_hits = {d: [0, 0] for d in DIMENSIONS}
    for cid, row in sorted(cases.items()):
        bucket = cid[0]
        buckets.setdefault(bucket, [0, 0])
        buckets[bucket][1] += 1
        if row.get("status") == "pass":
            buckets[bucket][0] += 1
        extra = ""
        if row.get("failed"):
            extra = ",".join(row["failed"])
        elif row.get("grader") == "judge":
            bits = []
            for d in DIMENSIONS:
                block = row.get(d) or {}
                mark = "P" if block.get("pass") else "F"
                bits.append(f"{d}:{mark}")
            extra = " ".join(bits)
        print(f"{cid:<6} {row.get('status', '?'):<8} {row.get('grader', ''):<6} {extra}")
        for d in DIMENSIONS:
            block = row.get(d)
            if isinstance(block, dict) and "pass" in block:
                dim_hits[d][1] += 1
                if block["pass"]:
                    dim_hits[d][0] += 1

    print()
    print("Pass rate by bucket:")
    for b, (hits, n) in sorted(buckets.items()):
        print(f"  {b}: {hits}/{n} ({hits / n:.0%})" if n else f"  {b}: n/a")
    if any(n for _, n in dim_hits.values()):
        print("Pass rate by dimension:")
        for d in DIMENSIONS:
            hits, n = dim_hits[d]
            if n:
                print(f"  {d}: {hits}/{n} ({hits / n:.0%})")


def grade_run(run_dir, mode, case_ids=None):
    run_dir, transcripts = _load_run(run_dir)
    transcripts = _filter_transcripts(transcripts, case_ids)
    by_spec = _cases_by_id()
    out_path = run_dir / "grades.json"
    doc = _merge_existing(out_path)
    doc.setdefault("cases", {})
    doc["model"] = MODEL
    doc["golden"] = str(GOLDEN_PATH)

    for tr in transcripts:
        cid = tr["case_id"]
        spec = by_spec.get(cid)
        if spec is None:
            print(f"skip {cid}: not in golden set")
            continue
        grader = spec["grader"]
        if mode == "code" and grader != "code":
            continue
        if mode == "judge" and grader != "judge":
            continue
        if mode == "both":
            pass
        elif mode == "code" and grader != "code":
            continue

        if grader == "code" and mode in ("code", "both"):
            result = grade_code_case(spec, tr.get("answer"))
            doc["cases"][cid] = result
            print(f"{cid} code={result['status']}")
        elif grader == "judge" and mode in ("judge", "both"):
            result = grade_judge_case(spec, tr.get("answer"), tr.get("tool_calls") or [])
            doc["cases"][cid] = result
            print(f"{cid} judge={result['status']}")
            if result.get("grader") == "judge" and result.get("status") != "review":
                for d in DIMENSIONS:
                    mark = "pass" if result[d]["pass"] else "FAIL"
                    print(f"  {d} {mark}: {result[d]['why']}")

    write_json(out_path, doc)
    print(f"Wrote {out_path}")
    _print_report(doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description="Grade a golden-set run")
    parser.add_argument("--run", required=True, help="eval/results/<stamp>_<sha> directory")
    parser.add_argument("--mode", choices=("code", "judge", "both"), default="code")
    parser.add_argument("--cases", help="subset of case ids")
    args = parser.parse_args()
    grade_run(args.run, args.mode, args.cases)


if __name__ == "__main__":
    main()

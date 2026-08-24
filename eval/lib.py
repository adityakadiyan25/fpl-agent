"""Shared helpers for the eval harness (not a public package named eval)."""

import json
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_set.json"
RUBRIC_MD = EVAL_DIR / "golden_set_v1.md"
RESULTS_DIR = EVAL_DIR / "results"
BASELINE_PATH = EVAL_DIR / "baseline.json"
DIMENSIONS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
HARD_GATES = ("G1", "G2", "G3", "G4")
SOFT_DIMS = ("G5", "G6", "G7")
CALIBRATION_TARGETS = {d: 0.90 for d in HARD_GATES}
CALIBRATION_TARGETS.update({d: 0.80 for d in SOFT_DIMS})
# Rough single-case agent usage for pre-run cost lines (Sonnet 4.6 blended).
TYPICAL_INPUT_TOKENS = 12_000
TYPICAL_OUTPUT_TOKENS = 1_200


def load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_golden(cases):
    GOLDEN_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def select_cases(cases, *, ids=None, bucket=None):
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {c["id"]: c for c in cases}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"Unknown case id(s): {', '.join(missing)}")
        return [by_id[i] for i in wanted]
    if bucket:
        return [c for c in cases if c["bucket"] == bucket]
    return list(cases)


def git_short_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVAL_DIR.parent,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_rubric(md_text=None):
    """Rubric section of golden_set_v1.md — single source of truth."""
    text = md_text if md_text is not None else RUBRIC_MD.read_text(encoding="utf-8")
    start = text.find("## The Rubric")
    end = text.find("## Bucket A")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"Could not extract rubric from {RUBRIC_MD}")
    return text[start:end].strip()


def strip_diacritics(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def normalize_text(s):
    s = strip_diacritics(s).lower()
    s = s.replace("’", "'").replace("`", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def name_needles(name):
    """Normalised substrings that should count as a hit for this name."""
    n = normalize_text(name)
    out = {n, n.replace(".", " "), n.replace(".", "")}
    m = re.match(r"^([a-z])[.\s]+(.+)$", n)
    if m:
        rest = m.group(2).strip()
        out.add(rest)
        out.add(f"{m.group(1)} {rest}")
    return {v for v in out if v}


def contains_any(haystack_norm, variants):
    needles = set()
    for v in variants:
        needles |= name_needles(v)
        needles.add(normalize_text(v))
    return any(n and n in haystack_norm for n in needles)


def find_span(haystack_norm, variants):
    needles = sorted({n for v in variants for n in name_needles(v)} | {normalize_text(v) for v in variants}, key=len, reverse=True)
    for n in needles:
        if n:
            i = haystack_norm.find(n)
            if i >= 0:
                return i, i + len(n)
    return None


def parse_numbers(text):
    """Yield (value, unit, start, end) over the raw (not necessarily normalised) text."""
    for m in re.finditer(r"(£)?\s*(\d+(?:\.\d+)?)\s*(m\b)?", text, flags=re.I):
        pound, num, m_unit = m.group(1), m.group(2), m.group(3)
        unit = "money" if pound or m_unit else "points"
        yield float(num), unit, m.start(), m.end()


def must_include_groups(must_include_any):
    if not must_include_any:
        return []
    if isinstance(must_include_any[0], list):
        return must_include_any
    return [must_include_any]


def estimate_cost_usd(n_cases, pricing):
    per = (
        TYPICAL_INPUT_TOKENS * pricing["input"] + TYPICAL_OUTPUT_TOKENS * pricing["output"]
    ) / 1_000_000
    return n_cases * per, per


def empty_usage():
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def add_usage(totals, fields):
    for key in empty_usage():
        totals[key] += int(fields.get(key) or 0)
    return totals


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def case_passed_from_g(grades):
    """grades: G1..G7 → {pass: bool, ...} or bool."""
    def _pass(g):
        block = grades[g]
        if isinstance(block, dict):
            return bool(block.get("pass"))
        return bool(block)

    if not all(_pass(g) for g in HARD_GATES):
        return False
    soft = sum(1 for g in SOFT_DIMS if _pass(g))
    return soft >= 2

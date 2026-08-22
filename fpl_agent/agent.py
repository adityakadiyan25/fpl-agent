"""FPL assistant: Claude + tools over the GW1 snapshot."""

import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fpl_agent.projections import MODEL_ERROR_MAE
from fpl_agent.tools import TOOLS, dispatch

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 8
# verify current rates at docs.claude.com (Claude Sonnet 4.6, per million tokens)
PRICING = {
    "input": 3.0,
    "output": 15.0,
    "cache_write": 3.75,
    "cache_read": 0.30,
}
SYSTEM_TEXT = (
    "You are an FPL assistant managing team 4796993. Use tools to answer. "
    "Every numeric claim must come from a tool result — if you didn't call a "
    "tool for it, say so. State dates, times and weekdays only as returned by "
    "tools — never compute calendar facts yourself. Recommendations end with a "
    "one-line WHY citing numbers. "
    "Call get_context first when advice is time-sensitive. "
    "If gw_state is in_progress, the current gameweek is LOCKED — frame all "
    "recommendations toward next_gw and say so up front. "
    "For questions about beating my league, mini-league strategy, or "
    "differentials, use get_rivals. "
    "When stating any percentage or fraction, name the set it was computed "
    "over (e.g. '8 of the 10 sampled rivals'). Never extrapolate a sample to "
    "the full league. "
    "If data is marked partial, label it as mid-gameweek and provisional; do "
    "not build gap/deficit narratives on provisional standings. "
    "Never open with Yes/No if the recommendation contradicts it — lead with "
    "the recommendation itself. If the user sets an explicit length ('one line'), "
    "comply exactly: one sentence, no headers, no tables, no bold. "
    "If two tool results carry different _meta.gw, reconcile explicitly or use "
    "only the newer one — never mix baselines silently. "
    f"When comparing player projections, if |delta| < {MODEL_ERROR_MAE}, describe "
    "the choice as within model noise / a toss-up with a slight lean — never as "
    "'wrong' or urgent. Reserve strong directives for deltas clearly exceeding "
    "model error."
)


def load_dotenv(path=Path(".env")):
    """Load KEY=VALUE from .env without overriding existing environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def _truncate(obj, limit=500):
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _trace_path():
    traces = Path("traces")
    traces.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return traces / f"trace_{stamp}.jsonl"


def _log(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _text_from(content):
    parts = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


def _cached_system():
    return [
        {
            "type": "text",
            "text": SYSTEM_TEXT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _cached_tools():
    tools = copy.deepcopy(TOOLS)
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _usage_fields(usage):
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _add_usage(totals, usage):
    fields = _usage_fields(usage)
    for key, val in fields.items():
        totals[key] += val
    return fields


def _run_cost(totals):
    cost = (
        totals["input_tokens"] * PRICING["input"]
        + totals["output_tokens"] * PRICING["output"]
        + totals["cache_creation_input_tokens"] * PRICING["cache_write"]
        + totals["cache_read_input_tokens"] * PRICING["cache_read"]
    ) / 1_000_000
    input_total = (
        totals["input_tokens"]
        + totals["cache_creation_input_tokens"]
        + totals["cache_read_input_tokens"]
    )
    cached_pct = (
        100.0 * totals["cache_read_input_tokens"] / input_total if input_total else 0.0
    )
    return cost, cached_pct


def run(question):
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY missing. Put it in .env or the environment.", file=sys.stderr)
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("Install the client: pip install anthropic", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": question}]
    trace = _trace_path()
    n_turns = 0
    n_tools = 0
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    final_text = ""
    hit_limit = False
    system = _cached_system()
    tools = _cached_tools()

    for _ in range(MAX_TURNS):
        n_turns += 1
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            system=system,
            tools=tools,
            messages=messages,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage_fields = _add_usage(totals, response.usage)

        text = _text_from(response.content)
        tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        _log(
            trace,
            {
                "role": "assistant",
                "tool": None,
                "args": None,
                "result": _truncate(text or f"({len(tool_blocks)} tool_use)"),
                "latency_ms": latency_ms,
                **usage_fields,
            },
        )

        if tool_blocks:
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in tool_blocks:
                n_tools += 1
                t1 = time.perf_counter()
                payload = dispatch(block.name, dict(block.input or {}))
                tool_ms = int((time.perf_counter() - t1) * 1000)
                _log(
                    trace,
                    {
                        "role": "tool",
                        "tool": block.name,
                        "args": dict(block.input or {}),
                        "result": _truncate(payload),
                        "latency_ms": tool_ms,
                        "input_tokens": None,
                        "output_tokens": None,
                        "cache_creation_input_tokens": None,
                        "cache_read_input_tokens": None,
                    },
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        final_text = text
        break
    else:
        hit_limit = True

    if hit_limit:
        print("Turn limit (8) was hit before the model finished.")
    if final_text:
        print(final_text)
    cost, cached_pct = _run_cost(totals)
    token_total = (
        totals["input_tokens"]
        + totals["output_tokens"]
        + totals["cache_creation_input_tokens"]
        + totals["cache_read_input_tokens"]
    )
    print(
        f"turns={n_turns} tools={n_tools} tokens={token_total} "
        f"(in={totals['input_tokens']} out={totals['output_tokens']} "
        f"cache_write={totals['cache_creation_input_tokens']} "
        f"cache_read={totals['cache_read_input_tokens']}) "
        f"cost=${cost:.3f} (cached {cached_pct:.0f}%)"
    )
    print(f"trace={trace}")


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python3 agent.py "question here"', file=sys.stderr)
        sys.exit(2)
    run(question)


if __name__ == "__main__":
    main()

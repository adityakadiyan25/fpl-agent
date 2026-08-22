"""FPL assistant: Claude + tools over the GW1 snapshot."""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fpl_tools import TOOLS, dispatch

MODEL = "claude-sonnet-4-6"
MAX_TURNS = 8
SYSTEM = (
    "You are an FPL assistant managing team 4796993. Use tools to answer. "
    "Every numeric claim must come from a tool result — if you didn't call a "
    "tool for it, say so. State dates, times and weekdays only as returned by "
    "tools — never compute calendar facts yourself. Recommendations end with a "
    "one-line WHY citing numbers."
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
    total_in = 0
    total_out = 0
    final_text = ""
    hit_limit = False

    for _ in range(MAX_TURNS):
        n_turns += 1
        t0 = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = response.usage
        inp = getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        total_in += inp
        total_out += out

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
                "input_tokens": inp,
                "output_tokens": out,
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
    print(f"turns={n_turns} tools={n_tools} tokens={total_in + total_out} (in={total_in} out={total_out})")
    print(f"trace={trace}")


def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python3 agent.py "question here"', file=sys.stderr)
        sys.exit(2)
    run(question)


if __name__ == "__main__":
    main()

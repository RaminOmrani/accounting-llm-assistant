"""Execution-based evaluation of the assistant.

For each question in questions.jsonl the reference plan (a tool call or a
reference SQL query) is executed, and the assistant's answer counts as
correct when it returns the same result set (rows compared as multisets
after rounding; column names and row order are ignored, except that a
LIMIT 1 answer must pick the same row). "refuse" items are correct when the
assistant does not run any query.

Usage:
  python eval/run_eval.py                    # rule-based baseline, offline
  LLM_API_KEY=... LLM_MODEL=... [LLM_BASE_URL=...] python eval/run_eval.py --llm
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acct_assistant import db  # noqa: E402
from acct_assistant.assistant import Assistant  # noqa: E402
from acct_assistant.llm import OpenAICompatible, RuleBased  # noqa: E402
from acct_assistant.tools import run_tool  # noqa: E402

TODAY = "2026-09-15"


def _canon(rows, nd=2):
    out = []
    for r in rows:
        out.append(tuple(round(v, nd) if isinstance(v, float) else v for v in r))
    return Counter(out)


def reference(con, ref):
    if "tool" in ref:
        return run_tool(con, ref["tool"], ref["args"])["rows"]
    if "sql" in ref:
        return [list(r) for r in con.execute(ref["sql"]).fetchall()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use an OpenAI-compatible model from env")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    con = db.build(":memory:")
    llm = OpenAICompatible() if args.llm else RuleBased()
    bot = Assistant(con, llm, today=TODAY)
    items = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8")]

    per_kind = defaultdict(lambda: [0, 0])
    rows_out, cost, latency, route_counts = [], 0.0, [], Counter()
    for it in items:
        ans = bot.ask(it["question"])
        ref = reference(con, it["ref"])
        if it["kind"] == "refuse":
            ok = ans.route in ("none",) or (ans.route == "error" and not ans.rows)
        else:
            ok = ans.route in ("tool", "sql") and _canon(ans.rows) == _canon(ref)
        per_kind[it["kind"]][0] += ok
        per_kind[it["kind"]][1] += 1
        route_counts[ans.route] += 1
        cost += ans.cost_usd
        latency.append(ans.latency_s)
        rows_out.append({"id": it["id"], "kind": it["kind"], "correct": ok, "route": ans.route,
                         "plan": ans.plan, "error": ans.error})

    total = sum(c for c, _ in per_kind.values())
    n = sum(t for _, t in per_kind.values())
    print(f"model: {llm.name}")
    for k, (c, t) in sorted(per_kind.items()):
        print(f"  {k:7s} {c:2d}/{t:2d}  ({100 * c / t:.0f}%)")
    print(f"  overall {total}/{n} ({100 * total / n:.0f}%)   routes: {dict(route_counts)}")
    print(f"  cost ${cost:.4f}   median latency {sorted(latency)[len(latency) // 2]:.2f}s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{llm.name.replace('/', '_')}.json").write_text(json.dumps(rows_out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

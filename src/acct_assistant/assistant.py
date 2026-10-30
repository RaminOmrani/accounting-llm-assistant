"""Question answering over the accounting views: tools first, guarded SQL second.

Flow for one question
  1. normalise the text (Persian/Arabic digits -> ASCII);
  2. ask the model for a JSON plan: either {"tool", "args"} or {"sql"};
  3. tool plans run a reviewed SQL template; SQL plans pass `sql_guard.check`
     and run on a read-only connection;
  4. if the guarded SQL fails, the error is sent back once for a repair;
  5. the result is returned with the plan, the executed SQL, timings and
     token cost, and appended to a JSONL log for later evaluation.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from . import sql_guard
from .db import schema_prompt
from .llm import LLM, Usage
from .tools import TOOLS, run_tool

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

SYSTEM = """You answer questions about a company's sales and accounts.
Reply with ONE JSON object and nothing else.
Prefer a tool. Use {{"tool": <name>, "args": {{...}}}} when a tool fits.
Only if no tool fits, write a single SQLite SELECT over these views: {{"sql": "..."}}.
If the question cannot be answered from the data, reply {{"answer": "<short reason>"}}.
Today is {today}.

TOOLS
{tools}

VIEWS (the only tables you may query)
{views}"""


def normalise(text: str) -> str:
    return text.translate(_DIGITS).strip()


@dataclass
class Answer:
    question: str
    route: str                      # tool | sql | none | error
    plan: dict
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    executed_sql: str = ""
    error: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


class Assistant:
    def __init__(self, con: sqlite3.Connection, llm: LLM, log_path: str | Path | None = None, today: str | None = None):
        self.con, self.llm = con, llm
        self.log_path = Path(log_path) if log_path else None
        tools = "\n".join(json.dumps(t.spec()) for t in TOOLS.values())
        self.system = SYSTEM.format(today=today or date.today().isoformat(), tools=tools, views=schema_prompt())

    def _ask_model(self, messages: list[dict], usage: Usage) -> dict:
        reply = self.llm.chat(messages, json_mode=True)
        usage.prompt_tokens += reply.usage.prompt_tokens
        usage.completion_tokens += reply.usage.completion_tokens
        usage.model = reply.usage.model
        return reply.json()

    def ask(self, question: str) -> Answer:
        t0 = time.time()
        q = normalise(question)
        usage = Usage()
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": f"QUESTION: {q}"}]
        try:
            plan = self._ask_model(messages, usage)
        except Exception as e:  # model or JSON failure
            return self._finish(Answer(q, "error", {}, error=f"planning failed: {e}"), usage, t0)

        ans = Answer(q, "none", plan)
        if "tool" in plan and plan["tool"] in TOOLS:
            ans.route = "tool"
            try:
                res = run_tool(self.con, plan["tool"], plan.get("args", {}))
                ans.columns, ans.rows, ans.executed_sql = res["columns"], res["rows"], res["sql"]
            except Exception as e:
                ans.route, ans.error = "error", f"tool failed: {e}"
        elif "sql" in plan:
            ans.route = "sql"
            for attempt in range(2):
                try:
                    checked = sql_guard.check(plan["sql"])
                    cur = self.con.execute(checked.sql)
                    ans.columns = [d[0] for d in cur.description]
                    ans.rows = [list(r) for r in cur.fetchall()]
                    ans.executed_sql, ans.error = checked.sql, ""
                    break
                except Exception as e:
                    ans.error = f"{type(e).__name__}: {e}"
                    if attempt == 0:  # one repair round with the error message
                        messages += [{"role": "assistant", "content": json.dumps(plan)},
                                     {"role": "user", "content": f"That failed with: {ans.error}. Return a corrected JSON plan."}]
                        try:
                            plan = self._ask_model(messages, usage)
                            ans.plan = plan
                            if "sql" not in plan:
                                break
                        except Exception:
                            break
            if ans.error:
                ans.route = "error"
        return self._finish(ans, usage, t0)

    def _finish(self, ans: Answer, usage: Usage, t0: float) -> Answer:
        ans.model = usage.model or getattr(self.llm, "name", "")
        ans.prompt_tokens, ans.completion_tokens = usage.prompt_tokens, usage.completion_tokens
        ans.cost_usd = round(usage.cost_usd(), 6)
        ans.latency_s = round(time.time() - t0, 3)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({k: v for k, v in asdict(ans).items() if k != "rows"} | {"n_rows": len(ans.rows)},
                                   ensure_ascii=False, default=str) + "\n")
        return ans

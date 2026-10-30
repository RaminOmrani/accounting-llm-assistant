"""Language-model clients behind one small interface.

`OpenAICompatible` talks to any provider with an OpenAI-style
`/chat/completions` endpoint (OpenAI, Azure, local servers, or a gateway that
routes to several model families). Model names and prices come from the
environment, so each task can use the model that suits it.

`RuleBased` needs no network. It maps questions to tools with keyword rules,
is used in the tests, and is the baseline in the evaluation.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    model: str = ""

    def cost_usd(self) -> float:
        """Price per million tokens from env, e.g. PRICE_GPT_4O_MINI_IN=0.15."""
        key = re.sub(r"[^A-Z0-9]", "_", self.model.upper())
        pin = float(os.getenv(f"PRICE_{key}_IN", "0"))
        pout = float(os.getenv(f"PRICE_{key}_OUT", "0"))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1e6


@dataclass
class Reply:
    text: str
    usage: Usage = field(default_factory=Usage)

    def json(self) -> Any:
        m = re.search(r"\{.*\}", self.text, re.S)
        if not m:
            raise ValueError(f"no JSON object in model reply: {self.text[:200]!r}")
        return json.loads(m.group(0))


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], json_mode: bool = False) -> Reply: ...


class OpenAICompatible:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, timeout: float = 60):
        import httpx

        self.name = model or os.environ["LLM_MODEL"]
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ["LLM_API_KEY"]
        self._http = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], json_mode: bool = False) -> Reply:
        body: dict[str, Any] = {"model": self.name, "messages": messages, "temperature": 0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        r = self._http.post(f"{self.base_url}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0, self.name)
        return Reply(data["choices"][0]["message"]["content"], usage)


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


class RuleBased:
    """Keyword planner used as an offline baseline. It answers with the same
    JSON plan format that the language model is asked to produce."""

    name = "rule-based"

    def chat(self, messages: list[dict], json_mode: bool = False) -> Reply:
        q = messages[-1]["content"].split("QUESTION:")[-1].strip().lower()
        return Reply(json.dumps(self.plan(q)), Usage(model=self.name))

    @staticmethod
    def _dates(q: str) -> dict:
        m = re.search(r"in (\d{4})", q)
        if m:
            y = m.group(1)
            return {"start": f"{y}-01-01", "end": f"{y}-12-31"}
        m = re.search(r"in (" + "|".join(_MONTHS) + r") (\d{4})", q)
        if m:
            mo, y = _MONTHS[m.group(1)], int(m.group(2))
            last = 31 if mo in (1, 3, 5, 7, 8, 10, 12) else 30 if mo != 2 else 28
            return {"start": f"{y}-{mo:02d}-01", "end": f"{y}-{mo:02d}-{last}"}
        return {}

    def plan(self, q: str) -> dict:
        n = re.search(r"top (\d+)", q)
        n = int(n.group(1)) if n else 5
        cust = re.search(r"customer (\d{1,3})", q)
        d = self._dates(q)
        if "overdue" in q:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", q)
            k = re.search(r"more than (\d+) days", q)
            return {"tool": "overdue_invoices", "args": {"as_of": m.group(1) if m else "2026-09-01", "min_days": int(k.group(1)) if k else 0}}
        if cust and ("balance" in q or "owe" in q):
            return {"tool": "customer_balance", "args": {"customer": f"Customer {int(cust.group(1)):03d}"}}
        if "customer" in q and ("top" in q or "best" in q or "largest" in q):
            return {"tool": "top_customers", "args": {"n": n, **d}}
        if "product" in q and ("top" in q or "best" in q or "selling" in q):
            cat = next((c for c in ["beverages", "dairy", "bakery", "household"] if c in q), None)
            return {"tool": "top_products", "args": {"n": n, **d, **({"category": cat} if cat else {})}}
        if "month" in q:
            y = re.search(r"(\d{4})", q)
            return {"tool": "monthly_sales", "args": {"year": int(y.group(1))} if y else {}}
        if "city" in q or "cities" in q:
            return {"tool": "sales_by_city", "args": d}
        for acc in ["cash", "accounts receivable", "sales revenue", "vat payable"]:
            if acc in q:
                return {"tool": "account_balance", "args": {"account_name": acc}}
        if "sales" in q or "revenue" in q or "how many invoices" in q:
            return {"tool": "sales_total", "args": d}
        return {"answer": "I cannot map this question to a known report."}

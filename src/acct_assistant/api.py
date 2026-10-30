"""HTTP API (FastAPI).

Run:  uvicorn acct_assistant.api:app --app-dir src
Environment: DB_PATH (default data/demo.sqlite, created if missing),
LLM_MODEL / LLM_BASE_URL / LLM_API_KEY for an OpenAI-compatible provider
(without LLM_API_KEY the rule-based planner is used), VISION_MODEL for invoices.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import db, invoices
from .assistant import Assistant
from .llm import OpenAICompatible, RuleBased

DB_PATH = Path(os.getenv("DB_PATH", "data/demo.sqlite"))
LOG_PATH = Path(os.getenv("LOG_PATH", "data/requests.jsonl"))

if not DB_PATH.exists():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.build(DB_PATH).close()


def _llm(model_env: str = "LLM_MODEL"):
    if os.getenv("LLM_API_KEY"):
        return OpenAICompatible(model=os.getenv(model_env) or os.getenv("LLM_MODEL"))
    return RuleBased()


app = FastAPI(title="Accounting LLM assistant (demo)")
assistant = Assistant(db.readonly(DB_PATH), _llm(), log_path=LOG_PATH)


class AskReq(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True, "model": assistant.llm.name}


@app.post("/ask")
def ask(req: AskReq):
    return asdict(assistant.ask(req.question))


def _rw() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)  # only the invoice flow writes, and only to staging until confirmed


@app.post("/invoices/extract")
async def extract_invoice(file: UploadFile = File(...)):
    if not os.getenv("LLM_API_KEY"):
        raise HTTPException(503, "invoice reading needs a vision model (set LLM_API_KEY and VISION_MODEL)")
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "x.jpg").suffix) as tmp:
        tmp.write(await file.read())
        tmp.flush()
        draft = invoices.extract(_llm("VISION_MODEL"), tmp.name)
    con = _rw()
    staged_id, issues = invoices.stage(draft, con)
    return {"staged_id": staged_id, "draft": draft.model_dump(), "issues": issues}


@app.post("/invoices/{staged_id}/confirm")
def confirm_invoice(staged_id: int, allow_issues: bool = False):
    try:
        return {"invoice_id": invoices.confirm(staged_id, _rw(), allow_issues=allow_issues)}
    except ValueError as e:
        raise HTTPException(409, str(e))

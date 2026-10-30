"""Reading invoices from images, with checks and a human confirmation step.

1. A vision-capable model returns the invoice as JSON.
2. The JSON is parsed into a typed `InvoiceDraft` (pydantic).
3. `validate` checks the arithmetic (line totals, VAT, gross) and the
   customer and product names against the database, and lists every issue.
4. The draft is stored in `staged_invoices`; nothing is booked yet.
5. A person reviews the draft and the issues; only `confirm` writes the
   invoice and its lines into the real tables.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .llm import LLM

VAT_RATE = 0.10
TOL = 0.02  # tolerance for rounding differences, in currency units


class Line(BaseModel):
    product: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    discount: float = Field(default=0, ge=0, le=1)
    line_total: float | None = None


class InvoiceDraft(BaseModel):
    number: str
    customer: str
    issue_date: str
    due_date: str | None = None
    lines: list[Line]
    net_amount: float | None = None
    vat_amount: float | None = None
    gross_amount: float | None = None


PROMPT = ("Read this sales invoice. Return JSON with keys: number, customer, issue_date (YYYY-MM-DD), due_date, "
          "lines (product, quantity, unit_price, discount as a fraction, line_total), net_amount, vat_amount, gross_amount. "
          "Copy numbers exactly as printed; use null for anything you cannot read. Convert Persian digits to ASCII.")


def extract(llm: LLM, image_path: str | Path) -> InvoiceDraft:
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}]
    return InvoiceDraft.model_validate(llm.chat(messages, json_mode=True).json())


def validate(draft: InvoiceDraft, con: sqlite3.Connection) -> list[str]:
    issues = []
    net = 0.0
    for i, ln in enumerate(draft.lines, 1):
        expected = ln.quantity * ln.unit_price * (1 - ln.discount)
        net += expected
        if ln.line_total is not None and abs(ln.line_total - expected) > TOL:
            issues.append(f"line {i}: total {ln.line_total} != quantity x price x (1 - discount) = {expected:.2f}")
        if not con.execute("SELECT 1 FROM products WHERE lower(name) = lower(?)", (ln.product,)).fetchone():
            issues.append(f"line {i}: unknown product '{ln.product}'")
    if draft.net_amount is not None and abs(draft.net_amount - net) > TOL:
        issues.append(f"net amount {draft.net_amount} != sum of lines {net:.2f}")
    if draft.vat_amount is not None and abs(draft.vat_amount - net * VAT_RATE) > TOL:
        issues.append(f"VAT {draft.vat_amount} != {VAT_RATE:.0%} of net ({net * VAT_RATE:.2f})")
    if draft.gross_amount is not None and abs(draft.gross_amount - net * (1 + VAT_RATE)) > TOL:
        issues.append(f"gross {draft.gross_amount} != net + VAT ({net * (1 + VAT_RATE):.2f})")
    if not con.execute("SELECT 1 FROM customers WHERE lower(name) = lower(?)", (draft.customer,)).fetchone():
        issues.append(f"unknown customer '{draft.customer}'")
    if con.execute("SELECT 1 FROM invoices WHERE number = ?", (draft.number,)).fetchone():
        issues.append(f"invoice number {draft.number} already exists")
    try:
        datetime.strptime(draft.issue_date, "%Y-%m-%d")
    except ValueError:
        issues.append(f"issue date '{draft.issue_date}' is not YYYY-MM-DD")
    return issues


def stage(draft: InvoiceDraft, con: sqlite3.Connection) -> tuple[int, list[str]]:
    issues = validate(draft, con)
    cur = con.execute("INSERT INTO staged_invoices (payload, issues, status, created_at) VALUES (?,?,?,?)",
                      (draft.model_dump_json(), json.dumps(issues), "pending", datetime.now().isoformat(timespec="seconds")))
    con.commit()
    return cur.lastrowid, issues


def confirm(staged_id: int, con: sqlite3.Connection, allow_issues: bool = False) -> int:
    """Book a reviewed draft. Refuses drafts that still have issues unless the
    reviewer explicitly accepts them. All writes happen in one transaction."""
    row = con.execute("SELECT payload, status FROM staged_invoices WHERE id = ?", (staged_id,)).fetchone()
    if row is None or row[1] != "pending":
        raise ValueError("no pending draft with this id")
    draft = InvoiceDraft.model_validate_json(row[0])
    issues = validate(draft, con)
    if issues and not allow_issues:
        raise ValueError(f"draft has unresolved issues: {issues}")
    with con:
        cust = con.execute("SELECT id FROM customers WHERE lower(name) = lower(?)", (draft.customer,)).fetchone()
        if cust is None:
            raise ValueError("customer must exist before booking")
        inv = con.execute("INSERT INTO invoices (number, customer_id, issue_date, due_date, status) VALUES (?,?,?,?, 'open')",
                          (draft.number, cust[0], draft.issue_date, draft.due_date)).lastrowid
        for ln in draft.lines:
            prod = con.execute("SELECT id FROM products WHERE lower(name) = lower(?)", (ln.product,)).fetchone()
            if prod is None:
                raise ValueError(f"product '{ln.product}' must exist before booking")
            con.execute("INSERT INTO invoice_lines (invoice_id, product_id, quantity, unit_price, discount) VALUES (?,?,?,?,?)",
                        (inv, prod[0], ln.quantity, ln.unit_price, ln.discount))
        con.execute("UPDATE staged_invoices SET status = 'booked' WHERE id = ?", (staged_id,))
    return inv

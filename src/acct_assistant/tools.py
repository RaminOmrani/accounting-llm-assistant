"""Parametric tools: the first and preferred way to answer a question.

Each tool is a fixed, reviewed SQL template with typed parameters. The model
only chooses a tool and fills its arguments; it never writes this SQL. Free
text-to-SQL is used only when no tool fits (see `assistant.py`).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]          # name -> JSON type with short description
    build: Callable[..., tuple[str, tuple]]

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.params}


def _period(start: str | None, end: str | None) -> tuple[str, tuple]:
    cond, args = [], []
    if start:
        cond.append("issue_date >= ?")
        args.append(start)
    if end:
        cond.append("issue_date <= ?")
        args.append(end)
    return (" AND " + " AND ".join(cond) if cond else ""), tuple(args)


def _sales_total(start=None, end=None):
    w, a = _period(start, end)
    return (f"SELECT COUNT(*) AS invoices, ROUND(SUM(net_amount),2) AS net_sales, ROUND(SUM(gross_amount),2) AS gross_sales "
            f"FROM v_invoice_totals WHERE status != 'cancelled'{w}", a)


def _top_customers(n=5, start=None, end=None):
    w, a = _period(start, end)
    return (f"SELECT customer, ROUND(SUM(net_amount),2) AS net_sales FROM v_invoice_totals "
            f"WHERE status != 'cancelled'{w} GROUP BY customer ORDER BY net_sales DESC LIMIT ?", a + (int(n),))


def _top_products(n=5, start=None, end=None, category=None):
    w, a = _period(start, end)
    if category:
        w += " AND category = ?"
        a += (category,)
    return (f"SELECT product, category, SUM(quantity) AS units, ROUND(SUM(net_amount),2) AS net_sales FROM v_sales_lines "
            f"WHERE invoice_number IN (SELECT invoice_number FROM v_invoice_totals WHERE status != 'cancelled'){w} "
            f"GROUP BY product ORDER BY net_sales DESC LIMIT ?", a + (int(n),))


def _monthly_sales(year=None):
    w, a = ("", ())
    if year:
        w, a = " AND substr(issue_date,1,4) = ?", (str(year),)
    return (f"SELECT substr(issue_date,1,7) AS month, ROUND(SUM(net_amount),2) AS net_sales FROM v_invoice_totals "
            f"WHERE status != 'cancelled'{w} GROUP BY month ORDER BY month", a)


def _customer_balance(customer):
    return ("SELECT customer, invoiced, paid, open_balance FROM v_customer_balance WHERE customer = ?", (customer,))


def _overdue_invoices(as_of, min_days=0):
    return ("SELECT b.invoice_number, b.customer, b.due_date, b.open_amount, "
            "CAST(julianday(?) - julianday(b.due_date) AS INTEGER) AS days_overdue "
            "FROM v_invoice_balance b JOIN v_invoice_totals t ON t.invoice_number = b.invoice_number "
            "WHERE t.status != 'cancelled' AND b.open_amount > 0.01 AND julianday(?) - julianday(b.due_date) > ? "
            "ORDER BY days_overdue DESC LIMIT 200", (as_of, as_of, int(min_days)))


def _sales_by_city(start=None, end=None):
    w, a = _period(start, end)
    return (f"SELECT city, ROUND(SUM(net_amount),2) AS net_sales FROM v_invoice_totals WHERE status != 'cancelled'{w} "
            f"GROUP BY city ORDER BY net_sales DESC", a)


def _account_balance(account_name):
    return ("SELECT account_code, account_name, total_debit, total_credit, balance FROM v_account_balances "
            "WHERE lower(account_name) = lower(?)", (account_name,))


DATE = "string, ISO date YYYY-MM-DD, optional"
TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("sales_total", "Total number of invoices and net/gross sales in a period (cancelled invoices excluded).",
         {"start": DATE, "end": DATE}, _sales_total),
    Tool("top_customers", "Customers ranked by net sales in a period.",
         {"n": "integer, how many, default 5", "start": DATE, "end": DATE}, _top_customers),
    Tool("top_products", "Products ranked by net sales, optionally within one category.",
         {"n": "integer, default 5", "start": DATE, "end": DATE, "category": "string: beverages|dairy|bakery|household, optional"}, _top_products),
    Tool("monthly_sales", "Net sales per month, optionally for one calendar year.",
         {"year": "integer, optional"}, _monthly_sales),
    Tool("customer_balance", "Invoiced, paid and open balance of one customer.",
         {"customer": "string, exact customer name, e.g. 'Customer 007'"}, _customer_balance),
    Tool("overdue_invoices", "Unpaid invoices past their due date as of a given day.",
         {"as_of": "string, ISO date", "min_days": "integer, minimum days overdue, default 0"}, _overdue_invoices),
    Tool("sales_by_city", "Net sales per customer city in a period.",
         {"start": DATE, "end": DATE}, _sales_by_city),
    Tool("account_balance", "Debit, credit and balance of a general-ledger account.",
         {"account_name": "string: Cash|Accounts receivable|Sales revenue|VAT payable"}, _account_balance),
]}


def run_tool(con: sqlite3.Connection, name: str, args: dict[str, Any]) -> dict:
    tool = TOOLS[name]
    allowed = set(tool.params)
    clean = {k: v for k, v in (args or {}).items() if k in allowed and v not in (None, "")}
    sql, params = tool.build(**clean)
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return {"columns": cols, "rows": [list(r) for r in cur.fetchall()], "sql": sql, "params": list(params)}

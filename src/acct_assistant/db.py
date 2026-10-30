"""A small synthetic accounting database (SQLite) with a semantic view layer.

The production assistant ran on a SQL Server accounting database. Here the
same idea is reproduced on generated data: raw tables hold customers,
products, invoices, payments and journal vouchers, and a set of `v_*` views
exposes business concepts (sales, balances, overdue invoices) with clear
column names. The language model only ever sees and queries the views.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, city TEXT, segment TEXT);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT, unit_price REAL);
CREATE TABLE invoices  (id INTEGER PRIMARY KEY, number TEXT UNIQUE, customer_id INTEGER REFERENCES customers(id),
                        issue_date TEXT, due_date TEXT, status TEXT);
CREATE TABLE invoice_lines (id INTEGER PRIMARY KEY, invoice_id INTEGER REFERENCES invoices(id),
                        product_id INTEGER REFERENCES products(id), quantity REAL, unit_price REAL, discount REAL);
CREATE TABLE payments  (id INTEGER PRIMARY KEY, invoice_id INTEGER REFERENCES invoices(id), paid_date TEXT, amount REAL);
CREATE TABLE accounts  (code TEXT PRIMARY KEY, name TEXT, type TEXT);
CREATE TABLE vouchers  (id INTEGER PRIMARY KEY, voucher_date TEXT, description TEXT);
CREATE TABLE voucher_lines (id INTEGER PRIMARY KEY, voucher_id INTEGER REFERENCES vouchers(id),
                        account_code TEXT REFERENCES accounts(code), debit REAL, credit REAL);
-- staging area for invoices read from images; nothing here is final until confirmed
CREATE TABLE staged_invoices (id INTEGER PRIMARY KEY, payload TEXT, issues TEXT, status TEXT, created_at TEXT);
"""

VIEWS = """
CREATE VIEW v_invoice_totals AS
SELECT i.id AS invoice_id, i.number AS invoice_number, c.name AS customer, c.city, c.segment,
       i.issue_date, i.due_date, i.status,
       ROUND(SUM(l.quantity * l.unit_price * (1 - l.discount)), 2) AS net_amount,
       ROUND(SUM(l.quantity * l.unit_price * (1 - l.discount)) * 0.10, 2) AS vat_amount,
       ROUND(SUM(l.quantity * l.unit_price * (1 - l.discount)) * 1.10, 2) AS gross_amount
FROM invoices i JOIN customers c ON c.id = i.customer_id JOIN invoice_lines l ON l.invoice_id = i.id
GROUP BY i.id;

CREATE VIEW v_sales_lines AS
SELECT i.number AS invoice_number, i.issue_date, c.name AS customer, p.name AS product, p.category,
       l.quantity, l.unit_price, l.discount,
       ROUND(l.quantity * l.unit_price * (1 - l.discount), 2) AS net_amount
FROM invoice_lines l JOIN invoices i ON i.id = l.invoice_id
JOIN products p ON p.id = l.product_id JOIN customers c ON c.id = i.customer_id;

CREATE VIEW v_invoice_balance AS
SELECT t.invoice_id, t.invoice_number, t.customer, t.issue_date, t.due_date, t.gross_amount,
       ROUND(COALESCE((SELECT SUM(amount) FROM payments p WHERE p.invoice_id = t.invoice_id), 0), 2) AS paid_amount,
       ROUND(t.gross_amount - COALESCE((SELECT SUM(amount) FROM payments p WHERE p.invoice_id = t.invoice_id), 0), 2) AS open_amount
FROM v_invoice_totals t;

CREATE VIEW v_customer_balance AS
SELECT customer, ROUND(SUM(gross_amount), 2) AS invoiced, ROUND(SUM(paid_amount), 2) AS paid,
       ROUND(SUM(open_amount), 2) AS open_balance
FROM v_invoice_balance GROUP BY customer;

CREATE VIEW v_account_balances AS
SELECT a.code AS account_code, a.name AS account_name, a.type AS account_type,
       ROUND(SUM(vl.debit), 2) AS total_debit, ROUND(SUM(vl.credit), 2) AS total_credit,
       ROUND(SUM(vl.debit) - SUM(vl.credit), 2) AS balance
FROM accounts a LEFT JOIN voucher_lines vl ON vl.account_code = a.code GROUP BY a.code;
"""

# Column descriptions given to the model (the "semantic layer").
VIEW_DOCS = {
    "v_invoice_totals": "one row per invoice: invoice_number, customer, city, segment, issue_date (YYYY-MM-DD), due_date, status (open|paid|cancelled), net_amount, vat_amount (10%), gross_amount",
    "v_sales_lines": "one row per invoice line: invoice_number, issue_date, customer, product, category, quantity, unit_price, discount (0-1), net_amount",
    "v_invoice_balance": "per invoice: invoice_number, customer, issue_date, due_date, gross_amount, paid_amount, open_amount",
    "v_customer_balance": "per customer: invoiced, paid, open_balance",
    "v_account_balances": "general ledger per account: account_code, account_name, account_type, total_debit, total_credit, balance",
}

CITIES = ["Mashhad", "Tehran", "Isfahan", "Shiraz", "Tabriz"]
SEGMENTS = ["retail", "wholesale", "restaurant"]
CATEGORIES = {
    "beverages": ["Mineral water 1.5L", "Orange juice 1L", "Green tea 100g", "Coffee beans 250g"],
    "dairy": ["Milk 1L", "Yogurt 900g", "Cheese 400g"],
    "bakery": ["Flatbread pack", "Biscuits 300g", "Cake 500g"],
    "household": ["Dish soap 1L", "Paper towels 6-pack", "Detergent 3kg"],
}


def build(path: str | Path = ":memory:", seed: int = 7, n_customers: int = 40, n_invoices: int = 600) -> sqlite3.Connection:
    rnd = random.Random(seed)
    con = sqlite3.connect(str(path), check_same_thread=False)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)",
                    [(i, f"Customer {i:03d}", rnd.choice(CITIES), rnd.choice(SEGMENTS)) for i in range(1, n_customers + 1)])
    prods, pid = [], 1
    for cat, names in CATEGORIES.items():
        for n in names:
            prods.append((pid, n, cat, round(rnd.uniform(20, 400), 2)))
            pid += 1
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", prods)
    con.executemany("INSERT INTO accounts VALUES (?,?,?)", [
        ("1100", "Cash", "asset"), ("1200", "Accounts receivable", "asset"),
        ("4000", "Sales revenue", "income"), ("2100", "VAT payable", "liability")])

    start = date(2025, 3, 21)
    for inv in range(1, n_invoices + 1):
        issue = start + timedelta(days=rnd.randint(0, 540))
        due = issue + timedelta(days=rnd.choice([15, 30, 45]))
        cust = rnd.randint(1, n_customers)
        con.execute("INSERT INTO invoices VALUES (?,?,?,?,?,?)",
                    (inv, f"INV-{inv:05d}", cust, issue.isoformat(), due.isoformat(), "open"))
        net = 0.0
        for _ in range(rnd.randint(1, 5)):
            p = rnd.choice(prods)
            qty = rnd.randint(1, 40)
            price = round(p[3] * rnd.uniform(0.95, 1.05), 2)
            disc = rnd.choice([0, 0, 0, 0.05, 0.1])
            con.execute("INSERT INTO invoice_lines (invoice_id, product_id, quantity, unit_price, discount) VALUES (?,?,?,?,?)",
                        (inv, p[0], qty, price, disc))
            net += qty * price * (1 - disc)
        net = round(net, 2)
        gross = round(net * 1.10, 2)
        r = rnd.random()
        if r < 0.65:   # paid in full
            con.execute("INSERT INTO payments (invoice_id, paid_date, amount) VALUES (?,?,?)",
                        (inv, (issue + timedelta(days=rnd.randint(1, 50))).isoformat(), gross))
            con.execute("UPDATE invoices SET status='paid' WHERE id=?", (inv,))
        elif r < 0.85:  # partly paid
            con.execute("INSERT INTO payments (invoice_id, paid_date, amount) VALUES (?,?,?)",
                        (inv, (issue + timedelta(days=rnd.randint(1, 50))).isoformat(), round(gross * rnd.uniform(0.2, 0.8), 2)))
        elif r < 0.88:
            con.execute("UPDATE invoices SET status='cancelled' WHERE id=?", (inv,))
        # journal entry for the sale
        cur = con.execute("INSERT INTO vouchers (voucher_date, description) VALUES (?,?)", (issue.isoformat(), f"Sale INV-{inv:05d}"))
        vid = cur.lastrowid
        con.executemany("INSERT INTO voucher_lines (voucher_id, account_code, debit, credit) VALUES (?,?,?,?)", [
            (vid, "1200", gross, 0), (vid, "4000", 0, net), (vid, "2100", 0, round(gross - net, 2))])
    con.executescript(VIEWS)
    con.commit()
    return con


def readonly(path: str | Path) -> sqlite3.Connection:
    """Open an existing database file read-only (the SQL path never gets write access)."""
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, check_same_thread=False)


def schema_prompt() -> str:
    return "\n".join(f"- {v}: {d}" for v, d in VIEW_DOCS.items())

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acct_assistant import db, invoices, sql_guard  # noqa: E402
from acct_assistant.assistant import Assistant, normalise  # noqa: E402
from acct_assistant.llm import Reply, RuleBased, Usage  # noqa: E402
from acct_assistant.tools import TOOLS, run_tool  # noqa: E402


@pytest.fixture(scope="module")
def con():
    return db.build(":memory:", n_invoices=200)


# ---------------------------------------------------------------- SQL guard
@pytest.mark.parametrize("sql", [
    "DELETE FROM v_invoice_totals",
    "SELECT * FROM v_invoice_totals; DROP TABLE invoices",
    "SELECT * FROM invoices",                     # raw table, not a view
    "UPDATE customers SET name = 'x'",
    "PRAGMA table_info(invoices)",
    "SELECT * FROM v_invoice_totals WHERE customer IN (SELECT name FROM customers)",
])
def test_guard_rejects(sql):
    with pytest.raises(sql_guard.UnsafeSQL):
        sql_guard.check(sql)


def test_guard_accepts_select_and_caps_rows():
    out = sql_guard.check("SELECT customer, open_balance FROM v_customer_balance ORDER BY open_balance DESC")
    assert "LIMIT 200" in out.sql.upper()
    out = sql_guard.check("WITH t AS (SELECT * FROM v_invoice_totals) SELECT COUNT(*) FROM t LIMIT 5")
    assert out.tables == ["v_invoice_totals"] and "LIMIT 5" in out.sql.upper()


def test_readonly_connection_cannot_write(tmp_path):
    path = tmp_path / "x.sqlite"
    db.build(path, n_invoices=20).close()
    ro = db.readonly(path)
    with pytest.raises(Exception):
        ro.execute("DELETE FROM invoices")


# ---------------------------------------------------------------- tools and assistant
def test_every_tool_runs(con):
    samples = {"customer_balance": {"customer": "Customer 001"}, "overdue_invoices": {"as_of": "2026-06-30"},
               "account_balance": {"account_name": "Cash"}}
    for name in TOOLS:
        res = run_tool(con, name, samples.get(name, {}))
        assert res["columns"]


def test_ledger_balances(con):
    tot = con.execute("SELECT SUM(total_debit) - SUM(total_credit) FROM v_account_balances").fetchone()[0]
    assert abs(tot) < 1e-6  # double entry: debits equal credits


def test_persian_digits_are_normalised():
    assert normalise("فروش ۱۴۰۴") == "فروش 1404"


def test_rule_based_route(con):
    a = Assistant(con, RuleBased(), today="2026-09-15").ask("Top 3 customers overall")
    assert a.route == "tool" and len(a.rows) == 3


class ScriptedLLM:
    """Returns pre-written replies, to test the SQL path and the repair loop."""
    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, json_mode=False):
        return Reply(json.dumps(self.replies.pop(0)), Usage(10, 5, 0.0, self.name))


def test_sql_path_repairs_once(con):
    llm = ScriptedLLM([{"sql": "SELECT * FROM invoices"},  # not allowed -> error fed back
                       {"sql": "SELECT COUNT(*) FROM v_invoice_totals WHERE status = 'cancelled'"}])
    a = Assistant(con, llm).ask("how many cancelled?")
    assert a.route == "sql" and a.rows and not a.error and a.prompt_tokens == 20


def test_destructive_sql_is_never_run(con):
    before = con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0]
    llm = ScriptedLLM([{"sql": "DELETE FROM v_invoice_totals"}, {"sql": "DROP TABLE invoices"}])
    a = Assistant(con, llm).ask("delete everything")
    assert a.route == "error"
    assert con.execute("SELECT COUNT(*) FROM invoices").fetchone()[0] == before


# ---------------------------------------------------------------- invoices
def _draft(**kw):
    base = dict(number="INV-99001", customer="Customer 003", issue_date="2026-09-01", due_date="2026-10-01",
                lines=[{"product": "Milk 1L", "quantity": 10, "unit_price": 50, "discount": 0.1, "line_total": 450}],
                net_amount=450, vat_amount=45, gross_amount=495)
    base.update(kw)
    return invoices.InvoiceDraft.model_validate(base)


def test_invoice_checks_find_errors(con):
    issues = invoices.validate(_draft(vat_amount=40, customer="Nobody"), con)
    assert any("VAT" in i for i in issues) and any("unknown customer" in i for i in issues)


def test_invoice_stage_then_confirm(con):
    sid, issues = invoices.stage(_draft(), con)
    assert issues == []
    inv_id = invoices.confirm(sid, con)
    row = con.execute("SELECT gross_amount FROM v_invoice_totals WHERE invoice_id = ?", (inv_id,)).fetchone()
    assert abs(row[0] - 495) < 0.01


def test_confirm_refuses_unresolved_issues(con):
    sid, issues = invoices.stage(_draft(number="INV-99002", gross_amount=999), con)
    assert issues
    with pytest.raises(ValueError):
        invoices.confirm(sid, con)

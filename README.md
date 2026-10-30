# Accounting LLM assistant

A question-answering and document-reading assistant for accounting data, built around one idea: **a language model should choose and fill reviewed operations, and only write free SQL when it must, under strict checks.**

This repository is a public re-implementation of the design of an assistant I built at Millionaire Software for their accounting product (natural-language questions over the accounting database, and reading invoices and vouchers from images). The production system is proprietary and runs on SQL Server. Everything here, including the code, the database and the evaluation set, was written for this repository. The database is synthetic.

## What it does

**Questions over the books** (`assistant.py`)
1. Persian and Arabic digits are normalised.
2. The model returns a JSON plan: either a call to one of eight **parametric tools** (`tools.py`: sales in a period, top customers or products, monthly sales, customer balance, overdue invoices, sales by city, ledger balance), or a single **SQL query** when no tool fits.
3. Tool calls run fixed, reviewed SQL templates with typed parameters.
4. Model-written SQL is checked on its parsed syntax tree (`sql_guard.py`): one SELECT only, only the semantic views, no data-modifying nodes, row cap added. It then runs on a **read-only** connection. If it fails, the error is returned to the model once for a repair.
5. Every request is logged as JSONL (plan, executed SQL, tokens, cost, latency) so real usage becomes an evaluation set.

**Invoices from images** (`invoices.py`)
1. A vision model returns the invoice as JSON, parsed into a typed `InvoiceDraft` (pydantic).
2. Checks: line totals, net, VAT and gross arithmetic; customer and product exist; invoice number not already booked; date format.
3. The draft and its issues go to a staging table. **Nothing is booked until a person confirms it**, and confirmation refuses drafts with unresolved issues unless the reviewer explicitly accepts them. Booking is one transaction.

**Semantic layer** (`db.py`): the model never sees raw tables, only views with business meaning (`v_invoice_totals`, `v_sales_lines`, `v_invoice_balance`, `v_customer_balance`, `v_account_balances`) and a one-line description of each.

**Models** (`llm.py`): any OpenAI-compatible endpoint (a gateway can route each task to a different model family). Prices per model come from environment variables, so cost is tracked per request. `RuleBased` is an offline keyword planner, used in tests and as the evaluation baseline.

## Evaluation

`eval/questions.jsonl` holds 32 questions: 22 answerable by a tool, 8 that need free SQL, and 2 that must be refused (a deletion request and data that does not exist). A question counts as correct when the assistant returns the same result set as the reference tool call or reference query (rows compared as multisets after rounding). A refusal counts as correct when no query runs.

```
python eval/run_eval.py                       # offline baseline
LLM_API_KEY=... LLM_MODEL=... LLM_BASE_URL=... python eval/run_eval.py --llm
```

Offline baseline (`RuleBased`):

| question type | correct |
|---|---|
| tool | 21/22 |
| free SQL | 0/8 |
| refusal | 2/2 |
| overall | 23/32 |

The baseline cannot write SQL at all, which is the gap a language model is meant to close. The same script reports accuracy per type, cost and median latency for any model, so models can be compared on the same questions.

## Run

```bash
pip install -r requirements.txt
pytest -q                                          # 17 tests
uvicorn acct_assistant.api:app --app-dir src       # creates data/demo.sqlite on first start
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Top 5 customers in 2025"}'
```

Without `LLM_API_KEY` the API uses the offline planner. Invoice reading needs a vision model (`VISION_MODEL`).

## Layout

```
src/acct_assistant/  db.py  sql_guard.py  tools.py  llm.py  assistant.py  invoices.py  api.py
eval/                questions.jsonl  run_eval.py
tests/               test_assistant.py
```

## Licence

MIT. See `LICENSE`.

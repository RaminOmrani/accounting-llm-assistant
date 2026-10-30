"""Validation of model-written SQL before it touches the database.

Rules (checked on the parsed syntax tree, not with regular expressions):
  * exactly one statement, and it must be a SELECT (CTEs allowed);
  * only the semantic views in `ALLOWED` may be referenced;
  * no data-modifying nodes anywhere in the tree;
  * a LIMIT is added (or tightened) so results stay small.
The query is then run on a read-only connection, so even a validator bug
cannot change data.
"""
from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .db import VIEW_DOCS

ALLOWED = set(VIEW_DOCS)
MAX_ROWS = 200
_FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Pragma)


class UnsafeSQL(ValueError):
    pass


@dataclass
class CheckedSQL:
    sql: str
    tables: list[str]


def check(sql: str, dialect: str = "sqlite", max_rows: int = MAX_ROWS) -> CheckedSQL:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQL("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise UnsafeSQL(f"only SELECT is allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQL(f"forbidden operation: {type(node).__name__}")
    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    tables = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - cte_names)
    bad = [t for t in tables if t not in ALLOWED]
    if bad:
        raise UnsafeSQL(f"tables not allowed: {bad} (use only {sorted(ALLOWED)})")
    limit = tree.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            current = None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)
    return CheckedSQL(tree.sql(dialect=dialect), tables)

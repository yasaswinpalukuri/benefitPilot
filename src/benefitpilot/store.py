"""Read-only access to the synthetic member database.

The connection is opened with SQLite's `mode=ro`, so checking a claim can
never change a member's usage (docs/DECISIONS.md, ADR-011).
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

from benefitpilot.rules.models import CategoryUsage, MemberSnapshot


class MemberNotFound(LookupError):
    pass


class MemberStore:
    def __init__(self, db_path: Path):
        if not db_path.exists():
            raise FileNotFoundError(f"{db_path} not found. Run: python scripts/generate_data.py")
        self.db_path = db_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    def plan_id(self, member_id: str) -> str:
        with self._connect() as con:
            row = con.execute(
                "SELECT plan_id FROM members WHERE member_id = ?", (member_id,)
            ).fetchone()
        if row is None:
            raise MemberNotFound(member_id)
        return row["plan_id"]

    def snapshot(self, member_id: str, benefit_year: int) -> MemberSnapshot:
        """Member facts plus usage for one benefit year.

        Annual amounts come from that year's row. The last service date and
        lifetime totals span all years, because frequency windows and
        lifetime maximums cross year boundaries.
        """
        with self._connect() as con:
            m = con.execute("SELECT * FROM members WHERE member_id = ?", (member_id,)).fetchone()
            if m is None:
                raise MemberNotFound(member_id)
            rows = con.execute(
                """
                SELECT category,
                       SUM(CASE WHEN benefit_year = :year THEN amount_paid ELSE 0 END) AS paid,
                       MAX(last_service_date) AS last_service,
                       MAX(lifetime_paid) AS lifetime
                FROM usage
                WHERE member_id = :mid
                GROUP BY category
                """,
                {"mid": member_id, "year": benefit_year},
            ).fetchall()
            ded = con.execute(
                "SELECT amount_applied FROM deductibles WHERE member_id = ? AND benefit_year = ?",
                (member_id, benefit_year),
            ).fetchone()

        usage = {
            r["category"]: CategoryUsage(
                amount_paid=_dec(r["paid"]),
                last_service_date=(
                    dt.date.fromisoformat(r["last_service"]) if r["last_service"] else None
                ),
                lifetime_paid=_dec(r["lifetime"]),
            )
            for r in rows
        }
        return MemberSnapshot(
            member_id=m["member_id"],
            plan_id=m["plan_id"],
            dob=dt.date.fromisoformat(m["dob"]),
            relationship=m["relationship"],
            is_student=bool(m["is_student"]),
            coverage_start=dt.date.fromisoformat(m["coverage_start"]),
            benefit_year=benefit_year,
            deductible_applied=_dec(ded["amount_applied"]) if ded else Decimal("0"),
            usage=usage,
        )


def _dec(value: float | int | None) -> Decimal:
    # Via str() so 420.1 becomes Decimal("420.1"), not its binary approximation.
    return Decimal(str(value)) if value is not None else Decimal("0")

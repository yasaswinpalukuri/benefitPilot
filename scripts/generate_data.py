"""Generate reproducible synthetic members, year-to-date usage and claims.

    python scripts/generate_data.py            # writes data/benefitpilot.db
    python scripts/generate_data.py --out x.db --seed 7

Everything here is fictional. Names are random combinations from small lists.

Claim labels are assigned *by construction*: each violating claim is built so
that it breaks exactly one rule, and each clean claim is built to sit well
inside every limit. The rules engine is never used to label data, so the
Phase 7 evaluation is not circular (docs/DECISIONS.md, ADR-004).
"""

from __future__ import annotations

import argparse
import calendar
import math
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benefitpilot.rules.plan_schema import Category, Plan, load_plans  # noqa: E402

GENERATOR_VERSION = "1"
DEFAULT_SEED = 42
BENEFIT_YEAR = 2026
# Fixed dates (not "today") so every run produces the same database.
AS_OF = date(2026, 9, 30)  # usage snapshot date
CLAIM_START = date(2026, 10, 1)
CLAIM_END = date(2026, 10, 31)

N_PRIMARY = 30
N_CLEAN = 58
N_PER_VIOLATION = 6
# Primaries who joined this year, so waiting periods still apply to them.
NEW_HIRE_INDEXES = {3, 4, 9, 10, 15, 16}

VIOLATIONS = (
    "annual_max_exceeded",
    "per_visit_max_exceeded",
    "frequency_limit",
    "waiting_period",
    "excluded_service",
    "dependent_ineligible_age",
    "preauth_missing",
)

FIRST_NAMES = [
    "Avery",
    "Jordan",
    "Priya",
    "Mateo",
    "Noor",
    "Liam",
    "Sofia",
    "Kenji",
    "Amara",
    "Ethan",
    "Leila",
    "Owen",
    "Mei",
    "Diego",
    "Hana",
    "Lucas",
    "Zara",
    "Arjun",
    "Chloe",
    "Tariq",
    "Isla",
    "Mason",
    "Yara",
    "Ravi",
    "Elena",
    "Kofi",
    "Maya",
    "Felix",
    "Anika",
    "Theo",
    "Nadia",
    "Samir",
    "Ivy",
    "Rowan",
    "Lina",
    "Omar",
    "Grace",
    "Hugo",
    "Asha",
    "Wei",
]
LAST_NAMES = [
    "Ashgrove",
    "Brightwater",
    "Calderon",
    "Dunmore",
    "Ellery",
    "Fairbanks",
    "Galloway",
    "Hartwell",
    "Iverson",
    "Jaswal",
    "Kowalczyk",
    "Lindqvist",
    "Marchetti",
    "Nakamura",
    "Okafor",
    "Pemberton",
    "Quintero",
    "Rasmussen",
    "Sandoval",
    "Thorne",
    "Underhill",
    "Valdez",
    "Whitlock",
    "Xu",
    "Yarrow",
    "Zielinski",
    "Abernathy",
    "Bellamy",
    "Chandran",
    "Delacroix",
]

# Typical billed amount (CAD) per category, used for realistic claim sizes.
COST_RANGE: dict[str, tuple[float, float]] = {
    "drug_tier_1": (10, 80),
    "drug_tier_2": (30, 200),
    "drug_tier_3": (80, 600),
    "physiotherapy": (60, 130),
    "chiropractic": (30, 90),
    "massage_therapy": (60, 120),
    "psychology": (120, 220),
    "eye_exam": (60, 150),
    "eyewear": (100, 500),
    "travel_emergency": (300, 8000),
    "dental_basic": (80, 350),
    "dental_major": (400, 2000),
    "orthodontics": (300, 1500),
}


@dataclass
class Person:
    member_id: str
    first_name: str
    last_name: str
    dob: date
    plan_id: str
    coverage_start: date
    relationship: str  # self | spouse | child
    primary_member_id: str | None
    is_student: bool = False


@dataclass
class Usage:
    amount_paid: float = 0.0  # plan-paid this benefit year
    visits: int = 0
    last_service_date: date | None = None  # most recent paid service, any year
    lifetime_paid: float = 0.0


@dataclass
class Claim:
    member_id: str
    plan_id: str
    category: str
    amount: float
    service_date: date
    preauth_on_file: bool
    trip_days: int | None
    expected_violation: str  # "none" or one of VIOLATIONS
    expected_outcome: str  # approve | partial | deny
    notes: str
    claim_id: str = ""


@dataclass
class Dataset:
    seed: int
    people: list[Person]
    usage: dict[tuple[str, str], Usage]
    deductibles: dict[str, float]
    claims: list[Claim] = field(default_factory=list)


# --- date helpers ------------------------------------------------------------


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


class _Builder:
    def __init__(self, seed: int, plans: dict[str, Plan]):
        self.rng = random.Random(seed)
        self.seed = seed
        self.plans = plans
        self.plan_ids = sorted(plans)
        self.people: list[Person] = []
        self.usage: dict[tuple[str, str], Usage] = {}
        self.deductibles: dict[str, float] = {}
        self.ineligible: set[str] = set()
        self.reserved: set[tuple[str, str]] = set()
        self.claims: list[Claim] = []

    # --- random helpers ------------------------------------------------------

    def _date_between(self, start: date, end: date) -> date:
        return start + timedelta(days=self.rng.randint(0, (end - start).days))

    def _dob_for_age(self, age: int) -> date:
        # Birthday between Jan 1 and Sep 1, so the age is already reached on
        # AS_OF and does not change during the claim window.
        year = AS_OF.year - age
        return self._date_between(date(year, 1, 1), date(year, 9, 1))

    def _money(self, lo: float, hi: float) -> float:
        return round(self.rng.uniform(lo, hi), 2)

    # --- people --------------------------------------------------------------

    def build_people(self) -> None:
        rng = self.rng
        for i in range(N_PRIMARY):
            plan_id = self.plan_ids[i % len(self.plan_ids)]
            pid = f"M{i + 1:03d}"
            last = rng.choice(LAST_NAMES)
            if i in NEW_HIRE_INDEXES:
                start = self._date_between(date(2026, 5, 1), date(2026, 8, 31))
            else:
                start = self._date_between(date(2012, 1, 1), date(2024, 6, 30))
            primary = Person(
                pid,
                rng.choice(FIRST_NAMES),
                last,
                self._date_between(date(1965, 1, 1), date(1998, 12, 31)),
                plan_id,
                start,
                "self",
                None,
            )
            self.people.append(primary)
            n_dep = 0
            if rng.random() < 0.5:
                n_dep += 1
                spouse_last = last if rng.random() < 0.5 else rng.choice(LAST_NAMES)
                self.people.append(
                    Person(
                        f"{pid}-D{n_dep}",
                        rng.choice(FIRST_NAMES),
                        spouse_last,
                        primary.dob + timedelta(days=rng.randint(-1800, 1800)),
                        plan_id,
                        start,
                        "spouse",
                        pid,
                    )
                )
            for _ in range(rng.choice([0, 0, 1, 2, 3])):
                n_dep += 1
                self.people.append(
                    Person(
                        f"{pid}-D{n_dep}",
                        rng.choice(FIRST_NAMES),
                        last,
                        self._dob_for_age(rng.randint(1, 17)),
                        plan_id,
                        start,
                        "child",
                        pid,
                    )
                )

        primaries = [p for p in self.people if p.relationship == "self"]

        def add_child(primary: Person, age: int, student: bool) -> Person:
            n = sum(1 for p in self.people if p.primary_member_id == primary.member_id) + 1
            child = Person(
                f"{primary.member_id}-D{n}",
                rng.choice(FIRST_NAMES),
                primary.last_name,
                self._dob_for_age(age),
                primary.plan_id,
                primary.coverage_start,
                "child",
                primary.member_id,
                student,
            )
            self.people.append(child)
            return child

        # Eligible adult students (under both plans' student limits).
        for primary in rng.sample(primaries, 4):
            add_child(primary, rng.randint(22, 23), student=True)

        # Deliberately ineligible dependents: over the child limit without
        # student status, or over the student limit.
        for primary in rng.sample(primaries, N_PER_VIOLATION):
            elig = self.plans[primary.plan_id].eligibility
            if rng.random() < 0.5:
                child = add_child(primary, elig.child_age_limit + rng.randint(0, 2), False)
            else:
                child = add_child(primary, elig.student_age_limit + rng.randint(0, 1), True)
            self.ineligible.add(child.member_id)

    # --- usage ---------------------------------------------------------------

    def _u(self, member_id: str, cat: str) -> Usage:
        return self.usage.setdefault((member_id, cat), Usage())

    def build_baseline_usage(self) -> None:
        rng = self.rng
        for p in self.people:
            plan = self.plans[p.plan_id]
            ded = plan.deductible.annual_per_person
            if ded > 0:
                self.deductibles[p.member_id] = (
                    ded if rng.random() < 0.7 else float(rng.randint(0, int(ded) - 1))
                )
            for name, cat in plan.categories.items():
                if rng.random() < 0.5:
                    continue
                u = self._u(p.member_id, name)
                cap = self._baseline_cap(plan, cat)
                if cap:
                    u.amount_paid = round(rng.uniform(0.05, 0.3) * cap, 2)
                    u.visits = rng.randint(1, 6)
                if cat.frequency:
                    u.last_service_date = self._date_between(date(2023, 1, 1), AS_OF)
                    u.visits = int(u.last_service_date.year == BENEFIT_YEAR)
                if cat.lifetime_max:
                    u.lifetime_paid = round(rng.uniform(0, 0.3) * cat.lifetime_max, 2)

    def _baseline_cap(self, plan: Plan, cat: Category) -> float:
        if cat.annual_max:
            return cat.annual_max
        if cat.group and (gmax := plan.groups[cat.group].combined_annual_max):
            return gmax / 3
        if cat.frequency or cat.per_trip_max:
            return 0.0
        return 400.0  # e.g. drugs on a plan with no drug maximum

    # --- rule helpers --------------------------------------------------------

    def _group_cats(self, plan: Plan, group: str) -> list[str]:
        return [n for n, c in plan.categories.items() if c.group == group]

    def _budget_key(self, plan: Plan, cat: str) -> str:
        """The usage bucket a claim draws from: the group if it has a combined max."""
        c = plan.categories[cat]
        if c.group and plan.groups[c.group].combined_annual_max is not None:
            return c.group
        return cat

    def _waiting_done(self, p: Person, plan: Plan, cat: str) -> bool:
        months = plan.waiting_periods.months.get(cat)
        return months is None or add_months(p.coverage_start, months) <= CLAIM_START

    def _remaining(self, p: Person, plan: Plan, cat: str) -> float:
        c = plan.categories[cat]
        u = self._u(p.member_id, cat)
        rem = math.inf
        if c.annual_max is not None:
            rem = min(rem, c.annual_max - u.amount_paid)
        if c.group and (gmax := plan.groups[c.group].combined_annual_max) is not None:
            used = sum(self._u(p.member_id, n).amount_paid for n in self._group_cats(plan, c.group))
            rem = min(rem, gmax - used)
        if c.lifetime_max is not None:
            rem = min(rem, c.lifetime_max - u.lifetime_paid)
        if c.per_trip_max is not None:
            rem = min(rem, c.per_trip_max)
        return rem

    def _reset_budget(self, p: Person, plan: Plan, cat: str, frac: float) -> None:
        """Set the claim's usage bucket to at most `frac` of its limit."""
        c = plan.categories[cat]
        key = self._budget_key(plan, cat)
        if key != cat:
            names = self._group_cats(plan, key)
            gmax = plan.groups[key].combined_annual_max
            for n in names:
                self._u(p.member_id, n).amount_paid = round(
                    self.rng.uniform(0, frac) * gmax / len(names), 2
                )
        elif c.annual_max is not None:
            self._u(p.member_id, cat).amount_paid = round(
                self.rng.uniform(0, frac) * c.annual_max, 2
            )
        if c.lifetime_max is not None:
            self._u(p.member_id, cat).lifetime_paid = round(
                self.rng.uniform(0, frac) * c.lifetime_max, 2
            )

    def _candidates(
        self,
        options: list[tuple[str, str]],
        *,
        eligible: bool = True,
        waiting_done: bool | None = True,
        people: list[Person] | None = None,
    ) -> list[tuple[Plan, str, Person]]:
        out = []
        for plan_id, cat in options:
            plan = self.plans[plan_id]
            c = plan.categories.get(cat)
            for p in people if people is not None else self.people:
                if p.plan_id != plan_id:
                    continue
                if eligible and p.member_id in self.ineligible:
                    continue
                if c is not None:
                    if (
                        waiting_done is not None
                        and self._waiting_done(p, plan, cat) != waiting_done
                    ):
                        continue
                    if c.age_limit and age_on(p.dob, CLAIM_END) >= c.age_limit - 2:
                        continue
                    if (p.member_id, self._budget_key(plan, cat)) in self.reserved:
                        continue
                out.append((plan, cat, p))
        return out

    def _take(self, cands: list[tuple[Plan, str, Person]]) -> tuple[Plan, str, Person]:
        if not cands:
            raise RuntimeError("no candidate left for claim; adjust generator sizes")
        plan, cat, p = self.rng.choice(cands)
        if cat in plan.categories:
            self.reserved.add((p.member_id, self._budget_key(plan, cat)))
        return plan, cat, p

    def _clear_frequency(self, p: Person, cat: Category, name: str, service: date) -> None:
        if cat.frequency is None:
            return
        u = self._u(p.member_id, name)
        if self.rng.random() < 0.5:
            u.last_service_date = None
        else:
            back = cat.frequency.months + self.rng.randint(1, 12)
            u.last_service_date = add_months(service, -back)

    def _add(
        self,
        plan: Plan,
        cat: str,
        p: Person,
        amount: float,
        service: date,
        violation: str,
        outcome: str,
        notes: str,
        *,
        preauth: bool | None = None,
        trip_days: int | None = None,
    ) -> None:
        if preauth is None:
            threshold = plan.preauthorization.thresholds.get(cat)
            preauth = threshold is not None and (threshold == 0 or amount > threshold)
        c = plan.categories.get(cat)
        if trip_days is None and c is not None and c.max_trip_days:
            trip_days = self.rng.randint(2, c.max_trip_days)
        self.claims.append(
            Claim(
                p.member_id,
                plan.plan_id,
                cat,
                round(amount, 2),
                service,
                preauth,
                trip_days,
                violation,
                outcome,
                notes,
            )
        )

    # --- violating claims ----------------------------------------------------

    def _met_deductible(self, p: Person, plan: Plan) -> None:
        if plan.deductible.annual_per_person > 0:
            self.deductibles[p.member_id] = plan.deductible.annual_per_person

    def build_annual_max(self) -> None:
        # Canonical demo case: Bronze physio, $420 used of $500, $120 session.
        plan_a = self.plans["PLAN-A"]
        cands = self._candidates(
            [("PLAN-A", "physiotherapy")],
            people=[p for p in self.people if p.relationship == "self"],
        )
        _, _, p = self._take(cands)
        self._u(p.member_id, "physiotherapy").amount_paid = 420.0
        self._u(p.member_id, "physiotherapy").visits = 6
        self._met_deductible(p, plan_a)
        self._add(
            plan_a,
            "physiotherapy",
            p,
            120.0,
            date(2026, 10, 15),
            "annual_max_exceeded",
            "partial",
            "canonical demo: 420 of 500 used; plan share of 120 exceeds remaining 80",
        )

        options = [(pid, c) for pid in self.plan_ids for c in ("physiotherapy", "psychology")]
        for _ in range(N_PER_VIOLATION - 1):
            plan, cat, p = self._take(self._candidates(options))
            c = plan.categories[cat]
            remaining = 0.0 if self.rng.random() < 0.3 else self._money(10, 90)
            self._u(p.member_id, cat).amount_paid = c.annual_max - remaining
            self._met_deductible(p, plan)
            amount = math.ceil((remaining + self.rng.uniform(20, 80)) / c.coinsurance)
            self._add(
                plan,
                cat,
                p,
                amount,
                self._date_between(CLAIM_START, CLAIM_END),
                "annual_max_exceeded",
                "partial" if remaining > 0 else "deny",
                f"remaining {remaining:.2f} of {c.annual_max:.0f} annual max",
            )

    def build_per_visit_max(self) -> None:
        options = [(pid, c) for pid in self.plan_ids for c in ("chiropractic", "massage_therapy")]
        for _ in range(N_PER_VIOLATION):
            plan, cat, p = self._take(self._candidates(options))
            c = plan.categories[cat]
            self._reset_budget(p, plan, cat, 0.3)
            self._met_deductible(p, plan)
            amount = c.per_visit_max / c.coinsurance + self.rng.uniform(15, 60)
            self._add(
                plan,
                cat,
                p,
                amount,
                self._date_between(CLAIM_START, CLAIM_END),
                "per_visit_max_exceeded",
                "partial",
                f"plan share exceeds {c.per_visit_max:.0f} per-visit max",
            )

    def build_frequency(self) -> None:
        options = [(pid, c) for pid in self.plan_ids for c in ("eye_exam", "eyewear")]
        for _ in range(N_PER_VIOLATION):
            plan, cat, p = self._take(self._candidates(options))
            c = plan.categories[cat]
            service = self._date_between(CLAIM_START, CLAIM_END)
            last = service - timedelta(days=self.rng.randint(30, c.frequency.months * 30 - 30))
            u = self._u(p.member_id, cat)
            u.last_service_date = last
            u.visits = int(last.year == BENEFIT_YEAR)
            amount = self._money(COST_RANGE[cat][0], c.per_visit_max)
            self._add(
                plan,
                cat,
                p,
                amount,
                service,
                "frequency_limit",
                "deny",
                f"last paid {last.isoformat()}; limit {c.frequency.count} per "
                f"{c.frequency.months} months",
            )

    def build_waiting_period(self) -> None:
        options = [(pid, "dental_major") for pid in self.plan_ids] + [("PLAN-B", "orthodontics")]
        for _ in range(N_PER_VIOLATION):
            plan, cat, p = self._take(self._candidates(options, waiting_done=False))
            self._reset_budget(p, plan, cat, 0.15)
            threshold = plan.preauthorization.thresholds.get(cat)
            lo, hi = COST_RANGE[cat]
            if threshold:
                hi = min(hi, threshold - 20)
            amount = self._money(lo, hi)
            months = plan.waiting_periods.months[cat]
            self._add(
                plan,
                cat,
                p,
                amount,
                self._date_between(CLAIM_START, CLAIM_END),
                "waiting_period",
                "deny",
                f"coverage started {p.coverage_start.isoformat()}; {months}-month wait",
            )

    def build_excluded(self) -> None:
        options = [(pid, e.code) for pid in self.plan_ids for e in self.plans[pid].exclusions.items]
        for _ in range(N_PER_VIOLATION):
            plan, code, p = self._take(self._candidates(options))
            self._add(
                plan,
                code,
                p,
                self._money(100, 2000),
                self._date_between(CLAIM_START, CLAIM_END),
                "excluded_service",
                "deny",
                f"{code} is excluded under {plan.plan_id} §8",
                preauth=False,
            )

    def build_dependent_age(self) -> None:
        people = [p for p in self.people if p.member_id in self.ineligible]
        options = [(pid, "physiotherapy") for pid in self.plan_ids]
        for _ in range(N_PER_VIOLATION):
            plan, cat, p = self._take(self._candidates(options, eligible=False, people=people))
            self._u(p.member_id, cat).amount_paid = 0.0
            service = self._date_between(CLAIM_START, CLAIM_END)
            self._add(
                plan,
                cat,
                p,
                self._money(60, 120),
                service,
                "dependent_ineligible_age",
                "deny",
                f"child aged {age_on(p.dob, service)}, student={p.is_student}",
            )

    def build_preauth(self) -> None:
        options = [
            (pid, c) for pid in self.plan_ids for c in self.plans[pid].preauthorization.thresholds
        ]
        for _ in range(N_PER_VIOLATION):
            plan, cat, p = self._take(self._candidates(options))
            c = plan.categories[cat]
            self._reset_budget(p, plan, cat, 0.15)
            threshold = plan.preauthorization.thresholds[cat]
            lo, hi = COST_RANGE[cat]
            if threshold > 0:
                lo, hi = threshold + 20, threshold * 1.8
            amount = self._money(lo, hi)
            amount = min(amount, 0.5 * self._remaining(p, plan, cat) / c.coinsurance)
            self._add(
                plan,
                cat,
                p,
                amount,
                self._date_between(CLAIM_START, CLAIM_END),
                "preauth_missing",
                "deny",
                "pre-authorization required, none on file",
                preauth=False,
            )

    # --- clean claims --------------------------------------------------------

    def build_clean(self) -> None:
        rng = self.rng
        made = 0
        while made < N_CLEAN:
            plan_id = rng.choice(self.plan_ids)
            cat = rng.choice(sorted(self.plans[plan_id].categories))
            cands = self._candidates([(plan_id, cat)])
            if not cands:
                continue
            plan, cat, p = self._take(cands)
            c = plan.categories[cat]
            service = self._date_between(CLAIM_START, CLAIM_END)
            self._reset_budget(p, plan, cat, 0.4 if c.annual_max else 0.15)
            self._clear_frequency(p, c, cat, service)
            amount = rng.uniform(*COST_RANGE[cat])
            if c.per_visit_max is not None:
                amount = min(amount, c.per_visit_max / c.coinsurance)
            amount = min(amount, 0.5 * self._remaining(p, plan, cat) / c.coinsurance)
            self._add(plan, cat, p, amount, service, "none", "approve", "within all limits")
            made += 1

    def build(self) -> Dataset:
        self.build_people()
        self.build_baseline_usage()
        self.build_annual_max()
        self.build_per_visit_max()
        self.build_frequency()
        self.build_waiting_period()
        self.build_excluded()
        self.build_dependent_age()
        self.build_preauth()
        self.build_clean()
        self.rng.shuffle(self.claims)
        for i, claim in enumerate(self.claims, start=1):
            claim.claim_id = f"C{i:04d}"
        return Dataset(self.seed, self.people, self.usage, self.deductibles, self.claims)


def generate(seed: int = DEFAULT_SEED, plans: dict[str, Plan] | None = None) -> Dataset:
    return _Builder(seed, plans or load_plans()).build()


# --- SQLite ------------------------------------------------------------------

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE members (
    member_id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    dob TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    coverage_start TEXT NOT NULL,
    relationship TEXT NOT NULL CHECK (relationship IN ('self', 'spouse', 'child')),
    primary_member_id TEXT REFERENCES members(member_id),
    is_student INTEGER NOT NULL
);
CREATE TABLE usage (
    member_id TEXT NOT NULL REFERENCES members(member_id),
    benefit_year INTEGER NOT NULL,
    category TEXT NOT NULL,
    amount_paid REAL NOT NULL,
    visits INTEGER NOT NULL,
    last_service_date TEXT,
    lifetime_paid REAL NOT NULL,
    PRIMARY KEY (member_id, benefit_year, category)
);
CREATE TABLE deductibles (
    member_id TEXT NOT NULL REFERENCES members(member_id),
    benefit_year INTEGER NOT NULL,
    amount_applied REAL NOT NULL,
    PRIMARY KEY (member_id, benefit_year)
);
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    member_id TEXT NOT NULL REFERENCES members(member_id),
    plan_id TEXT NOT NULL,
    category TEXT NOT NULL,
    amount REAL NOT NULL,
    service_date TEXT NOT NULL,
    preauth_on_file INTEGER NOT NULL,
    trip_days INTEGER,
    expected_violation TEXT NOT NULL,
    expected_outcome TEXT NOT NULL,
    notes TEXT NOT NULL
);
"""


def write_sqlite(ds: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO meta VALUES (?, ?)",
            [
                ("synthetic", "true — fictional people, no real PHI"),
                ("seed", str(ds.seed)),
                ("generator_version", GENERATOR_VERSION),
                ("benefit_year", str(BENEFIT_YEAR)),
                ("as_of", AS_OF.isoformat()),
            ],
        )
        con.executemany(
            "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    p.member_id,
                    p.first_name,
                    p.last_name,
                    p.dob.isoformat(),
                    p.plan_id,
                    p.coverage_start.isoformat(),
                    p.relationship,
                    p.primary_member_id,
                    int(p.is_student),
                )
                for p in ds.people
            ],
        )
        con.executemany(
            "INSERT INTO usage VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    mid,
                    BENEFIT_YEAR,
                    cat,
                    u.amount_paid,
                    u.visits,
                    u.last_service_date.isoformat() if u.last_service_date else None,
                    u.lifetime_paid,
                )
                for (mid, cat), u in sorted(ds.usage.items())
                if u.amount_paid or u.visits or u.last_service_date or u.lifetime_paid
            ],
        )
        con.executemany(
            "INSERT INTO deductibles VALUES (?, ?, ?)",
            [(mid, BENEFIT_YEAR, amt) for mid, amt in sorted(ds.deductibles.items())],
        )
        con.executemany(
            "INSERT INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.claim_id,
                    c.member_id,
                    c.plan_id,
                    c.category,
                    c.amount,
                    c.service_date.isoformat(),
                    int(c.preauth_on_file),
                    c.trip_days,
                    c.expected_violation,
                    c.expected_outcome,
                    c.notes,
                )
                for c in ds.claims
            ],
        )
        con.commit()
    finally:
        con.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "benefitpilot.db")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    ds = generate(args.seed)
    write_sqlite(ds, args.out)
    n_primary = sum(p.relationship == "self" for p in ds.people)
    n_bad = sum(c.expected_violation != "none" for c in ds.claims)
    print(
        f"Wrote {args.out}: {n_primary} members (+{len(ds.people) - n_primary} dependents), "
        f"{len(ds.claims)} claims ({n_bad} with a deliberate violation)."
    )


if __name__ == "__main__":
    main()

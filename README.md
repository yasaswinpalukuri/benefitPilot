# BenefitPilot

An AI assistant for health-insurance plan members. It answers coverage and benefits questions with citations to plan sections, and flags plan-rule violations in claims.

> **Synthetic data only.** Every plan, member and claim in this repository is fictional. There is no real personal health information anywhere in the project.

> **Status:** Phase 2 (rules engine and `/claims/check`). See [CLAUDE.md](CLAUDE.md) for the build plan and [docs/DECISIONS.md](docs/DECISIONS.md) for architecture decisions.

## Why System 1 / System 2

- **System 1, Jev (TypeSafe):** fast, calibrated typed judgments, used for routing, guardrails and urgency in one pass.
- **System 2, an LLM with retrieval:** slow and deliberate, and used only when a written, grounded answer is needed.
- **Deterministic code:** every dollar amount and limit check is done in plain Python, never by a model.

```mermaid
flowchart TD
    U[Member question] --> J1[System 1: Jev<br/>intent / medical-advice? / urgency / needs-member-data?]
    J1 -->|medical advice p≥0.5| R[Refuse + safe redirect]
    J1 -->|urgency=emergency| E[Emergency message]
    J1 -->|confidence 0.2–0.9| H[Human escalation queue]
    J1 -->|claim_check| RE[Rules engine<br/>limits, deductibles, eligibility]
    J1 -->|coverage / procedure| RAG[Hybrid retrieval<br/>BM25 + vectors]
    RE --> L[System 2: LLM<br/>phrase result + cite]
    RAG --> L
    L --> J2[System 1: output check<br/>contains medical advice? grounded?]
    J2 -->|pass| A[Answer + citations]
    J2 -->|fail| R
    A --> LOG[(Audit log)]
    R --> LOG
    H --> LOG
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
cp .env.example .env              # works as is: ROUTER_BACKEND=fallback needs no keys
python scripts/generate_data.py   # writes data/benefitpilot.db (seeded, reproducible)
pytest
```

With Docker:

```bash
docker compose up --build         # API on http://localhost:8000/health
```

## Check a claim (rules engine only, no LLM)

```bash
uvicorn benefitpilot.api.app:app --reload
curl -X POST localhost:8000/claims/check -H "Content-Type: application/json" \
  -d '{"member_id": "M021", "category": "physiotherapy", "amount": "120", "date": "2026-10-15"}'
```

The response gives the outcome (`approve` / `partial` / `deny`), the covered amount, what the member pays, the violations, and every rule checked with its plan citation. For example: the plan share is $96.00, and the $500 annual maximum (with $420 already used) reduces it to $80.00, citing `PLAN-A §4.2.1`. Interactive docs are at `/docs`.

## Data

| Path | What it is |
|---|---|
| `data/plans/plan_a.md`, `plan_b.md` | Plan booklets (Bronze, Gold) with stable section IDs such as `PLAN-A §4.2.1` |
| `config/plans/plan_a.yaml`, `plan_b.yaml` | Machine-readable rules. A test keeps them in sync with the booklets |
| `scripts/generate_data.py` | Seeded generator: about 30 members plus dependents, YTD usage, and 100 claims with labeled violations |

---

<sub>BenefitPilot is a portfolio project. It uses synthetic data only and does not give medical advice.</sub>

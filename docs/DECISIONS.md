# Architecture Decision Records

Each record gives the decision, the alternatives considered, why this option won, and the trade-offs. New records are added as each phase lands.

---

## ADR-001: System 1 / System 2 split: Jev routes, LLM writes, code computes

**Status:** Accepted (Phase 1)

**Decision.** Every request goes through a fast typed-judgment pass (System 1: Jev, or a fallback classifier) before any LLM call. The LLM (System 2) runs only when a written, grounded answer is needed. All dollar amounts and limit checks come from a deterministic rules engine.

**Alternatives.**
- *LLM-only agent with tools.* This is the simplest to build. But every request pays LLM latency and cost, routing confidence is just text the model generates, and guardrails depend on the same model that might fail them.
- *A single classifier with no LLM.* This can't produce grounded natural-language answers.

**Why.** Routing and safety are classification problems, and they benefit from *calibrated probabilities* that can be thresholded, logged and re-tuned. Writing answers is a generation problem. Splitting them lets the cheap path handle refusals, emergencies and escalations without an LLM call, and keeps guardrails independent of the generator.

**Trade-offs.** There are two model dependencies instead of one. The router can also misroute, which the escalation band (ADR, Phase 4) and the output check reduce.

---

## ADR-002: Plan rules live in YAML; booklets are hand-written Markdown, kept in sync by a test

**Status:** Accepted (Phase 1)

**Decision.** `config/plans/*.yaml` is the machine-readable source for the rules engine. `data/plans/*.md` is the human-readable booklet that retrieval cites. In each booklet, every number appears exactly once, in a two-column table under a stable `§x.y` heading. Prose avoids repeating numbers. `tests/test_plan_sync.py` renders the expected tables from the YAML and requires them to equal the parsed Markdown tables, with no missing, extra or different rows.

**Alternatives.**
- *Generate the Markdown from the YAML.* Drift becomes impossible, but the booklets read like generated output. Real plan booklets are prose written for people, and retrieval quality depends on realistic text.
- *Extract the YAML from the Markdown with an LLM.* This breaks non-negotiable 4 (money is computed by code) and isn't deterministic.
- *A single source in one format.* YAML isn't citable prose, and Markdown isn't safe for arithmetic.

**Why.** It gives realistic documents for RAG and exact rules for the engine, with a CI-enforced contract between them. Putting each number in a single place in the Markdown is what makes the test *complete*: if a number could also appear in prose, the test could pass while the prose drifted.

**Trade-offs.** Writers must follow the table convention. Procedural facts that the engine doesn't model (the claim submission deadline and the appeal window in §10) live in prose and aren't sync-checked. If the engine starts enforcing the submission deadline, it moves into a table and the YAML.

---

## ADR-003: Strict, validated plan schema (Pydantic, `extra="forbid"`)

**Status:** Accepted (Phase 1)

**Decision.** YAML is loaded through frozen Pydantic models that reject unknown keys and check cross-references: waiting periods and pre-authorization must name real categories, a category can't also be excluded, section IDs are unique, and coinsurance must be in (0, 1].

**Alternatives.** Loading raw dicts, or using JSON Schema.

**Why.** A typo such as `anual_max` would otherwise silently mean "no maximum", which is the most expensive kind of bug in a benefits system. Pydantic is already in the stack (FastAPI), and the models give the Phase 2 engine typed access.

**Trade-offs.** Adding a new rule type needs a schema change. That's intentional.

---

## ADR-004: Synthetic claims are labeled by construction, never by the rules engine

**Status:** Accepted (Phase 1)

**Decision.** `scripts/generate_data.py` first picks the label (one of seven violation types, or clean), then builds the member state and claim that produce it. Each violating claim breaks exactly one rule. Clean claims are built with a safety margin: at most half the remaining budget, under per-visit caps, with pre-authorization on file when required, and frequency windows clear. Tests re-check a few labels with independent logic.

**Alternatives.** Generate random claims and label them by running the rules engine.

**Why.** If the engine labels the eval data, the Phase 7 violation-detection metrics would measure the engine against itself and always report 100%. Labeling by construction gives ground truth that doesn't depend on the engine.

**Trade-offs.** Each claim has a single violation, so real-world claims that break several rules at once aren't covered yet. Expected *amounts* aren't labeled, only the violation type and the outcome (approve, partial or deny). Amount checks will be hand-written unit tests in Phase 2.

---

## ADR-005: Maximums cap what the plan pays; fixed dates and seed for reproducibility

**Status:** Accepted (Phase 1)

**Decision.**
- Annual, lifetime, per-visit and per-trip maximums limit the *plan-paid* amount, not the billed amount. YTD usage is stored as plan-paid dollars.
- Age limits are exclusive ("Under age 21").
- The generator uses a fixed seed (42), a fixed usage snapshot date (2026-09-30) and a fixed claim window (October 2026). It never uses "today".
- `data/benefitpilot.db` is gitignored and rebuilt by the script in CI and in the Docker build.

**Why.** Plan-paid maximums are how most Canadian group plans word their limits. They also make the canonical demo work: with $420 used of $500, a $120 physio session at 80% coinsurance has a plan share of $96, which is capped at the remaining $80. Fixed dates mean the same seed produces a byte-identical database on any day, which is tested.

**Trade-offs.** The generated DB isn't browsable on GitHub. Anyone looking needs to run one command.

---

## ADR-006: Default System 2 is local Ollama (qwen2.5:7b on a home server); Anthropic is optional

**Status:** Accepted (Phase 1)

**Decision.** `LLM_PROVIDER=ollama` by default, pointing at Ollama on a home server ("Groot") over Tailscale, using `qwen2.5:7b` for chat and `nomic-embed-text` for embeddings. `LLM_PROVIDER=anthropic` is the second option and is used for the final benchmark run. The OpenAI adapter in the original stack table is deferred, since nothing needs it yet.

**Why.** It's free to iterate on, it keeps synthetic-but-PHI-shaped prompts on hardware I control, and it makes the "System 1 saves expensive LLM calls" argument concrete. A slow CPU-only LLM makes the latency gap visible.

**Trade-offs.** CPU-only inference is slow, so timeouts are generous (`LLM_TIMEOUT_S=300`) and CI must never call it. Answer quality from a 7B model is lower than from a frontier model, so the benchmark reports both. The Tailscale address in `.env.example` is a private tailnet IP. Only my tailnet can reach it, but it's still infrastructure detail in a public repo.

---

## ADR-007: pip + venv + hatchling; dependencies added per phase

**Status:** Accepted (Phase 1)

**Decision.** Use a standard `pyproject.toml` built with hatchling and installed with `pip install -e ".[dev]"`. Each phase adds only the dependencies its code needs: Phase 1 needs pydantic, PyYAML and FastAPI. Ruff handles both lint and format.

**Alternatives.** uv or Poetry, with lockfiles.

**Why.** Zero extra tooling for reviewers or CI, and pip works everywhere. Adding dependencies only when needed keeps the Docker image and attack surface small, and each addition is easy to explain.

**Trade-offs.** There's no lockfile yet. Pinning (pip-tools or uv) can come before the final benchmark if reproducibility needs it.

---

## ADR-008: A `/health`-only API in Phase 1

**Status:** Accepted (Phase 1)

**Decision.** Phase 1 ships a FastAPI app with only `GET /health`, a Dockerfile that generates the data at build time, and a compose file with a healthcheck.

**Why.** It proves that `docker compose up` works on day one and gives CI a Docker build job, so packaging problems show up early instead of in Phase 8.

**Trade-offs.** It's a tiny bit of code before it's needed.

---

## ADR-009: The rules engine is a pure function over a member snapshot

**Status:** Accepted (Phase 2)

**Decision.** `check_claim(plan, member_snapshot, claim) -> ClaimResult` does no I/O, makes no model calls and mutates nothing. A separate `MemberStore` turns SQLite rows into a frozen `MemberSnapshot` for one benefit year.

**Alternatives.**
- *An engine that queries the database itself.* This is shorter, but every test needs a database, and "what did the engine see?" becomes hard to answer.
- *A rule-engine library* (durable-rules, or a Drools-style engine). It brings its own DSL and runtime for roughly 15 rules that are clearer as plain Python.

**Why.** Testability and auditability. Every engine test builds its input by hand and states the expected dollar amount, so the tests read like a worked example from a benefits booklet. In Phase 5, the snapshot plus the result is exactly what goes into the audit log: you can replay any decision without the database.

**Trade-offs.** The caller has to build the right snapshot (the right benefit year), so the engine refuses mismatched years and plans instead of silently using the wrong usage.

---

## ADR-010: Money is `Decimal`, rounded half-up to cents; maximums are applied in a fixed order

**Status:** Accepted (Phase 2)

**Decision.**
- All amounts are `decimal.Decimal`. Values from YAML and SQLite go through `str()` first, so 420.1 becomes `Decimal("420.1")` and not its binary approximation.
- The plan share is computed as (billed − deductible) × coinsurance and rounded half-up to cents once.
- Caps then apply in a fixed order: per-visit → per-trip → annual → combined group → lifetime. Each cap that binds lowers the payment and is reported.
- The API returns money as JSON strings ("80.00").

**Why.** `10.05 × 0.70` is exactly 7.035. With floats it becomes 7.03499…, which rounds to 7.03. With `Decimal` it's 7.035, which rounds to 7.04. A test covers this case. Applying caps in a fixed order makes "why was I paid $20?" answerable step by step (56 → 40 per-visit → 20 annual).

**Trade-offs.** `Decimal` is slower and more verbose than float, which doesn't matter at this scale. SQLite stores usage as REAL. That's safe for 2-decimal synthetic values because of the `str()` conversion, but integer cents would be more robust if this moved to Postgres.

---

## ADR-011: Report every rule checked, not just the first failure; `/claims/check` is read-only

**Status:** Accepted (Phase 2)

**Decision.**
- Excluded or unknown services stop evaluation right away, since no other rule is meaningful for them.
- Otherwise every applicable rule is evaluated and returned as a `RuleCheck` (rule, passed, effect, detail, citation, violation). A rule's effect is `deny` (the claim isn't payable), `reduce` (a cap lowers the payment) or `info` (the deductible).
- Outcome: **deny** if any deny-type rule failed, or a cap reduced the payment to $0. **partial** if a cap reduced it. **approve** otherwise.
- A claim fully absorbed by the deductible is *approved at $0*, not denied, because no rule was broken.
- The endpoint opens SQLite with `mode=ro`. It answers "what would happen", and doesn't adjudicate or update usage.
- An unknown category is a 422 listing the valid ones (a client typo). An excluded category is a normal 200 denial (a real answer).

**Alternatives.** Fail fast on the first violation, which is simpler. Or have the endpoint post the claim and update usage, like a real adjudicator.

**Why.** The "Why this answer?" panel and the audit log need the whole picture. A member with two problems should learn both at once. A read-only check is safe to call from the chat flow in Phase 5 without side effects.

**Trade-offs.** The responses are larger. Real adjudication (posting claims, coordination of benefits, appeals) is out of scope and is listed under limitations.

---

## ADR-012: Scope limits of the Phase 2 engine

**Status:** Accepted (Phase 2)

**Decision.**
- Frequency limits are "1 per N months" only. The schema enforces `count: 1` because usage stores the last service date, not a full service history.
- The pre-authorization threshold is strict: "over $500" means $500.00 doesn't need one.
- Age limits are evaluated on the service date, and the birthday counts: a child turning 21 that day is no longer "under 21".
- `/claims/check` takes two optional fields beyond the original spec, `preauth_on_file` (default false) and `trip_days` (required for travel claims). Without them, two of the plan rules couldn't be checked.

**Why.** Each limit is a deliberate, tested boundary rather than an accident. The tests cover the day before, the day of and the day after for frequency, waiting periods and age limits.

**Trade-offs.** Rules like "2 massages per month" would need a service-history table. That's a natural extension, but no plan here needs it.

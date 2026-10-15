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

**Trade-offs.** CPU-only inference is slow, so timeouts are generous (`LLM_TIMEOUT_S=300`) and CI must never call it. Answer quality from a 7B model is lower than from a frontier model, so the benchmark reports both. `.env.example` uses the Tailscale MagicDNS hostname (`http://groot:11434`) rather than the tailnet IP, so no infrastructure address is published. Only machines on my tailnet can resolve or reach it.

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

---

## ADR-013: One chunk per plan section; the chunk ID is the citation ID

**Status:** Accepted (Phase 3)

**Decision.**
- Each `§` section with its own body text becomes one chunk, with ID `PLAN-A §4.2.1`.
- The indexed text starts with the plan name, the section heading and its parent headings ("Part of: §4 Health benefits > §4.2 Paramedical services").
- Tables are flattened to `label: value` lines for indexing, and the original Markdown is kept for display.
- Heading-only sections (§4, §10) aren't chunks; they appear as context in their children.
- The synthetic-data banner isn't indexed.

**Alternatives.** Fixed-size token windows with overlap, or recursive character splitting.

**Why.**
- Plan booklets are already split into self-contained, citable units. A fixed window would cut "Annual maximum: $500" away from "Physiotherapy", or merge two benefits into one chunk, and then a citation would point at the wrong rule.
- Making the citation the ID means grounding needs no lookup table.
- Parent headings fix the "short leaf section" problem: §4.2.1 alone never says "paramedical".

**Trade-offs.** Chunk sizes are uneven (§8 Exclusions is long, §4.1.1 is two lines). That's fine at about 60 chunks. Long real booklets would need a size cap that splits *within* a section while keeping the same citation.

---

## ADR-014: nomic-embed-text over Ollama, an in-memory numpy index, and BM25-only as the offline mode

**Status:** Accepted (Phase 3). Changes the stack table: no Chroma/pgvector and no sentence-transformers.

**Decision.**
- Embeddings come from `nomic-embed-text` on the home Ollama server through `/api/embed`, with nomic's `search_document:` / `search_query:` task prefixes.
- Document vectors are cached on disk, keyed by sha256(model + text).
- The "vector store" is a normalized numpy matrix, where a dot product gives the cosine.
- With `EMBEDDINGS_BACKEND=none` (the default, used by CI) retrieval is BM25-only.
- If Ollama is configured but unreachable, retrieval degrades to BM25 and sets a `degraded` reason instead of failing the request.

**Alternatives.**
- *Chroma or pgvector.* Real services with persistence and ANN indexes, but pointless for 63 vectors, and one more container to run.
- *sentence-transformers in-process.* This adds about 2 GB of torch to the image, and the user already runs nomic on the home server.
- *A deterministic "hashing" embedder for offline use.* I built and measured this one. It made results **worse** than BM25 alone (hit@3 0.77 vs 0.86) because its "similarities" are noise that RRF then trusts, so I removed it. Fusion mechanics are unit-tested with a small stub embedder instead.

**Why.** It's the simplest thing that's correct at this scale, and it keeps CI offline. Degrading gracefully matters because the embedder runs on a home server reached over Tailscale.

**Trade-offs.** At thousands of chunks or multiple tenants, swap the matrix for pgvector. The `HybridRetriever` interface wouldn't change. The cache file must be invalidated if the model changes, which is handled because the model name is part of the key.

---

## ADR-015: RRF fusion where only meaningful BM25 matches vote; a cosine-based relevance gate

**Status:** Accepted (Phase 3). Thresholds are provisional and will be re-checked on the Phase 7 golden set.

**Decision.**
- Fusion is reciprocal-rank fusion (`k=60`) over the top 20 of each retriever.
- Only BM25 hits scoring ≥ `min_bm25` (4.0) get a vote. Ties go to the better vector rank.
- A separate **relevance gate** decides whether to answer at all: top cosine ≥ `min_cosine` (0.57) when vectors exist, otherwise top BM25 ≥ `min_bm25`.
- When the gate fails, the answer must be "I couldn't find that in your plan". Hits are still returned for the audit log.

**What I measured** (`python scripts/eval_retrieval.py`, 48 questions: 28 lexical, 15 paraphrase, 5 out of scope; nomic-embed-text on the home server):

| Variant | hit@1 | hit@3 | MRR | Gate accuracy |
|---|---|---|---|---|
| BM25 only | 0.70 | 0.86 | 0.78 | 0.69 |
| Vector only | 0.95 | 1.00 | 0.97 | 1.00 |
| Hybrid, plain RRF, BM25-or-cosine gate (first version; measured on the earlier 45-question set) | 0.88 | 0.95 | 0.91 | 0.93 |
| **Hybrid, gated BM25 votes, cosine gate (chosen)** | **0.93** | **1.00** | **0.96** | **1.00** |

**Why.**
- *Plain RRF hurt.* On paraphrases ("new specs", "my son in university"), BM25's top hits came from generic words ("plan", "benefits") with scores around 3. Any chunk in both lists beat the vector's correct top-1. Letting only meaningful lexical matches vote fixed this without dropping BM25 for exact rare terms.
- *BM25 can't gate relevance.* "Write me a poem **about** the ocean" scored 3.9 (it matched "§1 **About** this plan"), while "What is the deductible?" scored 1.5. Cosine separated the two groups cleanly: every answerable question scored ≥ 0.604 and every out-of-scope one ≤ 0.535. 0.57 is the midpoint.
- RRF uses ranks only, so BM25 scores (unbounded) and cosines never need to be put on one scale.

**Trade-offs and honesty notes.**
- On this corpus, vector search alone is as good as hybrid (hit@1 0.95 vs 0.93). I keep hybrid for three reasons: exact-term robustness as booklets grow, the BM25 fallback when the embedder is down, and because it's cheap.
- The thresholds were tuned on the same 48 questions they're reported on, so the numbers are optimistic. Phase 7 re-measures on the separate golden set.
- The cosine gap (0.535 vs 0.604) is narrow. Expect to revisit it once there are more out-of-scope examples.
- CI enforces BM25-only floors (lexical hit@3 ≥ 0.95, overall ≥ 0.80). The Ollama numbers are checked by an opt-in test (`RUN_OLLAMA_TESTS=1`).

---

## ADR-016: One typed question set behind a `DecisionRouter` protocol

**Status:** Accepted (Phase 4)

**Decision.**
- Both routers implement `async decide(state, questions) -> Decisions`.
- The questions are defined once (`routing/questions.py`) in TypeSafe's primitive vocabulary:
  - `intent` is a Choice over five options, including the no-match option `out_of_scope`.
  - `asks_medical_advice` and `needs_member_data` are Nouls.
  - `urgency` is a Score whose three levels each describe a concrete situation.
  - The output check uses two Nouls: `answer_contains_medical_advice` and `answer_supported_by_context`.
- The four routing questions are independent, so they go in **one** request. The output check needs the LLM's answer, so it's a second request.
- `Decisions` keeps the raw probabilities, the versioned model ID, latency, tokens and cost, and is stored verbatim in the audit log (Phase 5).
- `RouterState` carries the message, the plan name and, for the output check, the answer and the retrieved text. It never carries names or other identifiers.

**Alternatives.** An LLM prompt that returns a JSON route (`{"route": "coverage"}`), or separate classifiers with no shared contract.

**Why.**
- The TypeSafe docs recommend one narrow judgment per question, a no-match option, and sending independent questions together. Following that gives us calibrated, separately thresholdable signals instead of one opaque label.
- A shared question set means Jev and the fallback answer *identical* questions. That makes the Phase 7 head-to-head an apples-to-apples comparison.
- Storing raw probabilities means thresholds can be re-tuned from the logs with no re-inference.

**Trade-offs.** Writing good question criteria takes care, and a vague option description shows up as low confidence. The `intent` options carry `what` / `not_for` / `examples` fields because `coverage_question` and `claim_check` are easy to confuse.

---

## ADR-017: JevRouter pins `jev-1.13.0`, relies on the SDK's retry policy, and fails over to the fallback

**Status:** Accepted (Phase 4). Verified against docs.typesafe.ai and `typesafe-sdk` 0.7.5 on 2026-10-14.

**Decision.**
- The router uses `AsyncTypeSafeClient` with `model="jev-1.13.0"` sent on every request.
- Retries use `RetryPolicy(max_retries=4, backoff_initial=0.5, backoff_max=8.0, timeout=30.0)`.
- Cost is `input_tokens × $0.042 / 1M`, because Jev bills input tokens only.
- When `ROUTER_BACKEND=jev`, Jev is wrapped in a `FailoverRouter`. If it still fails after retries, the FallbackRouter answers, and the result is labelled `router="fallback (jev failed)"` with the error.
- Tests mock the **HTTP transport** (`httpx2.MockTransport`), not the SDK. The real SDK builds and parses every request.

**Why.**
- *Pinning:* aliases move. `jev-1.14.0` was released today, and `jev-latest` will eventually point to it. Thresholds tuned on 1.13 probabilities wouldn't silently carry over. Each response's `model` field is logged anyway.
- *Retries:* the SDK already does exponential backoff with jitter, honours `retry-after`, retries 408/429/5xx/529 and connection errors, and doesn't retry 4xx auth or validation errors. Writing a second retry loop on top would multiply attempts. Rate limits are per organization and not published, so it allows a couple more retries than the default.
- *Failover:* non-negotiable 6 (the project must run without Jev) also applies at runtime, for example during an outage or after a revoked key. Labelling the failover keeps the audit honest.
- *Transport-level mocks* also test that our questions serialize to the documented wire format: pinned model, Bearer auth, criteria shapes.

**Trade-offs.** A failover silently lowers routing quality for that request; it's visible in the audit log, not to the member. An alert on the failover rate would be the production follow-up. These tests check the documented response shape, and the live API will be checked once the key arrives.

---

## ADR-018: FallbackRouter is TF-IDF + logistic regression with high-recall safety rules

**Status:** Accepted (Phase 4)

**Decision.**
- One logistic regression per routing question, over word 1–2-gram and character 3–5-gram TF-IDF. Character n-grams let "physio" match "physiotherapy".
- The models are trained at startup from about 100 hand-written examples in `routing/data/router_training.jsonl`, which takes under a second.
- For `asks_medical_advice` and the emergency level of `urgency`, the final probability is the **max** of the model and a keyword rule (advice phrasing plus a symptom or treatment term; emergency phrases).
- The output check uses heuristics: dosage and advice phrasing for medical advice, and "every $ amount, % or N months in the answer must appear in the retrieved text" for grounding.
- Choice and Score confidence use TypeSafe's published formulas, so both routers report on the same scale.

**Alternatives.**
- An LLM limited to fixed options. It's slow on a CPU-only server and not calibrated.
- Embedding similarity to labelled examples. It needs the embedder, which can be offline.
- A pickled, pre-trained model. Unpickling is arbitrary code execution, and it hides how the model was built.

**Why.** The fallback has two jobs: run with no key, and be the *baseline* that Jev is measured against. A classical model is cheap, deterministic, and easy to explain. The rule layer exists because CI gates guardrail recall at 98% on this router, and with about 100 examples a classifier alone won't reliably reach that. Erring towards refusal matches the cost asymmetry.

**Trade-offs.**
- About 100 training examples is small, and the training data is kept **separate** from the Phase 7 golden and red-team sets so the benchmark doesn't leak.
- The keyword rules over-refuse some legitimate questions. Phase 7 will measure the false-refusal rate as well as recall.
- The output-check heuristics are weak: they can't detect unsupported claims that contain no numbers. That's a known gap that Jev fills.

---

## ADR-019: The routing policy works on raw probabilities, ordered by the cost of each error

**Status:** Accepted (Phase 4). Values are starting points, to be tuned in Phase 7.

**Decision.** `routing/policy.py` is pure code, evaluated in this order:
1. **Emergency** if P(urgency = life-threatening level) ≥ 0.5.
2. **Refuse** if `asks_medical_advice` ≥ 0.5 **or** P(intent = medical_advice_request) ≥ 0.5.
3. **Auto-route** if the top intent probability is ≥ 0.9.
4. **Escalate** to the human queue otherwise.

The output check fails if `answer_contains_medical_advice` ≥ 0.5 or `answer_supported_by_context` < 0.5.

**Why this order.**
- Someone with chest pain needs "call 911", not "I can't give medical advice", so emergency is checked before refusal.
- Refusal is checked before auto-routing because a confident "coverage" intent shouldn't override a medical-advice signal. "Is physio covered, and which stretches should I do?" must be refused.
- Two independent refusal signals, either of which refuses, trade some extra false refusals (cheap) for fewer misses (expensive).

**Details.**
- Thresholds compare *probabilities*, not TypeSafe's `confidence`. CLAUDE.md's rule is "top intent ≥ 0.9", and with 5 options a top probability of 0.9 is a confidence of 0.875, so the two aren't interchangeable. Both are logged.
- Emergency uses the *top level's probability*, not the Score value. A score of 1.0 can mean "all probability on 'soon'" or "half routine, half emergency", and only the second should trigger the emergency message. A test covers this case.
- The spec's lower escalation bound (0.2) was removed: with 5 intents the top probability is always at least 0.2, so the setting could never fire.

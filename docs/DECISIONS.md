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

---

## ADR-020: A LangGraph state machine where safety paths never reach the LLM and the output check can only make an answer safer

**Status:** Accepted (Phase 5)

**Decision.** `graph/pipeline.py` compiles this graph:

```
context → route ─┬→ canned (emergency / refuse / out_of_scope / escalate) → END
                 ├→ claim ───┐
                 └→ retrieve ┴→ generate → guard → output_check → END
```

- **Canned paths.** Emergency, refusal, out-of-scope and escalation replies are fixed text in `guardrails/messages.py`. These paths make no LLM call, so a prompt can't change what they say.
- **Early exits.** `claim` and `retrieve` can end the request without an LLM call: no member selected, claim details unclear, or retrieval found nothing relevant ("I couldn't find that in your plan").
- **The output check can only downgrade an answer.** If the answer contains medical advice, it becomes the refusal. If an unsupported *claim* explanation, it becomes the deterministic template. If an unsupported *coverage* answer, it's escalated to a human. It never turns a refusal into an answer.
- **One audit record per request.** `Pipeline.ask` wraps the graph and writes the audit record whichever path ran, so no branch can forget to log.
- **Citation fallback.** If the model's answer has no valid inline citation, the response lists the sections it was grounded in, and a guardrail note records that. Citations the model invents are dropped, because only IDs that were actually in its prompt count.

**Alternatives.** A plain `async def` with if/else, which is shorter. Or an LLM agent with tools, which is flexible but makes routing a model decision rather than code.

**Why.** The routing *is* the product here, and a declared graph makes it inspectable: every node has one job, and every edge is a named policy decision. LangGraph was also in the planned stack (I used it in Groot). Keeping the audit write *outside* the graph means "everything is logged" holds by construction, not by discipline in each branch.

**Trade-offs.** LangGraph adds a dependency and some indirection for a graph that has no cycles yet. It pays off when we add retries or human-in-the-loop resumption, which LangGraph supports with checkpointing.

---

## ADR-021: Offline relevance gate at BM25 ≥ 2.0, separate from the fusion vote threshold (amends ADR-015)

**Status:** Accepted (Phase 5)

**Context.** Phase 5's tests sent "Is physiotherapy covered?" through the whole pipeline in BM25-only mode. It scored 3.62 and was rejected by the 4.0 gate as "not in your plan". In Phase 3 that single threshold did two jobs: deciding which BM25 hits may vote in fusion, and gating relevance when no vectors exist.

**Decision.** Split it into `min_bm25: 4.0` (fusion vote) and `min_bm25_gate: 2.0` (relevance gate when there are no vectors). The gate threshold was chosen by measuring answerable questions against in-scope questions the plan doesn't cover (acupuncture, hearing aids, IVF, …):

| BM25 gate | Answerable questions passed | Not-in-plan questions blocked |
|---|---|---|
| 1.5 | 0.91 | 0.12 |
| **2.0** | **0.88** | **0.62** |
| 3.0 | 0.88 | 0.62 |
| 4.0 (old) | 0.65 | 0.75 |

**Why.** By the time a question reaches retrieval, the router has already handled out-of-scope chatter ("poem about the ocean"). The offline gate only has to separate *plan questions the plan answers* from *plan questions it doesn't*, which is a different job from Phase 3's. At 4.0 the gate turned away a third of legitimate questions. At 2.0, most not-in-plan questions are still blocked, and the output check catches ungrounded answers after the LLM.

**Trade-offs.** This is offline mode only; with vectors, the cosine gate (0.57) applies. Abbreviations like "physio" score 1.97 and still miss in BM25-only mode. A small domain synonym list is a cheap future fix. The measurement set is small and was written by me.

---

## ADR-022: Claim arguments are *selected*, not generated; money is checked again after phrasing

**Status:** Accepted (Phase 5)

**Decision.**
- **Selection, not generation.** For a member's claim question, code extracts every `$` amount with its surrounding words. Two extra Choice questions go out **in the same routing request**: `claim_category` (the member's plan categories, the excluded services, and `unclear`) and `claim_amount` (the candidate amounts and `none`). They're used only if the route is `claim_check`, and only when the selected option's probability is ≥ 0.5. Otherwise the assistant asks for the service and amount.
- **Service date.** The service date is today, which is injected so tests and evals are deterministic.
- **Money guard.** After the LLM phrases the rules-engine result, `guard` checks three things:
  1. **Numbers:** every `$` amount and `%` in the answer must appear in the engine's output, compared by value, so $80 matches $80.00.
  2. **Polarity:** a covered or partly covered result must not be described as "not covered" or "denied", and a denial must say so.
  3. **Bottom line:** the answer must state both what the plan pays and what the member pays.

  Any failure replaces the whole answer with `claim_template()`, a deterministic explanation built only from the result. The template is also used when the LLM is unavailable or the output check marks the answer as unsupported.

**Alternatives.** Ask the LLM to extract `{category, amount}` as JSON, or use regex plus keywords alone. The keyword version is what the FallbackRouter does, as its baseline.

**Why.**
- *Selection* means the rules engine can only ever receive a real plan category and an amount the member typed; a model can't invent "$1,200" or "acupuncture_premium". It follows TypeSafe's "select instead of generate" and "speculative fan-out" patterns, so it costs no extra round trip.
- The money guard enforces non-negotiable 4 *after* phrasing, too: "the LLM may phrase the result, but the numbers come from the rules engine" becomes something a test can check, not a hope about the prompt. A test feeds in an invented "$96 back" and checks that the template replaced it.

**Found in the first live run.** For the canonical claim ($420 used, $120 session), qwen2.5:7b wrote: "your plan has already paid $80 … a new session would not be covered." **Every number in it is real**, yet the conclusion is the opposite of the engine's: $80 *will* be paid. A numbers-only guard would have let it through. It was rejected that time only because of a separate bug: the fallback's grounding heuristic compared "$80" and "$80.00" as strings. Fixing that bug alone would have *served the wrong answer*. That is why the polarity and bottom-line checks exist. The draft is now a regression test (`tests/test_grounding.py`), and the claim facts given to the LLM were relabelled ("Amount the plan will pay for this claim: $80.00"), because the model had read the $80 as past usage.

**Trade-offs.**
- The guard is strict: an LLM that says "80%" in a claim explanation (true, but not in the result facts) falls back to the template. Correct but plainer wording is the safe direction.
- The polarity check is a regex over negation phrases. It's good enough to catch reversed conclusions, and it can be fooled by unusual phrasing. The System 1 output check is the second line of defence.
- Questions about past dates ("my claim from March") use today's date for now.

---

## ADR-023: System 2 is local qwen2.5:7b by default; the Anthropic adapter defaults to Claude Opus 5.5 with refusal fallbacks

**Status:** Accepted (Phase 5). The Anthropic details were checked against the bundled Claude API reference and `anthropic` 1.14.0.

**Decision.**
- **Common contract.** `LLMClient.complete(system, user) -> LLMResult` reports text, model, tokens, latency and cost.
- **Ollama:** `/api/chat` with `temperature 0`, a fixed `seed` and `num_predict` 400 for repeatable evals. The timeout is 300 s. Cost is reported as $0 (local hardware; electricity is ignored).
- **Anthropic:**
  - `AsyncAnthropic.beta.messages.create` with `ANTHROPIC_MODEL`, which defaults to `claude-opus-5-5` (the current default model; I didn't pick a cheaper tier on your behalf).
  - **No `temperature`:** current models reject sampling parameters.
  - **Effort:** `output_config.effort="low"`, because thinking can't be disabled on Opus 5.5 and this is a short phrasing task.
  - **Refusals:** server-side fallbacks are on (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). A final `stop_reason == "refusal"` raises `LLMError`, so the graph takes its safe path.
  - **Cost** comes from the model that actually served the response: Opus 5.5 is $4 / $20 per million input/output tokens, Sonnet 5.5 $2 / $10, Haiku 5.5 $0.10 / $0.50.
- **Prompts** (`llm/prompts.py`) are short: cite IDs in brackets, don't compute numbers, no medical advice, and treat `<member_question>` as data, not instructions.

**Why.** Local first keeps iteration free and keeps synthetic-but-PHI-shaped prompts on my own hardware. The Anthropic path is for the final benchmark, where answer quality and real cost are what's being compared. One interface means the graph and the audit don't know which one ran.

**Trade-offs.** qwen2.5:7b on a CPU takes about 25 prompt tokens/s plus 7 generated tokens/s, so a RAG answer takes tens of seconds. In the first live run (3 retrieved chunks, about 350–550 prompt tokens, 50–70 output tokens), LLM answers took 24–29 s end to end, while the emergency route, which has no LLM call, took 24 ms. Retrieval is limited to the top 3 chunks partly for this reason. Opus 5.5 with thinking always on costs more per answer than a non-thinking small model; `ANTHROPIC_MODEL` makes that a config change.

---

## ADR-024: Audit log in its own SQLite database: full JSON record plus indexed columns; escalations as a queue table

**Status:** Accepted (Phase 5)

**Decision.**
- `audit/log.py` writes one `AuditRecord` per request to `data/audit/audit.db`, which is separate from the read-only member database. The record holds:
  - the question, member and final route, plus the route reason and the initial route;
  - the router's name, versioned model, failover error, latency, tokens, cost and **all raw System 1 answers**;
  - the claim arguments with their probabilities;
  - the retrieval hits with BM25 and vector ranks, scores and the gate signals;
  - the full rules-engine result;
  - the LLM call (provider, model, tokens, latency, cost, the **SHA-256 of the prompt**, and the raw draft);
  - the guardrails that fired, the output-check decisions and verdict, and the final answer and citations;
  - latency per stage and the total cost.
- Escalations also get a row in an `escalations` table (status `open`), which `GET /escalations` serves.

**Alternatives.** JSON-lines files (simple, but no querying and awkward concurrent writes), Postgres (the right answer later), or an observability tool such as LangSmith (an external service, and these are health-shaped records).

**Why.**
- Raw probabilities make the log *replayable*: Phase 7 can re-tune thresholds by re-running the policy over stored decisions with no new inference.
- The prompt *hash* rather than the full prompt avoids storing chunk text a second time. The prompt can be rebuilt from code, the logged chunk IDs and the question. The raw draft is stored because it's the thing that was checked and maybe rejected.
- A separate database keeps the "the claim check never writes" guarantee from ADR-011 intact.

**Trade-offs.** SQLite serializes writes, which is fine for a demo and not for production traffic. There's no retention policy yet: a real deployment would need one, plus encryption at rest, since real audit logs would contain PHI.

---

## ADR-025: The UI is a pure API client, and the "Why this answer?" panel shows only what the API recorded

**Status:** Accepted (Phase 6)

**Decision.**
- `ui/streamlit_app.py` talks to FastAPI over HTTP (`ui/api_client.py`) and never imports the pipeline.
- For each answer it calls `POST /ask`, then `GET /audit/{id}`. The panel renders those two responses:
  - the route and its reason;
  - the System 1 probabilities against the policy's *own* thresholds (served by the new `GET /thresholds`);
  - the claim arguments System 1 selected;
  - the rules-engine table;
  - the retrieved sections with their BM25 and vector ranks;
  - the output check, the guardrails that fired, the LLM model and tokens, and per-stage latency and cost.
- Two supporting endpoints were added: `GET /members` and `GET /members/{id}/usage`. The usage summary is computed in `rules/summary.py`, in code, from the same snapshot and plan rules as the engine.
- Docker runs both services from one image: `api`, and `ui` with `BENEFITPILOT_API_URL=http://api:8000`. Streamlit is an optional `[ui]` extra.

**Alternatives.** A Streamlit app that imports the pipeline directly, which is simpler with one process. Or React, which is the planned stretch goal.

**Why.** The panel exists to make the audit trail visible. If it recomputed anything, it could disagree with the log. Being a pure client also means a React front end later is a rewrite of presentation only, with the same API. Thresholds come from the API so the dashed lines in the charts are the lines the policy used.

**Trade-offs.** There are two HTTP calls per answer (`/ask`, then `/audit`). It's negligible next to a 15–30 s local LLM call. The members endpoints expose synthetic names; in a real system they'd need authentication and per-member authorization.

---

## ADR-026: One chart language for every System 1 signal; bugs found by looking at the real UI

**Status:** Accepted (Phase 6)

**Decision.**
- **Bar charts.** Every Choice, Score and Noul is a horizontal bar chart on a fixed 0–100% axis, built in `ui/charts.py`. They use one hue (the palette's blue), 4 px rounded data-ends and value labels.
- **Threshold markers.** Each chart draws a dashed neutral rule, or a tick, at the threshold the policy actually uses.
- **Hover.** Tooltips give exact values.
- **Usage meters.** Year-to-date usage uses Streamlit progress meters with the same blue.
- **No red or green.** Status colours aren't used: a refusal isn't "bad data", and a nearly used benefit isn't an alarm.

**Why.** A reader should see *why* a route was taken: the bar crosses the line, or it doesn't. Every chart is a single series, so no legend is needed and no categorical palette needs validating.

**Bugs the tests didn't catch, found by running the real UI in a browser:**
1. **Dollar signs rendered as LaTeX.** Streamlit treats text between two `$` signs as math, so "$420 … $120" in a button label became a formula, and every answer with two amounts would have been garbled. Fix: `md()` escapes `$` wherever text is rendered as Markdown. A test now fails if an unescaped `$` reaches the page.
2. **Collapsed charts.** Streamlit sizes Altair charts to "fit", so a fixed pixel height had to hold the title and axes too, and all rows collapsed into one band. Fix: a step-based height (`alt.Step(28)` per row), which doesn't depend on an outer height.
3. **Invisible labels in dark mode.** Vega's default mark text is black, so value labels vanished on the dark theme. Fix: labels use the palette's secondary text ink for the active mode (`st.context.theme.type`), falling back to a mid-grey that's readable on both.
4. **Red meters, and lost dark mode.** Streamlit's default accent is red, which made "$420 of $500 used" look like an error. A bare `[theme] primaryColor` fixed the colour but silently forced light mode. Fix: `[theme.light]` and `[theme.dark]` sections in `.streamlit/config.toml`.

**Trade-offs.** AppTest checks content and structure, not pixels, so visual regressions still need a look. Chart *layout* is verified by rendering specs offline with `vl-convert` and by a manual browser check, not in CI.

---

## ADR-027: Test the UI against recorded *real* API responses

**Status:** Accepted (Phase 6)

**Decision.** `tests/test_ui.py` drives the actual Streamlit script with `streamlit.testing.v1.AppTest`. The API client is replaced by a fake that replays JSON captured, in the same test session, from the real FastAPI app: real routing policy, rules engine and audit log, with a scripted router and LLM. The tests:
- click the example questions and select a member;
- check the answer, route label, rules table, retrieved sections and the number of charts;
- check that the emergency path shows no LLM trace.

The API client itself is tested with `httpx.MockTransport`.

**Why.** Hand-written fake responses drift from the API. Recording them from the real app means a change to `/ask` or `/audit` that breaks the UI fails a test.

**Trade-offs.** AppTest can't see layout or colour; see ADR-026.

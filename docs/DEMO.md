# Demo script

A walkthrough of about 2 minutes that shows each route once and the same claim under both plans. Run the API with `ROUTER_BACKEND=jev` and `EMBEDDINGS_BACKEND=ollama`, so the panel shows Jev's probabilities and hybrid retrieval.

The two members:

| Member | Plan | Physio used this year | Physio rule |
|---|---|---|---|
| M021 Diego | PLAN-A Bronze | $420 | 80% coinsurance, $500 annual maximum |
| M006 Asha | PLAN-B Gold | $399.34 | 100% coverage, $1,500 annual maximum |

| # | Member | Question | What to point at in "Why this answer?" |
|---|---|---|---|
| 1 | M021 (Bronze) | Is physiotherapy covered under my plan? | Route `coverage`; the retrieved Bronze plan section and its citation |
| 2 | M021 (Bronze) | I have used $420 of physio this year. Will a $120 session be covered? | Route `claim_check`; the rules engine's steps: $96 at 80%, capped at the $80 left in the $500 maximum, so the member pays $40 (`PLAN-A §4.2.1`) |
| 3 | M006 (Gold) | Will a $120 physio session be covered? | Same question, different plan: 100% coverage, $120 paid, the member pays $0, $1,100.66 left (`PLAN-B §4.2.1`). The rules come from YAML, not from the model |
| 4 | M006 (Gold) | Will a $120 physio session be covered, and should I use heat or ice after? | Route `refuse_medical`: medical advice smuggled into a plan question, caught by its own System 1 question |
| 5 | any | I have crushing chest pain right now. | Route `emergency`: fixed 911 text, no LLM call |
| 6 | any | How do I submit a claim for dental? | Route `procedure`; the claims-submission section |

Close on the audit view (`GET /audit/{id}` at `localhost:8000/docs`): the same probabilities, chunks and rule result that the panel showed, stored for every request.

Run all six once before recording. The first question also loads the LLM on the server, which can take a minute.

## Recording the GIF

1. Start the API and the UI (see the README), and open the UI at `http://localhost:8501`.
2. Record the browser window with a screen-to-GIF tool (for example ScreenToGif on Windows), at about 1280 px wide and 10–15 fps.
3. Trim the waits for the LLM, and keep the file under 10 MB so GitHub renders it.
4. Save it as `docs/demo.gif` and uncomment the image line near the top of the README.

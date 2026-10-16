# Demo script

A 60–90 second walkthrough that shows each route once. Run the API with `ROUTER_BACKEND=jev` and `EMBEDDINGS_BACKEND=ollama` so the panel shows Jev's probabilities and hybrid retrieval.

| # | Member | Question | What to point at in "Why this answer?" |
|---|---|---|---|
| 1 | M021 | Is physiotherapy covered under my plan? | Route `coverage`; the retrieved plan sections and the citation |
| 2 | M021 | I have used $420 of physio this year. Will a $120 session be covered? | Route `claim_check`; the rules engine's steps: $96 at 80%, capped at the $80 left in the $500 maximum |
| 3 | M021 | Will a $120 physio session be covered, and should I use heat or ice after? | Route `refuse_medical`: medical advice smuggled into a plan question, caught by its own System 1 question |
| 4 | any | I have crushing chest pain right now. | Route `emergency`: fixed 911 text, no LLM call |
| 5 | any | How do I submit a claim for dental? | Route `procedure`; the claims-submission section |

Close on the audit view (`GET /audit/{id}`): the same probabilities, chunks and rule result the panel showed, stored for every request.

## Recording the GIF

1. Start the API and the UI (see the README), and open the UI at `http://localhost:8501`.
2. Record the browser window with a screen-to-GIF tool (for example ScreenToGif on Windows), at about 1280 px wide and 10–15 fps.
3. Trim the waits for the LLM, and keep the file under 10 MB so GitHub renders it.
4. Save it as `docs/demo.gif` and uncomment the image line near the top of the README.

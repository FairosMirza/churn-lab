# 🔁 Autonomous Churn Experimenter

A one-click AI experiment loop for a **cross-vertical super-app** (rides, food,
groceries, payments) that goes beyond scoring: it **simulates interventions**,
ranks them by revenue protected, and briefs leadership. Built for the Careem
Senior Data Scientist application — one submission covering **all three
optional challenges**:

| Challenge | Where it lives |
|---|---|
| **1 · Churn Predictor** | Trains churn models on a synthetic super-app dataset and maps high-risk users to concrete retention actions |
| **2 · Autonomous Experimenter** | One click: generate data → cross-validate 3 models → holdout leaderboard → churn drivers → **counterfactual what-if simulations** of 7 interventions |
| **3 · Experiment Narrator** | A prompt (visible in the app) turns the metrics JSON into a ≤250-word leadership brief — live via Claude if a key is provided, with a deterministic fallback, plus a **pre-generated Claude brief** embedded so reviewers see the LLM output with zero setup |

## Grounded in publicly-discussed user pain points

The experience features mirror themes super-app users in the region raise
publicly (app-store reviews, social media) — no internal or confidential data:

- **Captain cancellations** after accepting a ride
- **Long pickup wait times**
- **Surge / peak pricing frustration**
- **Slow refunds** to the wallet
- **Late food deliveries**, failed payments, unresolved support tickets

Plus the deliberate cross-vertical story: **users active in 2+ verticals churn
far less than single-vertical users** (~51% churn for zero-vertical users vs
~3% for four-vertical users in the default run) — so the most valuable
retention lever is *creating cross-vertical habits*, exactly the insight a
personalization team can act on.

The dataset is fully synthetic, self-created, and reproducible from a seed.

> **Product grounding:** every pain point, intervention, and benchmark in the
> app is backed by public sources (reviews, news, Careem's own publications) —
> see [RESEARCH.md](RESEARCH.md). Notably, our synthetic non-subscriber churn
> lift (2.8×) independently lands on Careem's publicly reported ~3× retention
> for Plus members.

## What the experiment loop does

1. **Generate** N synthetic users (sidebar-configurable).
2. **Compare models** — Logistic Regression, Random Forest, Gradient Boosting.
   **Selection uses mean 5-fold CV AUC on training data only**; the untouched
   holdout provides the final unbiased estimate (Raschka 2018,
   [arXiv:1811.12808](https://arxiv.org/abs/1811.12808)) — selecting on the
   holdout would bias its estimate ("test-set peeking").
3. **Score honestly** — ROC-AUC, PR-AUC vs the base rate, Brier score, and
   *recall in the riskiest 10%* — the metric a retention team actually cares about.
   What-if interventions carry assumed costs and true **ROI = (revenue protected −
   cost) / cost**, so ROI-negative plays are flagged instead of celebrated.
4. **Explain at three levels (root-cause analysis)** —
   *global*: permutation importance shows what the model relies on;
   *evidence*: churn-rate lifts for users exposed vs not exposed to each pain
   point explain *why the numbers look this way* (including an explicit
   confounding callout where the raw lift is misleading);
   *local*: per-user counterfactual attribution answers "why is **this** user
   at risk?" in plain language.
5. **Recommend** — top-decile-risk users segmented by dominant driver, mapped to
   actions, ranked by annual revenue protected (AED).
6. **Simulate** — the *what-if simulator* applies an intervention (e.g. "halve
   Captain cancellations") and re-scores every user counterfactually:
   churners prevented + revenue protected per intervention.
7. **Brief leadership** — a KPI view (customers at risk, AED revenue at risk,
   best intervention ROI), all interventions ranked, a one-line decision ask,
   and a ready A/B design (hypothesis, primary metric, guardrails) per play.
8. **Narrate** — a structured prompt converts the metrics JSON into an
   executive brief: verdict, key numbers, drivers, the play, caveats, decision ask.

## Run it

**The product (Churn Lab website — FastAPI + hand-crafted frontend):**

```bash
pip install -r requirements.txt
uvicorn server:app --port 8601   # → http://localhost:8601
```

**Or the Streamlit version (same engine, simpler shell):**

```bash
streamlit run app.py
```

Optional: provide an Anthropic API key (input field / `ANTHROPIC_API_KEY`) for a
live Claude brief. Without a key you get a deterministic template — and the
**Example Claude output** tab shows real LLM-quality output either way.
A `Dockerfile` is included for one-click deploys (Hugging Face Spaces, Render).

## Churn Lab — the product tool

`server.py` + `web/` is a working SaaS-style tool, not a report:

- **Experiment runner** — configure users / seed / holdout / models, hit *Run*,
  and the backend executes the real pipeline; every run lands in an in-memory
  **run registry** you can reload and compare (that's the "Autonomous
  Experimenter" made tangible).
- **Role-based workspaces** — Data Scientist (everything), Product (root causes,
  playbook, what-if, narrator), Executive (leadership + narrator).
- **Seven workspaces** — Experiments, Root causes, Playbook, What-if studio,
  Leadership, Narrator, and **Methodology**: the pipeline step-by-step, a
  formula-level glossary of every headline number, honest limitations, and the
  prototype-to-production path (label definition, calibration, serving,
  experimentation) — so the numbers are reproducible, not fiction.

## Files

- `server.py` + `web/` — **Churn Lab**, the product tool (FastAPI backend with a
  run registry; dependency-free HTML/CSS/JS frontend; design informed by
  Amplitude's app grammar, Careem's brand, and a validated chart palette)
- `app.py` — Streamlit version of the same loop
- `generate_data.py` — synthetic dataset generator (runnable as a script to export CSV)
- `experiments.py` — the experiment engine + RCA + what-if counterfactual simulator
- `narrator.py` — the narrator prompt, Claude call, template fallback, and sample brief

## Honest limitations

- The dataset is synthetic; drivers on real data will differ and must be re-estimated.
- Save rates and counterfactual simulations are planning estimates, not A/B
  results — the decision ask in the app is precisely to run those A/B tests.
- No hyperparameter search — the point is the *loop* (compare → explain →
  simulate → decide → narrate), which a search would slot into cleanly.

*Built in a focused session with AI assistance (Claude Code) — which is rather
the point of the exercise.*

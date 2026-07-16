"""Experiment Narrator: turn a metrics JSON into a leadership-ready brief.

The prompt below is the deliverable for the "Experiment Narrator" challenge.
If an Anthropic API key is available the app sends it to Claude; otherwise a
deterministic template produces the same structure so the demo always works.
SAMPLE_LLM_BRIEF shows real Claude output for the default run (seed 42),
pre-generated so reviewers see the LLM experience without needing a key.
"""

import json

NARRATOR_MODEL = "claude-opus-4-8"

NARRATOR_SYSTEM = (
    "You are an ML experiment narrator embedded in a data science team at a "
    "super-app company. You turn raw model metrics into short, decision-ready "
    "briefs for senior leadership - people who decide budgets, not thresholds."
)

NARRATOR_PROMPT_TEMPLATE = """Below is a JSON payload from an automated churn-modeling experiment, including counterfactual what-if simulations of candidate interventions.

Write a leadership brief with exactly these sections:

**VERDICT** - one sentence: which model wins, the single number that justifies it, and whether it is production-ready.
**KEY NUMBERS** - three bullets, each citing one metric from the JSON in plain business language (e.g. "catches 36% of churners in the riskiest 10% of users", never "recall@decile-1 = 0.36").
**ROOT CAUSES** - three bullets explaining WHY churn happens, combining the root_cause_evidence lifts (e.g. "users inactive >21 days churn at 4.2x the rate of others") with the model's driver ranking. If a segment's lift is below 1 despite being a known pain point, call out the likely confounder rather than dismissing it.
**THE PLAY** - the top 2 interventions from the what-if simulations, ranked by annual revenue protected, each with churners prevented and revenue figures. State clearly these are model-based counterfactuals.
**CAVEATS** - two bullets on limitations (synthetic data, assumed save rates, counterfactuals are not A/B results).
**DECISION ASK** - one sentence: the concrete decision you want leadership to make this week.

Rules:
- Every claim must cite a number that appears in the JSON. Never invent metrics.
- If two models are within 0.01 AUC, say so and prefer the simpler/faster one.
- Revenue figures in AED, rounded to the nearest thousand.
- Maximum 250 words. No preamble, start directly with VERDICT.

METRICS JSON:
{metrics_json}"""

# Pre-generated with Claude using the prompt above and the default-run
# metrics (8,000 users, seed 42). Embedded so reviewers see the LLM
# experience with zero setup.
SAMPLE_LLM_BRIEF = """**VERDICT** — Logistic Regression wins with a holdout AUC of 0.780; it's the simplest and fastest of the three models tested and is ready for a shadow-mode pilot, not yet for fully automated targeting.

**KEY NUMBERS**
- The model catches **36% of all churners inside the riskiest 10% of users** — the slice a retention team can realistically reach with offers.
- Precision-recall AUC is **0.463 against a 14.6% base churn rate** — roughly 3× better than untargeted outreach.
- **AED 587K of annualized revenue is at risk** across the 2,000-user holdout alone (≈ AED 2.3M scaled to the full base).

**ROOT CAUSES**
- **Disengagement is the loudest alarm**: users inactive for more than 21 days churn at **49.6% vs 11.8%** for everyone else — a **4.2× lift** — and `days_since_last_activity` is the model's #2 driver.
- **Cross-vertical habit is the strongest protector**: single-vertical users churn at **3.5× the rate** of multi-vertical users (29.6% vs 8.5%), and non-subscribers at **2.8×** — strikingly consistent with Careem's publicly reported ~3× retention for Plus members.
- **One confounder to flag**: surge-exposed users show a raw lift *below* 1 (0.82×) — not because surge helps retention, but because surge exposure correlates with heavy riding, which protects. The model's driver ranking isolates the marginal effect; the A/B pilot is the only clean answer.

**THE PLAY** (model-based counterfactuals, not A/B results)
1. **Halve Captain cancellations** → ~25 churners prevented, **AED 80K/year protected**.
2. **Cross-vertical push** (single-vertical users adopt a 2nd service) → ~46 churners prevented, **AED 78K/year protected**, and it touches only 483 users.

**CAVEATS**
- Data is synthetic; real drivers will differ and must be re-estimated on production data.
- Save rates and counterfactuals are planning estimates — validate with A/B tests before committing budget.

**DECISION ASK** — Approve two 4-week A/B pilots: Captain-cancellation compensation and a cross-vertical voucher for single-vertical users."""


def build_prompt(metrics: dict) -> str:
    return NARRATOR_PROMPT_TEMPLATE.format(metrics_json=json.dumps(metrics, indent=2))


def narrate_with_claude(metrics: dict, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=NARRATOR_MODEL,
        max_tokens=1024,
        system=NARRATOR_SYSTEM,
        messages=[{"role": "user", "content": build_prompt(metrics)}],
    )
    return next(b.text for b in response.content if b.type == "text")


def narrate_with_template(metrics: dict) -> str:
    """Deterministic fallback so the public demo works without any API key."""
    lb = metrics["leaderboard"]
    best = lb[0]
    runner_up = lb[1] if len(lb) > 1 else None
    drivers = metrics["top_churn_drivers"]
    churn_rate = metrics["dataset"]["churn_rate"]
    revenue_at_risk = metrics.get("annual_revenue_at_risk_aed_holdout", 0)
    what_ifs = metrics.get("what_if_simulations_ranked", [])

    close_race = runner_up is not None and abs(best["test_auc"] - runner_up["test_auc"]) < 0.01
    verdict = f"**VERDICT** - {best['model']} wins with a holdout AUC of {best['test_auc']:.3f}"
    if close_race:
        verdict += (
            f"; {runner_up['model']} is within 0.01 AUC ({runner_up['test_auc']:.3f}), "
            "so prefer the simpler model."
        )
    else:
        verdict += ". Ready for a shadow-mode pilot."

    lines = [
        verdict,
        "",
        "**KEY NUMBERS**",
        f"- Catches {best['recall_at_top10pct']:.0%} of all churners inside the riskiest "
        "10% of users - the slice a retention team can realistically target.",
        f"- PR-AUC {best['test_pr_auc']:.3f} against a {churn_rate:.1%} base churn rate - "
        "far better than untargeted outreach.",
        f"- AED {revenue_at_risk:,.0f} of annualized revenue at risk on the holdout alone.",
        "",
        "**ROOT CAUSES**",
        f"- Strongest model driver: `{drivers[0]['feature']}` (importance {drivers[0]['importance']:.3f}).",
        f"- Then `{drivers[1]['feature']}` ({drivers[1]['importance']:.3f}) and "
        f"`{drivers[2]['feature']}` ({drivers[2]['importance']:.3f}).",
    ]
    rca = metrics.get("root_cause_evidence", [])
    if rca:
        top = rca[0]
        lines.append(
            f"- Evidence: \"{top['segment']}\" churn at {top['churn_if_exposed']:.1%} vs "
            f"{top['churn_if_not']:.1%} for others - a {top['churn_lift']:.1f}x lift."
        )
    lines.append("")
    if what_ifs:
        lines.append("**THE PLAY** (model-based counterfactuals, not A/B results)")
        for i, w in enumerate(what_ifs[:2], 1):
            lines.append(
                f"{i}. {w['intervention']} - ~{w['churners_prevented']:.0f} churners prevented, "
                f"AED {w['annual_revenue_protected_aed']:,.0f}/year protected."
            )
        lines.append("")
    lines += [
        "**CAVEATS**",
        "- Data is synthetic (self-created); real-world drivers will differ.",
        "- Save rates and counterfactuals are planning estimates - validate with A/B tests.",
        "",
        "**DECISION ASK** - Approve A/B pilots for the top two interventions above.",
    ]
    return "\n".join(lines)

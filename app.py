"""Autonomous Experimenter - churn prediction for a cross-vertical super-app.

One click runs the full loop: generate data -> cross-validate 3 models ->
evaluate on holdout -> extract churn drivers -> simulate interventions ->
leadership view -> narrate the results (LLM or template).
"""

import json
import os

import streamlit as st

from experiments import (
    EXPERIMENT_DESIGNS,
    WHAT_IF_PRESETS,
    explain_user,
    metrics_payload,
    rca_evidence,
    run_all_what_ifs,
    run_experiments,
    what_if,
)
from generate_data import PAIN_POINT_MAP, TARGET, generate_users
from narrator import (
    NARRATOR_MODEL,
    SAMPLE_LLM_BRIEF,
    build_prompt,
    narrate_with_claude,
    narrate_with_template,
)

st.set_page_config(page_title="Autonomous Churn Experimenter", page_icon="🔁", layout="wide")

st.title("🔁 Autonomous Churn Experimenter")
st.caption(
    "An AI-driven experiment loop for a cross-vertical super-app: it generates data "
    "grounded in publicly-discussed user pain points, trains and compares models, "
    "finds churn drivers, **simulates interventions**, builds a leadership view, and "
    "narrates the results. Covers all three challenges: **Churn Predictor**, "
    "**Autonomous Experimenter**, and **Experiment Narrator**."
)

with st.sidebar:
    st.header("Experiment settings")
    n_users = st.slider("Users to simulate", 2000, 20000, 8000, step=1000)
    seed = st.number_input("Random seed", value=42, step=1)
    st.divider()
    st.subheader("Narrator (optional)")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="If provided, Claude writes the executive brief live. Without it the app "
        "uses a deterministic template - and a pre-generated Claude brief is shown "
        "in the Narrator section either way.",
    ) or os.environ.get("ANTHROPIC_API_KEY", "")
    st.divider()
    st.markdown(
        "**Data note:** fully synthetic, self-created dataset - no confidential or "
        "real user data. Experience features mirror pain points *publicly discussed* "
        "by super-app users (app-store reviews, social media)."
    )

df = generate_users(n_users=n_users, seed=int(seed))

# ---------------------------------------------------------------- 1. Data
st.header("1 · The dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", f"{len(df):,}")
c2.metric("Churn rate", f"{df[TARGET].mean():.1%}")
c3.metric("Multi-vertical users", f"{(df['verticals_active'] >= 2).mean():.0%}")
c4.metric("Avg monthly spend", f"AED {df['monthly_spend_aed'].mean():,.0f}")

with st.expander("Preview data, pain-point grounding & the cross-vertical effect"):
    st.markdown(
        "**Grounding:** the experience features simulate pain points users of "
        "regional super-apps raise publicly:"
    )
    st.table(
        {"Feature": list(PAIN_POINT_MAP.keys()),
         "Publicly-discussed pain point": list(PAIN_POINT_MAP.values())}
    )
    st.markdown(
        "**Public evidence behind the design** (no internal data used): "
        "Captain cancellations & long waits and refund/billing frustration are the top themes in "
        "[Trustpilot](https://www.trustpilot.com/review/careem.com) and "
        "[PissedConsumer](https://careem.pissedconsumer.com/review.html) reviews; "
        "peak-pricing frustration is documented by "
        "[MENAbytes](https://www.menabytes.com/careem-peak-pricing-saudi-customers-not-happy/) and "
        "[Lovin Dubai](https://lovin.co/dubai/en/news/careem-uber-taxis-rain); "
        "Careem publicly reports that **Plus subscribers retain ~3x and use ~2x as many services** "
        "([Everything App strategy](https://why.careem.com/en/building-the-everything-app/)) - "
        "the cross-vertical effect this dataset is built around."
    )
    st.dataframe(df.head(10), width="stretch")
    st.markdown("**Churn rate by number of active verticals** - the habit-stacking effect:")
    st.bar_chart(df.groupby("verticals_active")[TARGET].mean())

# ---------------------------------------------------------- 2. Experiments
st.header("2 · Run the autonomous experiment loop")
st.markdown(
    "Trains **Logistic Regression, Random Forest and Gradient Boosting** with "
    "5-fold cross-validation, evaluates on a 25% holdout, picks a winner, and "
    "pre-computes counterfactual intervention simulations."
)

if st.button("▶ Run experiments", type="primary"):
    with st.spinner("Running cross-validated experiments and what-if simulations..."):
        results = run_experiments(df, seed=int(seed))
        st.session_state["results"] = results
        st.session_state["what_ifs"] = run_all_what_ifs(results)

results = st.session_state.get("results")

if results:
    what_ifs = st.session_state["what_ifs"]

    st.subheader("Model leaderboard")
    lb = results["leaderboard"].copy()
    lb.index = lb.index + 1
    st.dataframe(
        lb.style.format(
            {
                "cv_auc_mean": "{:.3f}", "cv_auc_std": "{:.3f}", "cv_pr_auc_mean": "{:.3f}",
                "test_auc": "{:.3f}", "test_pr_auc": "{:.3f}",
                "recall_at_top10pct": "{:.1%}", "brier_score": "{:.3f}",
                "train_seconds": "{:.1f}s",
            }
        ).highlight_max(subset=["cv_auc_mean"], color="#1a7f37"),
        width="stretch",
    )
    best = lb.iloc[0]
    st.success(
        f"**Winner: {results['best_model']}** - selected by mean CV AUC {best['cv_auc_mean']:.3f}; "
        f"unbiased holdout AUC {best['test_auc']:.3f}, catching "
        f"{best['recall_at_top10pct']:.0%} of churners in the riskiest 10% of users."
    )

    # ------------------------------------------------------ 3. Root cause
    st.header("3 · Root-cause analysis: why users churn")
    tab_global, tab_evidence, tab_local = st.tabs(
        ["Model drivers (global)", "Evidence: pain-point churn lifts", "Explain one user (local)"]
    )

    with tab_global:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("**Churn drivers** (permutation importance, holdout AUC drop):")
            st.bar_chart(results["importance"].set_index("feature")["importance"], horizontal=True)
        with col_b:
            st.markdown("**Highest-risk users** (top of the holdout risk ranking):")
            st.dataframe(
                results["at_risk_sample"][
                    ["churn_risk", "driver", "verticals_active",
                     "captain_cancellations_90d", "days_since_last_activity"]
                ].style.format({"churn_risk": "{:.0%}"}),
                width="stretch", height=280,
            )

    with tab_evidence:
        st.markdown(
            "**Why are we seeing these numbers?** Churn rate for users *exposed* to each "
            "pain point vs everyone else. Observational evidence, not causality:"
        )
        rca = rca_evidence(df)
        st.dataframe(
            rca.style.format(
                {"share_of_users": "{:.0%}", "churn_if_exposed": "{:.1%}",
                 "churn_if_not": "{:.1%}", "churn_lift": "{:.1f}x"}
            ).bar(subset=["churn_lift"], color="#e07a5f"),
            width="stretch",
        )
        non_sub = rca[rca["segment"] == "Not a subscriber"]
        if len(non_sub):
            st.success(
                f"**External validation:** non-subscribers churn at "
                f"{non_sub.iloc[0]['churn_lift']:.1f}x here - strikingly consistent with "
                "Careem's publicly reported ~3x retention for Careem Plus members."
            )
        surge = rca[rca["segment"] == "Surge on >35% of rides"]
        if len(surge) and surge.iloc[0]["churn_lift"] < 1:
            st.warning(
                "**Confounding alert** - surge-exposed users show a lift *below* 1 "
                f"({surge.iloc[0]['churn_lift']:.1f}x). That's not because surge helps "
                "retention: surge exposure correlates with heavy riding, and heavy riders "
                "churn less. The model's driver ranking isolates the marginal effect; an "
                "A/B test is the only clean answer. This is exactly why raw comparisons "
                "and model attribution are shown side by side."
            )

    with tab_local:
        st.markdown(
            "**Why is *this* user at risk?** Counterfactual attribution: each feature is "
            "reset to the typical (median) value and the risk change is measured."
        )
        top_users = results["at_risk_sample"]
        options = {
            f"User #{idx} - {row['churn_risk']:.0%} risk ({row['driver']})": int(idx)
            for idx, row in top_users.iterrows()
        }
        choice = st.selectbox("Pick a high-risk user", list(options.keys()))
        base_risk, contrib = explain_user(results, options[choice])
        lc, rc = st.columns([1, 1])
        with lc:
            st.metric("Predicted churn risk", f"{base_risk:.0%}")
            top3 = contrib.head(3)
            reasons = "; ".join(
                f"**{r['feature']}** = {r['user_value']:g} (typical {r['typical_value']:g}) "
                f"adds {r['risk_contribution']:+.0%}"
                for _, r in top3.iterrows()
            )
            st.markdown(f"Main reasons vs a typical user: {reasons}.")
            st.caption(
                "Positive = pushes risk up vs a typical user. Attribution is "
                "model-based, one feature at a time (no interactions)."
            )
        with rc:
            st.bar_chart(contrib.set_index("feature")["risk_contribution"], horizontal=True)

    # ------------------------------------------------------ 4. Playbook
    st.header("4 · Retention playbook")
    st.markdown(
        "Top-decile-risk users, segmented into **retention segments** (rule-based action mapping) and mapped to an "
        "action, ranked by annual revenue protected. Save rates are planning assumptions."
    )
    st.dataframe(
        results["segments"].drop(columns=["monthly_spend_aed"]).style.format(
            {"avg_risk": "{:.0%}", "assumed_save_rate": "{:.0%}",
             "annual_revenue_protected_aed": "AED {:,.0f}"}
        ),
        width="stretch",
    )

    # ------------------------------------------------------ 5. What-if
    st.header("5 · What-if simulator")
    st.markdown(
        "Pick an intervention; the winning model re-scores **every holdout user** under "
        "the counterfactual and measures the impact. This is where the experimenter "
        "becomes a decision tool, not just a scorer."
    )
    preset = st.selectbox("Intervention", list(WHAT_IF_PRESETS.keys()))
    sim = what_if(results, preset)
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Users affected", f"{sim['users_affected']:,}")
    w2.metric(
        "Predicted churn rate",
        f"{sim['new_churn_rate']:.1%}",
        delta=f"{sim['new_churn_rate'] - sim['baseline_churn_rate']:+.2%}",
        delta_color="inverse",
    )
    w3.metric("Churners prevented", f"{sim['churners_prevented']:.0f}")
    w4.metric("Revenue protected / yr", f"AED {sim['annual_revenue_protected_aed']:,.0f}")
    st.caption(
        "Model-based counterfactual on the holdout set - a prioritisation signal, "
        "not an A/B result."
    )

    # ------------------------------------------------------ 6. Leadership
    st.header("6 · Leadership view")
    top_play = what_ifs.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers at risk (top decile)", f"{int(results['segments']['users'].sum()):,}")
    k2.metric("Annual revenue at risk (holdout)", f"AED {results['revenue_at_risk_annual_aed']:,.0f}")
    k3.metric("#1 churn driver", results["importance"].iloc[0]["feature"])
    k4.metric("Highest est. annual upside", f"AED {top_play['annual_revenue_protected_aed']:,.0f}/yr")

    st.markdown("**All interventions, ranked by simulated annual revenue protected:**")
    st.dataframe(
        what_ifs.style.format(
            {"baseline_churn_rate": "{:.1%}", "new_churn_rate": "{:.1%}",
             "churners_prevented": "{:.0f}", "users_affected": "{:,.0f}",
             "annual_revenue_protected_aed": "AED {:,.0f}"}
        ).highlight_max(subset=["annual_revenue_protected_aed"], color="#1a7f37"),
        width="stretch",
    )
    st.info(
        f"**The ask:** approve 4-week A/B pilots for the top two plays - "
        f"**{what_ifs.iloc[0]['intervention']}** and **{what_ifs.iloc[1]['intervention']}** - "
        f"with a combined simulated upside of AED "
        f"{what_ifs.head(2)['annual_revenue_protected_aed'].sum():,.0f}/year on the holdout alone."
    )
    with st.expander("How we'd actually test the top plays (A/B design)"):
        for i in range(2):
            play = what_ifs.iloc[i]["intervention"]
            design = EXPERIMENT_DESIGNS.get(play)
            if design:
                st.markdown(
                    f"**{i + 1}. {play}**\n"
                    f"- *Hypothesis:* {design['hypothesis']}\n"
                    f"- *Primary metric:* {design['primary_metric']}\n"
                    f"- *Guardrails:* {design['guardrail']}\n"
                    f"- *Setup:* 50/50 randomised among affected users, 4 weeks, "
                    f"sized on the simulated effect above."
                )

    # ------------------------------------------------------ 7. Narrator
    st.header("7 · Experiment Narrator")
    metrics = metrics_payload(results, what_ifs, rca_evidence(df))

    tab_live, tab_example, tab_prompt = st.tabs(
        ["Live narrator", "Example Claude output (pre-generated)", "The prompt (Challenge 3)"]
    )

    with tab_live:
        if st.button("✍️ Narrate this run"):
            if api_key:
                try:
                    with st.spinner(f"Asking {NARRATOR_MODEL}..."):
                        st.session_state["brief"] = narrate_with_claude(metrics, api_key)
                    st.session_state["brief_source"] = f"Claude ({NARRATOR_MODEL})"
                except Exception as e:
                    st.warning(f"Claude call failed ({e}); using the template narrator instead.")
                    st.session_state["brief"] = narrate_with_template(metrics)
                    st.session_state["brief_source"] = "deterministic template"
            else:
                st.session_state["brief"] = narrate_with_template(metrics)
                st.session_state["brief_source"] = "deterministic template (no API key provided)"
        if "brief" in st.session_state:
            st.markdown(f"*Generated by: {st.session_state['brief_source']}*")
            st.markdown(st.session_state["brief"])
        else:
            st.caption("Click to narrate the current run. Without an API key you get the "
                       "deterministic template; the next tab shows real LLM-quality output.")

    with tab_example:
        st.caption(
            "Pre-generated with Claude using this exact prompt on the default run "
            "(8,000 users, seed 42) - so you can see the LLM experience without an API key."
        )
        st.markdown(SAMPLE_LLM_BRIEF)

    with tab_prompt:
        st.code(build_prompt(metrics), language="text")
        with st.expander("Raw metrics JSON handed to the narrator"):
            st.json(json.loads(json.dumps(metrics, default=str)))
else:
    st.info("Click **Run experiments** to start the loop.")

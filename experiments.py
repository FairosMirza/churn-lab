"""Autonomous experiment engine: train, compare, interpret and *simulate*.

One call to run_experiments() executes the whole loop -- split, cross-validate
several models, evaluate the best on a holdout, extract churn drivers, map
high-risk users to retention actions, and quantify revenue at risk.

what_if() then uses the winning model as a counterfactual simulator: apply an
intervention to the feature matrix, re-score every user, and measure churners
prevented and revenue protected.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import FEATURES, TARGET

MODELS = {
    "Logistic Regression": lambda seed: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    ),
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed
    ),
    "Gradient Boosting": lambda seed: HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, random_state=seed
    ),
}

# Rule-based mapping from a user's dominant churn driver to a retention action.
# Drivers mirror pain points publicly discussed by super-app users in the
# region. Assumed save rates are planning numbers, not measured outcomes.
ACTIONS = [
    ("Wallet refund friction", "Instant-refund guarantee + apology wallet credit", 0.30),
    ("Payment friction", "Payment-fix flow + automatic retry with wallet balance", 0.30),
    ("Captain cancellations", "Priority re-matching + automatic cancellation compensation", 0.25),
    ("Service recovery", "Priority support callback + goodwill credit", 0.25),
    ("Long pickup waits", "Priority dispatch pilot in the user's area + accurate ETAs", 0.18),
    ("Late deliveries", "Delivery-time guarantee: auto-credit if the order is late", 0.20),
    ("Single-vertical habit", "Cross-vertical voucher (e.g. food credit for rides-only users)", 0.20),
    ("Surge sensitivity", "Fare-lock passes / Careem Plus upsell to cap price shocks", 0.15),
    ("Promo dependency", "Shift to loyalty points instead of discounts", 0.12),
    ("Gone quiet", "Win-back push notification + small ride credit", 0.15),
    ("Low engagement", "Personalised engagement nudge", 0.08),
]

# Root-cause evidence: exposure definitions used to compare churn rates for
# users affected vs not affected by each pain point ("why these numbers").
RCA_SEGMENTS = [
    ("Single-vertical users (<=1 active vertical)", lambda d: d["verticals_active"] <= 1),
    ("Not a subscriber", lambda d: d["careem_plus"] == 0),
    (">=2 Captain cancellations in 90d", lambda d: d["captain_cancellations_90d"] >= 2),
    ("Avg pickup wait > 12 min", lambda d: d["avg_wait_time_min"] > 12),
    ("Surge on >35% of rides", lambda d: d["surge_exposure_share"] > 0.35),
    (">=1 delayed wallet refund in 90d", lambda d: d["refund_delays_90d"] >= 1),
    (">=2 late deliveries in 90d", lambda d: d["delivery_delays_90d"] >= 2),
    (">=1 failed payment in 90d", lambda d: d["failed_payments_90d"] >= 1),
    (">=2 support tickets in 90d", lambda d: d["support_tickets_90d"] >= 2),
    ("Inactive > 21 days", lambda d: d["days_since_last_activity"] > 21),
    ("Promo on >60% of orders", lambda d: d["promo_share"] > 0.6),
]

# PM layer: how each top intervention would actually be tested.
EXPERIMENT_DESIGNS = {
    "Halve Captain cancellations": {
        "hypothesis": "Auto-compensating cancelled rides and priority re-matching reduces churn among affected riders",
        "primary_metric": "60-day churn rate of riders with a cancelled ride",
        "guardrail": "Captain acceptance rate; compensation cost per ride",
    },
    "Cross-vertical push: single-vertical users adopt a 2nd vertical": {
        "hypothesis": "A targeted second-vertical voucher creates a habit that lifts retention",
        "primary_metric": "2nd-vertical activation within 30 days; 90-day churn",
        "guardrail": "Voucher cost per activation; cannibalisation of organic cross-sell",
    },
    "Instant wallet refunds (no refund delays)": {
        "hypothesis": "Instant refunds to the wallet remove a trust-breaking moment",
        "primary_metric": "60-day churn of users who requested a refund",
        "guardrail": "Refund fraud rate; wallet float cost",
    },
    "Eliminate failed payments": {
        "hypothesis": "Auto-retry with wallet balance rescues failed checkouts",
        "primary_metric": "Checkout completion rate; 60-day churn of affected users",
        "guardrail": "Involuntary wallet usage complaints",
    },
    "Halve surge exposure": {
        "hypothesis": "Fare-lock passes / subscription upsell cap price shocks for price-sensitive users",
        "primary_metric": "Ride frequency of surge-exposed users; churn",
        "guardrail": "Marketplace balance (driver supply at peak); revenue per ride",
    },
    "Cut pickup wait times by 25%": {
        "hypothesis": "Better dispatch and honest ETAs keep the ride habit alive",
        "primary_metric": "Repeat-ride rate within 14 days of a long-wait ride",
        "guardrail": "Captain idle time / utilisation",
    },
    "On-time deliveries (no delivery delays)": {
        "hypothesis": "A delivery-time guarantee with auto-credit restores trust after a late order",
        "primary_metric": "Next-order rate within 30 days of a late delivery",
        "guardrail": "Credit cost per order; restaurant partner SLA disputes",
    },
}

# What-if intervention presets: label -> function(features_df) -> modified copy.
WHAT_IF_PRESETS = {
    "Halve Captain cancellations": lambda X: X.assign(
        captain_cancellations_90d=(X["captain_cancellations_90d"] * 0.5).round()
    ),
    "Cut pickup wait times by 25%": lambda X: X.assign(
        avg_wait_time_min=(X["avg_wait_time_min"] * 0.75).round(1)
    ),
    "Instant wallet refunds (no refund delays)": lambda X: X.assign(refund_delays_90d=0),
    "Eliminate failed payments": lambda X: X.assign(failed_payments_90d=0),
    "Cross-vertical push: single-vertical users adopt a 2nd vertical": lambda X: X.assign(
        verticals_active=np.where(
            X["verticals_active"] == 1, 2, X["verticals_active"]
        )
    ),
    "Halve surge exposure": lambda X: X.assign(
        surge_exposure_share=(X["surge_exposure_share"] * 0.5).round(2)
    ),
    "On-time deliveries (no delivery delays)": lambda X: X.assign(delivery_delays_90d=0),
}


def _dominant_driver(row) -> str:
    if row["refund_delays_90d"] >= 1:
        return "Wallet refund friction"
    if row["failed_payments_90d"] >= 1:
        return "Payment friction"
    if row["captain_cancellations_90d"] >= 2:
        return "Captain cancellations"
    if row["support_tickets_90d"] >= 2:
        return "Service recovery"
    if row["avg_wait_time_min"] > 12:
        return "Long pickup waits"
    if row["delivery_delays_90d"] >= 2:
        return "Late deliveries"
    if row["verticals_active"] <= 1 and row["tenure_months"] >= 3:
        return "Single-vertical habit"
    if row["surge_exposure_share"] > 0.35:
        return "Surge sensitivity"
    if row["promo_share"] > 0.6:
        return "Promo dependency"
    if row["days_since_last_activity"] > 21:
        return "Gone quiet"
    return "Low engagement"


def run_experiments(
    df: pd.DataFrame, seed: int = 42, test_size: float = 0.25, models: list = None
) -> dict:
    """Run the full experiment loop and return everything as one results dict."""
    chosen = {k: v for k, v in MODELS.items() if not models or k in models}
    if not chosen:
        chosen = MODELS

    X, y = df[FEATURES], df[TARGET]
    spend = df["monthly_spend_aed"]
    X_train, X_test, y_train, y_test, _, spend_test = train_test_split(
        X, y, spend, test_size=test_size, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    leaderboard_rows = []
    fitted = {}

    for name, build in chosen.items():
        model = build(seed)
        t0 = time.time()
        scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision"], n_jobs=-1,
        )
        model.fit(X_train, y_train)
        fitted[name] = model

        proba = model.predict_proba(X_test)[:, 1]
        top_decile = proba >= np.quantile(proba, 0.9)
        churners_captured = y_test[top_decile].sum() / max(y_test.sum(), 1)

        leaderboard_rows.append({
            "model": name,
            "cv_auc_mean": scores["test_roc_auc"].mean(),
            "cv_auc_std": scores["test_roc_auc"].std(),
            "cv_pr_auc_mean": scores["test_average_precision"].mean(),
            "test_auc": roc_auc_score(y_test, proba),
            "test_pr_auc": average_precision_score(y_test, proba),
            "recall_at_top10pct": churners_captured,
            "brier_score": brier_score_loss(y_test, proba),
            "train_seconds": time.time() - t0,
        })

    leaderboard = (
        pd.DataFrame(leaderboard_rows)
        .sort_values("test_auc", ascending=False)
        .reset_index(drop=True)
    )
    best_name = leaderboard.loc[0, "model"]
    best_model = fitted[best_name]

    # Churn drivers via permutation importance on the holdout set.
    perm = permutation_importance(
        best_model, X_test, y_test, scoring="roc_auc",
        n_repeats=5, random_state=seed, n_jobs=-1,
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Retention actions for the top-decile-risk users (scored on holdout).
    # Positional index so rows line up with X_holdout for local explanations.
    scored = X_test.reset_index(drop=True).copy()
    scored["churn_risk"] = best_model.predict_proba(X_test)[:, 1]
    scored["monthly_spend_aed"] = spend_test.values
    at_risk = scored[scored["churn_risk"] >= scored["churn_risk"].quantile(0.9)].copy()
    at_risk["driver"] = at_risk.apply(_dominant_driver, axis=1)

    action_lookup = {driver: (action, save) for driver, action, save in ACTIONS}
    seg = (
        at_risk.groupby("driver")
        .agg(
            users=("churn_risk", "size"),
            avg_risk=("churn_risk", "mean"),
            monthly_spend_aed=("monthly_spend_aed", "sum"),
        )
        .reset_index()
    )
    seg["recommended_action"] = seg["driver"].map(lambda d: action_lookup[d][0])
    seg["assumed_save_rate"] = seg["driver"].map(lambda d: action_lookup[d][1])
    seg["est_users_saved"] = (seg["users"] * seg["assumed_save_rate"]).round(0).astype(int)
    seg["annual_revenue_protected_aed"] = (
        seg["monthly_spend_aed"] * seg["assumed_save_rate"] * 12
    ).round(0)
    segments = seg.sort_values("annual_revenue_protected_aed", ascending=False).reset_index(drop=True)

    # Expected annual revenue at risk across the whole holdout: risk x spend x 12.
    revenue_at_risk = float((scored["churn_risk"] * scored["monthly_spend_aed"]).sum() * 12)

    return {
        "leaderboard": leaderboard,
        "best_model": best_name,
        "best_model_obj": best_model,
        "importance": importance,
        "segments": segments,
        "at_risk_sample": at_risk.sort_values("churn_risk", ascending=False).head(15),
        "X_holdout": X_test.reset_index(drop=True),
        "spend_holdout": spend_test.reset_index(drop=True),
        "revenue_at_risk_annual_aed": revenue_at_risk,
        "meta": {
            "n_users": len(df),
            "churn_rate": float(y.mean()),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_folds": 5,
            "seed": seed,
        },
    }


def rca_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Global root-cause evidence: churn rate for users exposed to each pain
    point vs everyone else, and the relative lift. This is the 'why are we
    seeing these numbers' table -- observational evidence, not causality."""
    rows = []
    for label, condition in RCA_SEGMENTS:
        mask = condition(df)
        if mask.sum() == 0 or (~mask).sum() == 0:
            continue
        exposed = df.loc[mask, TARGET].mean()
        not_exposed = df.loc[~mask, TARGET].mean()
        rows.append({
            "segment": label,
            "share_of_users": mask.mean(),
            "churn_if_exposed": exposed,
            "churn_if_not": not_exposed,
            "churn_lift": exposed / max(not_exposed, 1e-9),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("churn_lift", ascending=False)
        .reset_index(drop=True)
    )


def explain_user(results: dict, row_idx: int, top_k: int = 8):
    """Local explanation: why is THIS user at risk?

    Model-agnostic counterfactual attribution: for each feature, replace the
    user's value with the holdout median and measure how much their predicted
    risk drops. A positive delta means the feature is pushing risk UP relative
    to a typical user."""
    model = results["best_model_obj"]
    X = results["X_holdout"]
    x = X.iloc[[row_idx]]
    base_risk = float(model.predict_proba(x)[0, 1])
    medians = X.median()

    contribs = []
    for feat in FEATURES:
        if x[feat].iloc[0] == medians[feat]:
            continue
        x_cf = x.copy()
        x_cf[feat] = medians[feat]
        cf_risk = float(model.predict_proba(x_cf)[0, 1])
        contribs.append({
            "feature": feat,
            "user_value": float(x[feat].iloc[0]),
            "typical_value": float(medians[feat]),
            "risk_contribution": base_risk - cf_risk,
        })
    table = (
        pd.DataFrame(contribs)
        .sort_values("risk_contribution", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return base_risk, table


def what_if(results: dict, preset_name: str) -> dict:
    """Counterfactual simulation: apply one intervention, re-score all users."""
    model = results["best_model_obj"]
    X = results["X_holdout"]
    spend = results["spend_holdout"]

    base_p = model.predict_proba(X)[:, 1]
    X_new = WHAT_IF_PRESETS[preset_name](X.copy())
    new_p = model.predict_proba(X_new)[:, 1]

    users_affected = int((X_new != X).any(axis=1).sum())
    return {
        "intervention": preset_name,
        "users_affected": users_affected,
        "baseline_churn_rate": float(base_p.mean()),
        "new_churn_rate": float(new_p.mean()),
        "churners_prevented": float(np.maximum(base_p - new_p, 0).sum()),
        "annual_revenue_protected_aed": float(
            (np.maximum(base_p - new_p, 0) * spend).sum() * 12
        ),
    }


def run_all_what_ifs(results: dict) -> pd.DataFrame:
    """Simulate every preset and rank by revenue protected (leadership view)."""
    rows = [what_if(results, name) for name in WHAT_IF_PRESETS]
    return (
        pd.DataFrame(rows)
        .sort_values("annual_revenue_protected_aed", ascending=False)
        .reset_index(drop=True)
    )


def metrics_payload(results: dict, what_ifs: pd.DataFrame = None, rca: pd.DataFrame = None) -> dict:
    """Compact JSON-safe payload of the experiment results, fed to the narrator."""
    lb = results["leaderboard"].round(4)
    payload = {
        "experiment": "churn prediction on synthetic cross-vertical super-app data",
        "dataset": results["meta"],
        "leaderboard": lb.to_dict(orient="records"),
        "best_model": results["best_model"],
        "top_churn_drivers": results["importance"].head(6).round(4).to_dict(orient="records"),
        "retention_segments": results["segments"]
        .drop(columns=["monthly_spend_aed"])
        .round(3)
        .to_dict(orient="records"),
        "annual_revenue_at_risk_aed_holdout": round(results["revenue_at_risk_annual_aed"]),
    }
    if what_ifs is not None:
        payload["what_if_simulations_ranked"] = what_ifs.round(3).to_dict(orient="records")
    if rca is not None:
        payload["root_cause_evidence"] = rca.round(3).to_dict(orient="records")
    return payload

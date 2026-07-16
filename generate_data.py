"""Synthetic cross-vertical super-app churn dataset.

Simulates a Careem-like Everything App: users move around (rides), order food
and groceries (Quik), and pay with the wallet. Churn is driven by plausible
behavioural patterns -- most importantly, users active in multiple verticals
churn far less than single-vertical users.

The experience features are grounded in pain points *publicly discussed* by
super-app / ride-hailing users in the region (app-store reviews, social media):
Captain (driver) cancellations, long pickup waits, surge-price sensitivity,
wallet refund delays, and late food deliveries.

Fully synthetic and self-created: no confidential or real user data.
"""

import numpy as np
import pandas as pd

FEATURES = [
    # engagement & relationship
    "tenure_months",
    "rides_per_month",
    "food_orders_per_month",
    "quik_orders_per_month",
    "pay_txns_per_month",
    "verticals_active",
    "careem_plus",
    "promo_share",
    "days_since_last_activity",
    # experience pain points (publicly-discussed themes)
    "captain_cancellations_90d",
    "avg_wait_time_min",
    "surge_exposure_share",
    "refund_delays_90d",
    "delivery_delays_90d",
    "failed_payments_90d",
    "support_tickets_90d",
]

TARGET = "churned"

# feature -> the publicly-discussed pain point it represents
PAIN_POINT_MAP = {
    "captain_cancellations_90d": "Captain cancels after accepting the ride",
    "avg_wait_time_min": "Long pickup wait times",
    "surge_exposure_share": "Surge / peak pricing frustration",
    "refund_delays_90d": "Slow refunds to the Careem Pay wallet",
    "delivery_delays_90d": "Late or delayed food deliveries",
    "failed_payments_90d": "Card / payment failures at checkout",
    "support_tickets_90d": "Unresolved support issues",
}


def generate_users(n_users: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(2.2, 9.0, n_users), 1, 72).round(1)

    # Zero-inflated activity per vertical: many users never touch a vertical.
    def vertical_usage(p_active, mean):
        active = rng.random(n_users) < p_active
        usage = rng.gamma(1.6, mean / 1.6, n_users) * active
        return np.round(usage, 1)

    rides = vertical_usage(0.85, 6.0)
    food = vertical_usage(0.55, 4.0)
    quik = vertical_usage(0.30, 3.0)
    pay = vertical_usage(0.40, 8.0)

    verticals_active = (
        (rides > 0.5).astype(int)
        + (food > 0.5).astype(int)
        + (quik > 0.5).astype(int)
        + (pay > 0.5).astype(int)
    )

    total_activity = rides + food + quik + 0.5 * pay

    # Subscribers skew toward heavier, multi-vertical users.
    plus_logit = -2.5 + 0.35 * verticals_active + 0.04 * total_activity
    careem_plus = (rng.random(n_users) < 1 / (1 + np.exp(-plus_logit))).astype(int)

    promo_share = np.clip(rng.beta(1.6, 3.5, n_users), 0, 1).round(2)
    support_tickets = rng.poisson(0.35, n_users)
    failed_payments = rng.poisson(0.12, n_users)

    # --- Experience pain points (publicly-discussed themes) ---
    is_rider = rides > 0.5
    # exposure scales with usage: more rides -> more chances of a bad ride
    captain_cancels = rng.poisson(0.25 + 0.06 * rides) * is_rider
    avg_wait = np.where(
        is_rider, np.clip(rng.gamma(3.0, 2.4, n_users), 1, 30), 0.0
    ).round(1)
    surge_share = np.where(is_rider, np.clip(rng.beta(1.5, 6.0, n_users), 0, 1), 0.0).round(2)
    refund_delays = rng.poisson(0.08, n_users)
    delivery_delays = rng.poisson(0.10 + 0.06 * food)

    # Inactive-days is inversely related to activity, with noise.
    base_gap = 30 / (1 + 0.4 * total_activity)
    days_inactive = np.clip(rng.gamma(2.0, base_gap / 2.0, n_users), 0, 90).round(0)

    # Monthly revenue proxy (AED) -- business column, not a model feature.
    monthly_spend = np.round(
        rides * 22 + food * 38 + quik * 30 + pay * 6 + rng.gamma(2, 8, n_users), 0
    )

    # Churn = no activity in the following 60 days. The latent drivers:
    logit = (
        -1.5
        + 0.05 * days_inactive           # disengagement is the loudest signal
        - 0.72 * verticals_active        # cross-vertical habit protects hard
        - 0.85 * careem_plus             # subscribers stay
        + 1.0 * promo_share              # promo-chasers leave when promos stop
        + 0.30 * support_tickets         # unresolved issues push people out
        + 0.50 * failed_payments         # payment friction
        + 0.30 * np.minimum(captain_cancels, 4)  # cancelled rides burn trust
        + 0.030 * avg_wait               # long waits erode the habit
        + 0.80 * surge_share             # price shock frustration
        + 0.55 * refund_delays           # money stuck in the wallet
        + 0.25 * np.minimum(delivery_delays, 4)  # cold food, cold loyalty
        - 0.012 * tenure                 # long-tenured users are stickier
        - 0.03 * total_activity
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churned = (rng.random(n_users) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "user_id": [f"u_{i:06d}" for i in range(n_users)],
            "tenure_months": tenure,
            "rides_per_month": rides,
            "food_orders_per_month": food,
            "quik_orders_per_month": quik,
            "pay_txns_per_month": pay,
            "verticals_active": verticals_active,
            "careem_plus": careem_plus,
            "promo_share": promo_share,
            "days_since_last_activity": days_inactive,
            "captain_cancellations_90d": captain_cancels,
            "avg_wait_time_min": avg_wait,
            "surge_exposure_share": surge_share,
            "refund_delays_90d": refund_delays,
            "delivery_delays_90d": delivery_delays,
            "failed_payments_90d": failed_payments,
            "support_tickets_90d": support_tickets,
            "monthly_spend_aed": monthly_spend,
            TARGET: churned,
        }
    )


if __name__ == "__main__":
    df = generate_users()
    df.to_csv("superapp_churn.csv", index=False)
    print(f"Wrote superapp_churn.csv: {len(df)} users, churn rate {df[TARGET].mean():.1%}")

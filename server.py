"""Churn Lab: FastAPI backend for the Autonomous Churn Experimenter tool.

Every experiment run executes the real pipeline (simulate -> cross-validate ->
holdout -> RCA -> counterfactual what-ifs -> narration inputs) and is kept in
an in-memory run registry, so the product behaves like a tool: configure a
run, execute it, compare it against history.

Run locally:  uvicorn server:app --port 8601
"""

import itertools
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from experiments import (
    EXPERIMENT_DESIGNS,
    MODELS,
    explain_user,
    metrics_payload,
    rca_evidence,
    run_all_what_ifs,
    run_experiments,
)
from generate_data import PAIN_POINT_MAP, TARGET, generate_users
from narrator import (
    NARRATOR_MODEL,
    SAMPLE_LLM_BRIEF,
    build_prompt,
    narrate_with_claude,
    narrate_with_template,
)

RUNS = {}          # run_id -> {"bootstrap": ..., "metrics": ...}
RUN_ORDER = []     # newest last
_run_counter = itertools.count(1)

MAX_KEPT_RUNS = 20


def execute_run(n_users: int, seed: int, test_size: float, models: list) -> str:
    """The autonomous loop, end to end. Returns the new run id."""
    df = generate_users(n_users=n_users, seed=seed)
    results = run_experiments(df, seed=seed, test_size=test_size, models=models)
    what_ifs = run_all_what_ifs(results)
    rca = rca_evidence(df)
    metrics = metrics_payload(results, what_ifs, rca)

    explanations = []
    for idx, row in results["at_risk_sample"].iterrows():
        base_risk, table = explain_user(results, int(idx), top_k=7)
        explanations.append({
            "user": f"User #{int(idx)}",
            "risk": float(base_risk),
            "driver": row["driver"],
            "contributions": table.round(4).to_dict(orient="records"),
        })

    wi = what_ifs.round(4).to_dict(orient="records")
    for row in wi:
        row["design"] = EXPERIMENT_DESIGNS.get(row["intervention"])

    run_id = f"run-{next(_run_counter):03d}"
    lb = results["leaderboard"].round(4)
    best = lb.iloc[0]

    bootstrap = {
        "run": {
            "id": run_id,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "params": {
                "n_users": n_users,
                "seed": seed,
                "test_size": test_size,
                "models": list(results["leaderboard"]["model"]),
            },
        },
        "meta": {
            "n_users": n_users,
            "seed": seed,
            "churn_rate": float(df[TARGET].mean()),
            "multi_vertical_share": float((df["verticals_active"] >= 2).mean()),
            "avg_monthly_spend_aed": float(df["monthly_spend_aed"].mean()),
            "n_holdout": results["meta"]["n_test"],
        },
        "leaderboard": lb.to_dict(orient="records"),
        "best_model": results["best_model"],
        "churn_by_verticals": {
            str(k): float(v)
            for k, v in df.groupby("verticals_active")[TARGET].mean().items()
        },
        "importance": results["importance"].round(4).to_dict(orient="records"),
        "rca": rca.round(4).to_dict(orient="records"),
        "segments": results["segments"]
        .drop(columns=["monthly_spend_aed"])
        .round(4)
        .to_dict(orient="records"),
        "what_ifs": wi,
        "explanations": explanations,
        "leadership": {
            "customers_at_risk": int(results["segments"]["users"].sum()),
            "annual_revenue_at_risk_aed": float(results["revenue_at_risk_annual_aed"]),
            "top_driver": results["importance"].iloc[0]["feature"],
            "best_play": wi[0]["intervention"],
            "best_play_revenue": wi[0]["annual_revenue_protected_aed"],
            "top2_combined_revenue": wi[0]["annual_revenue_protected_aed"]
            + wi[1]["annual_revenue_protected_aed"],
        },
        "narrative": {
            "template_brief": narrate_with_template(metrics),
            "sample_llm_brief": SAMPLE_LLM_BRIEF,
            "prompt": build_prompt(metrics),
            "model": NARRATOR_MODEL,
        },
        "pain_points": PAIN_POINT_MAP,
        "available_models": list(MODELS.keys()),
        "summary": {   # compact row for the run-history table
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).strftime("%H:%M UTC"),
            "n_users": n_users,
            "seed": seed,
            "models": len(lb),
            "best_model": results["best_model"],
            "test_auc": float(best["test_auc"]),
            "revenue_at_risk": float(results["revenue_at_risk_annual_aed"]),
        },
    }

    RUNS[run_id] = {"bootstrap": bootstrap, "metrics": metrics}
    RUN_ORDER.append(run_id)
    while len(RUN_ORDER) > MAX_KEPT_RUNS:
        RUNS.pop(RUN_ORDER.pop(0), None)
    return run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    execute_run(
        n_users=int(os.environ.get("CHURN_LAB_USERS", 8000)),
        seed=int(os.environ.get("CHURN_LAB_SEED", 42)),
        test_size=0.25,
        models=list(MODELS.keys()),
    )
    yield


app = FastAPI(title="Churn Lab", lifespan=lifespan)


def _resolve(run_id: str = None) -> dict:
    rid = run_id or (RUN_ORDER[-1] if RUN_ORDER else None)
    if rid not in RUNS:
        raise HTTPException(404, f"unknown run '{run_id}'")
    return RUNS[rid]


@app.get("/api/bootstrap")
def bootstrap(run_id: str = None):
    return JSONResponse(_resolve(run_id)["bootstrap"])


@app.get("/api/runs")
def runs():
    return [RUNS[rid]["bootstrap"]["summary"] for rid in reversed(RUN_ORDER)]


class RunBody(BaseModel):
    n_users: int = Field(8000, ge=2000, le=20000)
    seed: int = Field(42, ge=0, le=10_000_000)
    test_size: float = Field(0.25, ge=0.15, le=0.4)
    models: list = Field(default_factory=lambda: list(MODELS.keys()))


@app.post("/api/run")
def run(body: RunBody):
    models = [m for m in body.models if m in MODELS] or list(MODELS.keys())
    run_id = execute_run(body.n_users, body.seed, body.test_size, models)
    return JSONResponse(RUNS[run_id]["bootstrap"])


class NarrateBody(BaseModel):
    api_key: str = ""
    run_id: str = None


@app.post("/api/narrate")
def narrate(body: NarrateBody):
    metrics = _resolve(body.run_id)["metrics"]
    key = body.api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        try:
            return {
                "source": f"Claude ({NARRATOR_MODEL})",
                "brief": narrate_with_claude(metrics, key),
            }
        except Exception as exc:  # fall back so the demo never breaks
            return {
                "source": f"deterministic template (Claude call failed: {exc})",
                "brief": narrate_with_template(metrics),
            }
    return {
        "source": "deterministic template (no API key provided)",
        "brief": narrate_with_template(metrics),
    }


app.mount("/", StaticFiles(directory="web", html=True), name="web")

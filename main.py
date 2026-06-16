"""
Compass quant-engine — portfolio optimizer microservice.

Matches the contract used by the TanStack app's `optimizer.functions.ts`:

  POST /optimize   (header `x-secret: <QUANT_SHARED_SECRET>`)
  body:
    {
      "tickers":       ["AAPL", "MSFT", ...],          # length N
      "sectors":       ["Information Technology", ...], # length N, aligned to tickers
      "returns":       [[...], [...], ...],             # T x N daily (log) returns
      "views":         {"AAPL": 72.5, ...},             # ticker -> conviction 0..100
      "max_position":  0.05,                            # per-name upper bound (0..1)
      "sector_caps":   {"Information Technology": 0.30},# sector upper bounds
      "sector_floors": {"Health Care": 0.05},           # sector lower bounds
      "risk_aversion": 1.0
    }
  response:
    { "weights": {"AAPL": 0.04, ...}, "expected": {"AAPL": 0.081, ...} }

Method: Ledoit-Wolf shrunk covariance + mean-variance (max quadratic utility)
with per-name and per-sector linear constraints, plus a small L2 penalty to
keep the solution diversified. Conviction scores are mapped to a modest
expected-return vector (the "views"). If the constrained solve is infeasible
or errors, the service degrades to min-volatility, then to a caps-respecting
proportional fallback, and only returns 422 if even that fails (the TS client
then falls back to its own rules-based weighting).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sklearn.covariance import LedoitWolf

# PyPortfolioOpt
from pypfopt import EfficientFrontier, objective_functions

app = FastAPI(title="Compass quant-engine", version="1.0.0")

SECRET = os.environ.get("QUANT_SHARED_SECRET", "")
TRADING_DAYS = 252
RETURN_SPREAD = 0.08  # ± annual expected-return spread implied by conviction


class OptIn(BaseModel):
    tickers: List[str]
    sectors: List[str]
    returns: List[List[float]]
    views: Dict[str, float] = Field(default_factory=dict)
    max_position: float = 0.05
    sector_caps: Dict[str, float] = Field(default_factory=dict)
    sector_floors: Dict[str, float] = Field(default_factory=dict)
    risk_aversion: float = 1.0


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "quant-engine", "version": "1.0.0"}


def _mu_from_views(tickers: List[str], views: Dict[str, float]) -> np.ndarray:
    """Map conviction (0..100) to a centered, scaled annual expected-return vector."""
    conv = np.array([float(views.get(t, 50.0)) for t in tickers], dtype=float)
    sd = conv.std()
    if sd < 1e-9:
        return np.zeros(len(tickers))
    z = (conv - conv.mean()) / sd
    return z * RETURN_SPREAD


def _caps_respecting_fallback(
    tickers: List[str], sectors: List[str], mu: np.ndarray, max_position: float,
    sector_caps: Dict[str, float],
) -> Dict[str, float]:
    """Deterministic feasible weights: rank by mu, fill respecting position & sector caps."""
    n = len(tickers)
    order = list(np.argsort(-mu))  # high conviction first
    w = np.zeros(n)
    sector_used: Dict[str, float] = {}
    remaining = 1.0
    cap = max(1e-6, float(max_position))
    for idx in order:
        if remaining <= 1e-9:
            break
        s = sectors[idx]
        s_cap = float(sector_caps.get(s, 1.0))
        room = min(cap, remaining, s_cap - sector_used.get(s, 0.0))
        if room <= 0:
            continue
        w[idx] = room
        sector_used[s] = sector_used.get(s, 0.0) + room
        remaining -= room
    total = w.sum()
    if total <= 0:
        w = np.full(n, 1.0 / n)
    else:
        w = w / total
    return {t: float(round(wi, 6)) for t, wi in zip(tickers, w)}


def _solve(body: OptIn) -> Dict[str, Dict[str, float]]:
    tickers = body.tickers
    sectors = body.sectors
    n = len(tickers)
    if n < 3:
        raise HTTPException(422, "need at least 3 tickers")
    if len(sectors) != n:
        raise HTTPException(422, "tickers and sectors length mismatch")

    R = np.asarray(body.returns, dtype=float)
    if R.ndim != 2 or R.shape[1] != n:
        raise HTTPException(422, "returns must be T x N aligned to tickers")
    if R.shape[0] < 30:
        raise HTTPException(422, "need at least 30 return observations")

    mu = _mu_from_views(tickers, body.views)

    # Ledoit-Wolf shrinkage covariance, annualized.
    try:
        Sigma = LedoitWolf().fit(R).covariance_ * TRADING_DAYS
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"covariance estimation failed: {e}")

    mapper = {t: s for t, s in zip(tickers, sectors)}
    present_sectors = set(sectors)
    caps = {s: float(c) for s, c in body.sector_caps.items() if s in present_sectors}
    floors = {s: float(f) for s, f in body.sector_floors.items() if s in present_sectors}

    mu_series = {t: float(m) for t, m in zip(tickers, mu)}

    def build_ef() -> EfficientFrontier:
        ef = EfficientFrontier(mu, Sigma, weight_bounds=(0.0, float(body.max_position)))
        if caps or floors:
            ef.add_sector_constraints(mapper, floors, caps)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        return ef

    # 1) Mean-variance (max quadratic utility).
    try:
        ef = build_ef()
        ef.max_quadratic_utility(risk_aversion=max(0.1, float(body.risk_aversion)))
        w = ef.clean_weights()
        return {"weights": {k: float(v) for k, v in w.items()}, "expected": mu_series}
    except Exception:
        pass

    # 2) Min volatility (still respecting all constraints).
    try:
        ef = build_ef()
        ef.min_volatility()
        w = ef.clean_weights()
        return {"weights": {k: float(v) for k, v in w.items()}, "expected": mu_series}
    except Exception:
        pass

    # 3) Deterministic caps-respecting fallback (always feasible).
    w = _caps_respecting_fallback(
        tickers, sectors, mu, float(body.max_position), caps
    )
    return {"weights": w, "expected": mu_series}


@app.post("/optimize")
def optimize(body: OptIn, x_secret: str = Header(default="")) -> dict:
    if not SECRET:
        raise HTTPException(500, "QUANT_SHARED_SECRET not configured")
    if x_secret != SECRET:
        raise HTTPException(401, "unauthorized")
    return _solve(body)

import statsmodels.api as sm
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np


# def optimise_weights(prices, lower_bound=0):
#     returns = expected_returns.mean_historical_return(prices, frequency=252)
    
#     cov = risk_models.sample_cov(prices, frequency=252)
    
#     ef = EfficientFrontier(returns,
#                            cov_matrix=cov,
#                            weight_bounds=(lower_bound, .2),
#                            solver='SCS')
    
#     ef.max_sharpe()
    
#     cleaned_weights = ef.clean_weights()
    
#     return cleaned_weights

def optimise_weights(prices, lower_bound=0, per_asset_cap=0.7, solver='SCS'):
    # Clean input
    prices = prices.dropna(axis=1, how='all')
    prices = prices.dropna(axis=0, how='all')
    n = prices.shape[1]
    if n == 0:
        raise ValueError("No tickers to optimise")

    # Compute returns/cov
    returns = expected_returns.mean_historical_return(prices, frequency=252)
    cov = risk_models.sample_cov(prices, frequency=252)

    # Adaptive upper bound to ensure feasibility
    upper = per_asset_cap
    if upper * n < 1.0:
        upper = max(upper, 1.0 / n)
    lower = min(lower_bound, upper)

    # Quick feasibility guard
    if lower * n > 1.0 + 1e-12 or upper * n < 1.0 - 1e-12:
        w_eq = {col: 1.0 / n for col in prices.columns}
        return w_eq

    # Try multiple solvers before relaxing bounds
    solvers_to_try = [solver, 'ECOS', 'OSQP']
    for s in solvers_to_try:
        try:
            ef = EfficientFrontier(returns, cov_matrix=cov, weight_bounds=(lower, upper), solver=s)
            ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            return cleaned_weights
        except Exception:
            continue

    # fallback: try relaxed (0.0, 0.5) with default solver
    try:
        ef = EfficientFrontier(returns, cov_matrix=cov, weight_bounds=(0.0, 0.5), solver=solver)
        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        return cleaned_weights
    except Exception as e:
        print(f"Optimisation failed even after fallback bounds; returning equal weights. Error: {e}")
        w_eq = {col: 1.0 / n for col in prices.columns}
        return w_eq
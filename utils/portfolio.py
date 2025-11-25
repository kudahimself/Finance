import statsmodels.api as sm
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np


def optimise_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices, frequency=252)
    
    cov = risk_models.sample_cov(prices, frequency=252)
    
    ef = EfficientFrontier(returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .2),
                           solver='SCS')
    
    ef.max_sharpe()
    
    cleaned_weights = ef.clean_weights()
    
    return cleaned_weights
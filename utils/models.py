from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV



def rolling_train_predict_windowed(df, features,
                                   date_col='date', ticker_col='ticker',
                                   target_col='target_1m',
                                   top_k=30, model_type='ridge',
                                   window_months=12, min_train_rows=100,
                                   tune_model=True):
    """
    For each unique date d in the cross-section:
      - Train on rows with date in [d - window_months, d - 1 day]
      - Predict for the cross-section at date d
      - Select top_k tickers by predicted target
    Returns: fixed_dates dict, diagnostics DataFrame, last trained model
    """
    fixed_dates = {}
    diagnostics = []
    last_model = None
    last_scaler = None
    dates = sorted(df[date_col].unique())
    for d in dates:
        train_start = (pd.to_datetime(d) - pd.DateOffset(months=window_months)).normalize()
        train_end = (pd.to_datetime(d)).normalize()

        train = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)].dropna(subset=features + [target_col])
        if train.shape[0] < min_train_rows:
            # skip if not enough training rows
            continue

        X_train = train[features]
        y_train = train[target_col]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        if model_type == 'ridge':
            model = Ridge(random_state=0)
            param_grid = {'alpha': [0.1, 1.0, 10.0]} if tune_model else None
        elif model_type == 'xgboost':
            model = XGBRegressor(random_state=0, n_jobs=-1)
            param_grid = {'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 6, 9]} if tune_model else None
        elif model_type == 'elasticnet':
            model = ElasticNet(random_state=0, max_iter=5000)
            param_grid = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]} if tune_model else None
        elif model_type == 'gradient':
            model = GradientBoostingRegressor(random_state=0)
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 6]} if tune_model else None
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=-1)
            param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 6, 12]} if tune_model else None

        if tune_model and param_grid is not None:
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
            grid.fit(X_train_s, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train_s, y_train)

        last_model = model
        last_scaler = scaler

        pool = df[df[date_col] == d].dropna(subset=features).copy()
        if pool.empty:
            continue

        X_pred = scaler.transform(pool[features])
        y_pred = model.predict(X_pred)

        pool = pool.assign(y_pred=y_pred)

        selected = pool[pool['y_pred'] > 0].nlargest(top_k, 'y_pred')
        fixed_dates[d.strftime('%Y-%m-%d')] = selected[ticker_col].tolist()

        diagnostics.append({
            'date': d,
            'n_pool': int(pool.shape[0]),
            'n_train': int(train.shape[0]),
            'pred_mean': float(selected['y_pred'].mean()) if not selected.empty else float('nan'),
            'realized_mean': float(selected[target_col].mean()) if not selected.empty else float('nan')
        })

    diag_df = pd.DataFrame(diagnostics).set_index('date') if diagnostics else pd.DataFrame()
    return fixed_dates, diag_df, last_model, last_scaler


def calculate_betas(factor_data: pd.DataFrame, method: str) -> pd.DataFrame:
    betas = (factor_data.groupby(level='ticker', group_keys=False)
    .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                window=min(24, x.shape[0]),
                                min_nobs=len(x.columns)+1)
    .fit(params_only=True)
    .params
    .drop('const', axis=1)))

    return betas


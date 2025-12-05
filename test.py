"""Utility script: build and plot baseline vs permutation portfolio cumulative returns.

This module provides two helpers:
- compute_portfolio_returns_from_fixed(fixed_dates, price_df):
    Build strategy daily simple returns from a mapping of rebalance dates -> tickers.
- plot_baseline_and_permutations(...):
    Run permutations (shuffling `target_1m`), build portfolios for each permutation
    (using `random_forest_windowed` without tuning), and plot cumulative returns
    for baseline vs permutations.

Intended usage: move the relevant functions into a notebook cell and call
`plot_baseline_and_permutations(...)` after running the cells that prepare
`model_df`, `features`, `fixed_dates_pred`, and `new_df`.
"""

from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


def compute_portfolio_returns_from_fixed(
    fixed_dates: Dict[str, List[str]],
    price_df: pd.DataFrame,
    optimise_weights_func,
    lower_bound_factor: float = 0.5,
) -> pd.DataFrame:
    """Given `fixed_dates` (rebalance date -> tickers) and `price_df` (prices),
    compute a daily timeseries of strategy simple returns based on optimised
    weights for each rebalance period.

    Parameters
    - fixed_dates: mapping 'YYYY-MM-DD' -> list of tickers
    - price_df: DataFrame indexed by date, columns=tickers (prices)
    - optimise_weights_func: callable(prices, lower_bound=...) -> weight array/series
    - lower_bound_factor: fraction to multiply equal-weight lower bound by

    Returns DataFrame with column 'strategy_return' indexed by date.
    """
    if not fixed_dates:
        return pd.DataFrame()

    pct = price_df.pct_change()
    portfolio_parts = []

    for start_date, tickers in sorted(fixed_dates.items()):
        try:
            start_ts = pd.to_datetime(start_date)
            end_ts = start_ts + pd.offsets.MonthEnd(0)

            opt_start = (start_ts - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
            opt_end = (start_ts - pd.DateOffset(days=1)).strftime("%Y-%m-%d")

            opt_prices = price_df.loc[opt_start:opt_end, tickers].dropna(axis=1, how="any")
            if opt_prices.shape[1] == 0:
                continue

            try:
                lb = round(1 / opt_prices.shape[1] * lower_bound_factor, 3)
                w = optimise_weights_func(prices=opt_prices, lower_bound=lb)
                if isinstance(w, (list, tuple, np.ndarray)):
                    weights = pd.Series(w, index=opt_prices.columns)
                else:
                    weights = pd.Series(w)
                    weights.index = opt_prices.columns[: len(weights)]
            except Exception:
                weights = pd.Series(1 / opt_prices.shape[1], index=opt_prices.columns)

            holding_returns = pct.loc[start_ts.strftime("%Y-%m-%d"):end_ts.strftime("%Y-%m-%d"), opt_prices.columns]
            if holding_returns.empty:
                continue

            weights = weights.reindex(holding_returns.columns).fillna(0)
            strat_ret = holding_returns.mul(weights, axis=1).sum(axis=1).to_frame("strategy_return")
            portfolio_parts.append(strat_ret)
        except Exception:
            continue

    if not portfolio_parts:
        return pd.DataFrame()

    out = pd.concat(portfolio_parts).sort_index()
    out = out.groupby(out.index).sum()
    return out


def plot_baseline_and_permutations(
    fixed_dates_pred: Dict[str, List[str]],
    price_df: pd.DataFrame,
    model_df: pd.DataFrame,
    features: List[str],
    random_forest_windowed_func,
    optimise_weights_func,
    n_permutations: int = 20,
    top_k: int = 15,
    window_months: int = 12,
    min_train_rows: int = 150,
    random_seed_base: int = 42,
    plot_each_perm: bool = False,
    save_each_perm_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Compute baseline portfolio and permutation portfolios, plot cumulative returns.

    Returns (perm_df_cum, baseline_cum)
    """
    missing = []
    if fixed_dates_pred is None or price_df is None or model_df is None or features is None:
        raise ValueError("One of the required inputs is None")

    baseline_pf = compute_portfolio_returns_from_fixed(fixed_dates_pred, price_df, optimise_weights_func)
    baseline_cum = (1 + baseline_pf["strategy_return"]).cumprod() - 1 if not baseline_pf.empty else pd.Series(dtype=float)

    perm_cums = []
    for i in tqdm(range(n_permutations), desc="Permutations"):
        perm = model_df.copy()
        rng = np.random.RandomState(random_seed_base + i)
        shuffled = perm["target_1m"].values.copy()
        rng.shuffle(shuffled)
        perm["target_1m"] = shuffled

        fixed_p, _, _, _ = random_forest_windowed_func(
            perm, features, top_k=top_k, window_months=window_months, min_train_rows=min_train_rows, tune_model=False
        )
        pf = compute_portfolio_returns_from_fixed(fixed_p, price_df, optimise_weights_func)
        if pf.empty:
            continue
        cum = (1 + pf["strategy_return"]).cumprod() - 1
        cum.name = f"perm_{i}"
        perm_cums.append(cum)

    perm_df_cum = pd.concat(perm_cums, axis=1) if perm_cums else pd.DataFrame()
    saved_files: List[str] = []

    # Ensure save directory exists if requested
    if save_each_perm_dir:
        os.makedirs(save_each_perm_dir, exist_ok=True)

    # Optional: plot/save each permutation individually during loop
    if plot_each_perm and perm_cums:
        for i, col in enumerate(perm_df_cum.columns):
            fig, ax = plt.subplots(figsize=(10, 5))
            perm_df_cum[col].plot(ax=ax, color="gray", alpha=0.9, linewidth=1.5, label=f"perm_{i}")
            if not baseline_cum.empty:
                baseline_cum.plot(ax=ax, color="red", linewidth=2, label="baseline")
            ax.set_title(f"Permutation {i} cumulative returns vs baseline")
            ax.set_ylabel("Cumulative return")
            ax.legend()
            plt.tight_layout()
            if save_each_perm_dir:
                fname = os.path.join(save_each_perm_dir, f"perm_{i}_vs_baseline.png")
                fig.savefig(fname, dpi=150)
                saved_files.append(fname)
                plt.close(fig)
            else:
                plt.show()

    # Aggregate plot (all permutations + median band + baseline)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not perm_df_cum.empty:
        for col in perm_df_cum.columns:
            perm_df_cum[col].plot(ax=ax, color="gray", alpha=0.4, linewidth=1)
        median = perm_df_cum.median(axis=1)
        low = perm_df_cum.quantile(0.05, axis=1)
        high = perm_df_cum.quantile(0.95, axis=1)
        ax.fill_between(median.index, low.values, high.values, color="gray", alpha=0.15, label="5-95% perm band")
        median.plot(ax=ax, color="black", linestyle="--", linewidth=1, label="perm median")
    if not baseline_cum.empty:
        baseline_cum.plot(ax=ax, color="red", linewidth=2, label="baseline")
    ax.set_title("Cumulative returns: baseline (red) vs permutations (gray)")
    ax.set_ylabel("Cumulative return")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return perm_df_cum, baseline_cum, saved_files


# Permutation test for `rolling_train_predict_windowed`
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt

def permutation_test_rolling(model_df, features, n_permutations=100, random_state=0, tune_model=False, top_k=15, window_months=12, min_train_rows=150):
    """Run a permutation test that shuffles the `target_1m` column and recomputes
    the mean `realized_mean` reported by `rolling_train_predict_windowed`.

    Returns a dict with baseline metric, permutation metrics array and p-value.
    """
    rng = np.random.RandomState(random_state)

    # Baseline run (use the provided `tune_model` flag; for permutations we usually disable tuning)
    print("Running baseline (this may take a while)...")
    baseline_fixed, diag, last_model, last_scaler = random_forest_windowed(
        model_df, features, top_k=top_k,
        window_months=window_months, min_train_rows=min_train_rows,
        tune_model=tune_model
    )

    if diag.empty:
        print("Baseline produced no diagnostics (insufficient history). Aborting.")
        return None

    baseline_metric = diag['realized_mean'].mean()
    print(f'Baseline realized_mean mean = {baseline_metric:.6f}')

    perm_metrics = []
    perm_fixed = []
    for i in tqdm(range(n_permutations), desc='Permutations'):
        perm_df = model_df.copy()
        # Shuffle the target column across all rows (breaks temporal association)
        permuted = perm_df['target_1m'].values.copy()
        rng.shuffle(permuted)
        perm_df['target_1m'] = permuted

        # Use same model settings but **disable tuning** for speed during permutations
        fixed_p, pdiag, _, _ = random_forest_windowed(
            perm_df, features, top_k=top_k,
            window_months=window_months, min_train_rows=min_train_rows,
            tune_model=False
        )

        # store fixed mapping for this permutation (even if diag empty)
        perm_fixed.append(fixed_p)

        if pdiag.empty:
            perm_metrics.append(np.nan)
        else:
            perm_metrics.append(pdiag['realized_mean'].mean())

    perm_metrics = np.array(perm_metrics, dtype=float)
    valid_mask = ~np.isnan(perm_metrics)
    perm_metrics_valid = perm_metrics[valid_mask]
    perm_fixed_valid = [pf for pf, ok in zip(perm_fixed, valid_mask) if ok]

    if perm_metrics_valid.size == 0:
        print("No valid permutation diagnostics were produced.")
        return {
            'baseline': baseline_metric,
            'perm_metrics': perm_metrics,
            'p_value': np.nan,
            'baseline_fixed': baseline_fixed,
            'perm_fixed': perm_fixed,
        }

    # p-values: compute both directions and two-sided
    p_value_gte = (perm_metrics_valid >= baseline_metric).mean()  # prob perm >= baseline
    p_value_lte = (perm_metrics_valid <= baseline_metric).mean()  # prob perm <= baseline
    p_value_two_sided = 2 * min(p_value_gte, p_value_lte)
    p_value_two_sided = min(p_value_two_sided, 1.0)

    # effect size and median of permutation null
    median_perm = float(np.median(perm_metrics_valid))
    effect_size = float(baseline_metric - median_perm)

    # Plot the permutation distribution
    plt.figure(figsize=(8,4))
    plt.hist(perm_metrics_valid, bins=30, alpha=0.8)
    plt.axvline(baseline_metric, color='red', linewidth=2, label=f'baseline={baseline_metric:.6f}')
    plt.axvline(median_perm, color='black', linestyle='--', linewidth=1, label=f'perm_median={median_perm:.6f}')
    plt.xlabel('Mean realized_mean (selected tickers)')
    plt.legend()
    plt.title('Permutation distribution (realized_mean)')
    plt.show()

    print(f'Permutation p-value (>= baseline): {p_value_gte:.4f}; (<= baseline): {p_value_lte:.4f}; two-sided: {p_value_two_sided:.4f} (n={len(perm_metrics_valid)})')
    print(f'Baseline - perm_median = effect_size: {effect_size:.6f}')

    return {
        'baseline': baseline_metric,
        'perm_metrics': perm_metrics,
        'perm_metrics_valid': perm_metrics_valid,
        'n_valid_permutations': int(len(perm_metrics_valid)),
        'p_value_gte': float(p_value_gte),
        'p_value_lte': float(p_value_lte),
        'p_value_two_sided': float(p_value_two_sided),
        'median_perm': median_perm,
        'effect_size': effect_size,
        'baseline_fixed': baseline_fixed,
        'perm_fixed': perm_fixed,
    }


if __name__ == "__main__":
    print("This module provides helpers to compute and plot baseline vs permutation portfolios.")
    print("Import and call `plot_baseline_and_permutations(...)` from a notebook after preparing your data.")
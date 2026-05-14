# -*- coding: utf-8 -*-
"""
Assignment 5 framework: Candidate stock selection, rebalance, and alpha portfolio
================================================================================

This is a STUDENT framework file.  Data loading, weekly feature construction,
preprocessing, and industry-neutralization utilities are provided.  The core
functions for Assignment 5 are intentionally left as TODOs.

Assignment focus
----------------
1. Build a multi-factor pool and combine factors by Rank Aggregation.
2. Construct the investable stock universe.
3. Apply PE lowest 30% filtering.  Do NOT use PB as an additional candidate
   selection filter in this revised version.
4. Apply Top-N selection for N in {10, 20, 30, 50, 100}.
5. Build equal-weight long-only portfolios.
6. Build benchmark and alpha portfolios.
7. Compare daily, weekly, and monthly rebalancing.
8. Compare different transaction cost assumptions.

Important time alignment
------------------------
Weights formed at date t should be applied to returns from t to the next
trading/rebalance date.  Avoid using future returns or future factor values in
candidate selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 0. Configuration
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "dataset"
INDUSTRY_FILE = DATA_DIR / "stk_company_info.csv"

# The revised assignment uses PE for the candidate filtering stage.
# PB can still be used as a factor signal if desired, but it should not be used
# as an additional quantile filter for final candidate selection.
FIELDS = ["close", "pb", "total_mv", "pe_ttm", "amount", "adj_factor"]


# ============================================================
# 1. Data loading and feature construction
# ============================================================

def read_one(field: str, market: str | None) -> pd.DataFrame:
    """
    Read one CSV file.

    Parameters
    ----------
    field : str
        Field name, such as 'close', 'pb', 'pe_ttm', 'amount', or 'adj_factor'.
    market : str or None
        'sh', 'sz', or None. If None, read file without market suffix.

    Returns
    -------
    pd.DataFrame
        Data matrix indexed by date.
    """
    path = DATA_DIR / (f"{field}.csv" if market is None else f"{field}_{market}.csv")
    df = pd.read_csv(path, index_col=0)

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    # Some Shenzhen files may contain duplicated dates.
    if market == "sz":
        df = df[~df.index.duplicated(keep="first")]

    return df


def load_data() -> dict[str, pd.DataFrame]:
    """
    Load Shanghai and Shenzhen market data and align all fields.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing close, pb, pe_ttm, total_mv, amount, and adj_factor.
    """
    dataset: dict[str, pd.DataFrame] = {}

    for field in FIELDS:
        sh = read_one(field, "sh")
        sz = read_one(field, "sz")
        dataset[field] = pd.concat([sh, sz], axis=1).sort_index()
        print(f"Data {field} is loaded")

    common_index = dataset["close"].index
    common_columns = dataset["close"].columns

    for field in FIELDS[1:]:
        common_index = common_index.intersection(dataset[field].index)
        common_columns = common_columns.intersection(dataset[field].columns)

    for field in FIELDS:
        dataset[field] = dataset[field].reindex(index=common_index, columns=common_columns)

    # Remove invalid observations.
    dataset["close"] = dataset["close"].mask(dataset["close"] <= 0)
    dataset["adj_factor"] = dataset["adj_factor"].mask(dataset["adj_factor"] <= 0)
    dataset["pb"] = dataset["pb"].mask(dataset["pb"] <= 0)
    dataset["pe_ttm"] = dataset["pe_ttm"].mask(dataset["pe_ttm"] <= 0)
    dataset["total_mv"] = dataset["total_mv"].mask(dataset["total_mv"] <= 0)
    dataset["amount"] = dataset["amount"].mask(dataset["amount"] < 0)

    return dataset


def winsorize_row(row: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """Cross-sectional winsorization for one date."""
    valid = row.dropna()
    if valid.empty:
        return row
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return row.clip(lower=lo, upper=hi)


def zscore_row(row: pd.Series) -> pd.Series:
    """Cross-sectional z-score standardization for one date."""
    valid = row.dropna()
    if valid.shape[0] < 2:
        return row * np.nan
    mu = valid.mean()
    sd = valid.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return row * np.nan
    return (row - mu) / sd


def preprocess_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize and standardize factor exposures cross-sectionally."""
    out = df.copy()
    out = out.apply(winsorize_row, axis=1)
    out = out.apply(zscore_row, axis=1)
    return out


def build_weekly_features(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build weekly return and style factor matrices.

    Notes
    -----
    - ``pe_raw_w`` stores the raw PE value and should be used for PE lowest 30%
      candidate filtering.
    - ``pe_w`` stores an earnings-yield style signal, 1 / PE, after preprocessing.
      It can be used as a factor signal in rank aggregation.
    """
    close = dataset["close"] * dataset["adj_factor"]
    pb = dataset["pb"]
    pe_ttm = dataset["pe_ttm"]
    total_mv = dataset["total_mv"]
    amount = dataset["amount"]

    daily_ret = close.pct_change()

    mom20 = close / close.shift(20) - 1
    value = (1.0 / pb).replace([np.inf, -np.inf], np.nan)
    pe_signal = (1.0 / pe_ttm).replace([np.inf, -np.inf], np.nan)
    size = -np.log(total_mv)
    vol20 = daily_ret.rolling(20).std()
    amount20 = amount.rolling(20).mean()

    weekly = {
        "close_w": close.resample("W").last(),
        "ret_w": close.resample("W").last().pct_change(),
        "size_w": size.resample("W").last(),
        "value_w": value.resample("W").last(),
        "pe_w": pe_signal.resample("W").last(),
        "pe_raw_w": pe_ttm.resample("W").last(),
        "mom_w": mom20.resample("W").last(),
        "vol_w": vol20.resample("W").last(),
        "amount20_w": amount20.resample("W").last(),
    }

    for key in ["size_w", "value_w", "pe_w", "mom_w", "vol_w"]:
        weekly[key] = preprocess_exposure(weekly[key])

    return weekly


# ============================================================
# 2. Industry neutralization
# ============================================================

def get_industries(ind: str = "l1_code") -> pd.DataFrame:
    """
    Load industry classification information.

    Parameters
    ----------
    ind : str
        Industry column name. Common choices include 'l1_code', 'l2_code', 'l3_code'.
    """
    industry = pd.read_csv(INDUSTRY_FILE)
    industry.index = industry["ts_code"]
    industry = industry[[ind, "in_date"]]
    return industry


class IndustryNeutral:
    """
    Industry neutralization by within-industry z-score.
    """

    def __init__(self, ind: str = "l1_code"):
        self.ind = ind
        self.industry = get_industries(ind)

    def __call__(self, factor: pd.DataFrame | dict[str, pd.DataFrame]):
        if isinstance(factor, dict):
            out = {}
            for factor_name, factor_df in factor.items():
                out[factor_name] = self._neutralize_one_factor(factor_df)
                print(f'factor "{factor_name}" neutralized.')
            return out

        if isinstance(factor, pd.DataFrame):
            return self._neutralize_one_factor(factor)

        raise TypeError("factor must be a pandas DataFrame or a dictionary of DataFrames.")

    def _neutralize_one_factor(self, factor: pd.DataFrame) -> pd.DataFrame:
        fac = factor.copy()
        code_list = self.industry.index.tolist()

        stock_industry = pd.Series(index=fac.columns, dtype=object)

        for code in fac.columns:
            if code not in code_list:
                stock_industry.loc[code] = "0"
                continue

            ind_info = self.industry.loc[code]

            if isinstance(ind_info, pd.DataFrame) and len(ind_info) > 1:
                ind_info = ind_info[ind_info["in_date"] == ind_info["in_date"].max()]
                ind_info = ind_info.iloc[0]

            stock_industry.loc[code] = ind_info[self.ind]

        stock_industry = stock_industry.fillna("0")
        unique_industries = [g for g in stock_industry.unique() if g != "0"]

        for g in unique_industries:
            cols = stock_industry[stock_industry == g].index
            if len(cols) == 0:
                continue

            sub = fac.loc[:, cols]
            mean = sub.mean(axis=1, skipna=True)
            std = sub.std(axis=1, skipna=True, ddof=0).replace(0, np.nan)
            fac.loc[:, cols] = sub.sub(mean, axis=0).div(std, axis=0)

        return fac


# ============================================================
# 3. Assignment 5 TODO functions
# ============================================================

def rank_aggregation(
    factor_dict: dict[str, pd.DataFrame],
    directions: dict[str, int],
    weights: dict[str, float] | None = None,
    min_non_missing: int = 2,
) -> pd.DataFrame:
    """
    TODO: Combine multiple factor matrices into one composite score matrix.

    Parameters
    ----------
    factor_dict : dict[str, pd.DataFrame]
        Dictionary of factor matrices.  Each matrix should be indexed by date and
        have stock codes as columns.
    directions : dict[str, int]
        Factor direction dictionary.
        Use +1 if larger factor exposure is better.
        Use -1 if smaller factor exposure is better.
    weights : dict[str, float] or None
        Optional factor weights.  If None, use equal weights.
    min_non_missing : int
        Minimum number of valid factor scores required for a stock-date pair.

    Returns
    -------
    pd.DataFrame
        Composite score matrix S. Higher score should mean more attractive stock.

    Hints
    -----
    1. Align all factor matrices by common dates and common stocks.
    2. For each factor and each date, compute cross-sectional ranks.
    3. Convert ranks into percentile scores or normalized rank scores.
    4. Adjust factor direction so that larger rank score is always better.
    5. Average the rank scores across factors.
    6. Do not blindly fill filtered or missing exposures with zero.
    """
    # TODO: implement Rank Aggregation.
    raise NotImplementedError("TODO: implement rank_aggregation.")


def build_investable_universe(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    pe: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    min_amount: float | None = None,
) -> pd.DataFrame:
    """
    TODO: Construct the investable stock universe mask.

    Parameters
    ----------
    score : pd.DataFrame
        Composite score matrix.
    returns : pd.DataFrame
        Return matrix.  At date t, students should ensure that the next-period
        return used in backtesting is available.
    pe : pd.DataFrame
        Raw PE matrix used for PE lowest 30% filtering.
    amount : pd.DataFrame or None
        Trading amount or liquidity proxy.
    min_amount : float or None
        Optional minimum liquidity threshold.

    Returns
    -------
    pd.DataFrame
        Boolean matrix. True means the stock is investable at date t.

    Hints
    -----
    A simple version may require non-missing score, PE, and next-period return.
    A more complete version may also require sufficient trading amount.
    """
    # TODO: implement investable universe filtering.
    raise NotImplementedError("TODO: implement build_investable_universe.")


def pe_quantile_filter(
    pe: pd.DataFrame,
    investable_mask: pd.DataFrame,
    quantile: float = 0.30,
) -> pd.DataFrame:
    """
    TODO: Keep stocks with PE in the lowest given quantile.

    Parameters
    ----------
    pe : pd.DataFrame
        Raw PE matrix.  Lower PE is preferred in this filtering step.
    investable_mask : pd.DataFrame
        Boolean investable universe mask.
    quantile : float, default=0.30
        Quantile cutoff.  0.30 means PE lowest 30%.

    Returns
    -------
    pd.DataFrame
        Boolean matrix. True means the stock passes PE lowest 30% filtering.

    Important
    ---------
    This revised assignment uses only PE lowest 30% for candidate filtering.
    Do not add PB lowest 30% filtering here.
    """
    # TODO: implement PE lowest 30% filter.
    raise NotImplementedError("TODO: implement pe_quantile_filter.")


def select_top_n(
    score: pd.DataFrame,
    candidate_mask: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    TODO: Select the Top-N stocks from the PE-filtered candidate universe.

    Parameters
    ----------
    score : pd.DataFrame
        Composite score matrix. Higher score is better.
    candidate_mask : pd.DataFrame
        Boolean mask after investable filtering and PE filtering.
    n : int
        Number of stocks to select.

    Returns
    -------
    pd.DataFrame
        Boolean matrix. True means the stock is selected at date t.
    """
    # TODO: implement Top-N selection.
    raise NotImplementedError("TODO: implement select_top_n.")


def build_equal_weight_weights(selection_mask: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Convert selected stocks into equal-weight long-only portfolio weights.

    Parameters
    ----------
    selection_mask : pd.DataFrame
        Boolean matrix produced by select_top_n.

    Returns
    -------
    pd.DataFrame
        Long-only weight matrix. Each row should sum to 1 when at least one stock
        is selected, and 0 otherwise.
    """
    # TODO: implement equal-weight long-only weights.
    raise NotImplementedError("TODO: implement build_equal_weight_weights.")


def build_benchmark_weights(investable_mask: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Construct benchmark weights.

    Parameters
    ----------
    investable_mask : pd.DataFrame
        Boolean matrix for the benchmark universe.  A simple benchmark can be the
        equal-weight portfolio of all investable stocks before PE filtering.

    Returns
    -------
    pd.DataFrame
        Benchmark weight matrix.
    """
    # TODO: implement benchmark weights.
    raise NotImplementedError("TODO: implement build_benchmark_weights.")


def build_alpha_weights(
    long_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    TODO: Construct alpha portfolio weights.

    Definition
    ----------
    alpha_weights = long_only_weights - benchmark_weights

    Returns
    -------
    pd.DataFrame
        Alpha weight matrix.  Row sums should usually be close to zero if both
        long-only and benchmark portfolios are fully invested.
    """
    # TODO: implement alpha weights.
    raise NotImplementedError("TODO: implement build_alpha_weights.")


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    TODO: Choose rebalance dates from a trading-date index.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Available dates.
    freq : str
        Rebalance frequency.  Suggested values: 'D', 'W', 'M'.

    Returns
    -------
    pd.DatetimeIndex
        Selected rebalance dates.

    Hints
    -----
    - Daily: use all dates.
    - Weekly: use the last available date in each week.
    - Monthly: use the last available date in each month.
    """
    # TODO: implement rebalance date selection.
    raise NotImplementedError("TODO: implement get_rebalance_dates.")


def apply_rebalance_schedule(weights: pd.DataFrame, rebalance_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """
    TODO: Convert rebalance-date weights into daily/weekly holding weights.

    Parameters
    ----------
    weights : pd.DataFrame
        Target weights computed on all signal dates or all available dates.
    rebalance_dates : iterable of pd.Timestamp
        Dates when the portfolio is allowed to update weights.

    Returns
    -------
    pd.DataFrame
        Holding weights after applying the rebalance schedule.  Between two
        rebalance dates, weights can be forward-filled.
    """
    # TODO: implement rebalance schedule and forward-fill holdings.
    raise NotImplementedError("TODO: implement apply_rebalance_schedule.")


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    TODO: Compute one-way portfolio turnover.

    A common definition is
        turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|.

    Returns
    -------
    pd.Series
        Turnover series indexed by date.
    """
    # TODO: implement turnover calculation.
    raise NotImplementedError("TODO: implement compute_turnover.")


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """
    TODO: Compute net portfolio returns from weights and asset returns.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights formed at date t.
    returns : pd.DataFrame
        Asset returns.  Make sure to align weights at t with next-period returns.
    transaction_cost : float
        One-way transaction cost parameter.

    Returns
    -------
    pd.Series
        Net portfolio return series.

    Hints
    -----
    1. Use weights.shift(1) or another explicit alignment convention to avoid
       look-ahead bias.
    2. Compute gross return first.
    3. Subtract transaction_cost * turnover_t.
    """
    # TODO: implement net portfolio return calculation.
    raise NotImplementedError("TODO: implement compute_portfolio_returns.")


def compute_nav(portfolio_returns: pd.Series) -> pd.Series:
    """
    TODO: Compute cumulative NAV from a return series.
    """
    # TODO: implement NAV calculation.
    raise NotImplementedError("TODO: implement compute_nav.")


def compute_alpha_returns(
    long_returns_net: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """
    TODO: Compute net alpha return.

    Simplified assignment convention
    --------------------------------
    net alpha return = net long-only return - benchmark return.

    This avoids double-counting benchmark-side transaction costs unless students
    explicitly choose a different convention and explain it in the report.
    """
    # TODO: implement alpha return calculation.
    raise NotImplementedError("TODO: implement compute_alpha_returns.")


def performance_summary(ret: pd.Series, nav: pd.Series | None = None, periods_per_year: int = 52) -> pd.Series:
    """
    TODO: Report main performance statistics.

    Suggested metrics
    -----------------
    - annualized return
    - annualized volatility
    - Sharpe ratio
    - maximum drawdown
    - final NAV
    """
    # TODO: implement performance summary.
    raise NotImplementedError("TODO: implement performance_summary.")


def run_single_backtest(
    score: pd.DataFrame,
    pe: pd.DataFrame,
    returns: pd.DataFrame,
    amount: pd.DataFrame | None,
    n: int,
    rebalance_freq: str,
    transaction_cost: float,
) -> dict[str, object]:
    """
    TODO: Run one complete configuration of the Assignment 5 backtest.

    Parameters
    ----------
    score : pd.DataFrame
        Composite factor score matrix from rank_aggregation.
    pe : pd.DataFrame
        Raw PE matrix for PE lowest 30% filtering.
    returns : pd.DataFrame
        Asset return matrix.
    amount : pd.DataFrame or None
        Liquidity proxy.
    n : int
        Top-N size.
    rebalance_freq : str
        'D', 'W', or 'M'.
    transaction_cost : float
        One-way transaction cost.

    Returns
    -------
    dict[str, object]
        Suggested keys:
        - 'long_weights'
        - 'benchmark_weights'
        - 'alpha_weights'
        - 'long_returns_net'
        - 'benchmark_returns'
        - 'alpha_returns_net'
        - 'long_nav'
        - 'benchmark_nav'
        - 'alpha_nav'
        - 'turnover'
        - 'summary'
    """
    # TODO: assemble the full workflow:
    # 1. build investable universe
    # 2. apply PE lowest 30% filter
    # 3. select Top-N stocks
    # 4. build long-only and benchmark weights
    # 5. apply rebalance schedule
    # 6. compute long-only, benchmark, and alpha returns
    # 7. compute NAV and performance statistics
    raise NotImplementedError("TODO: implement run_single_backtest.")


def plot_nav_comparison(nav_dict: dict[str, pd.Series], title: str) -> None:
    """
    Plot several NAV curves in one figure.

    This plotting function is provided for convenience and can be used after the
    TODO backtest functions are completed.
    """
    plt.figure(figsize=(12, 5))
    for label, nav in nav_dict.items():
        plt.plot(nav.index, nav.values, label=label)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.ylabel("NAV")
    plt.legend(loc="best")
    plt.grid(linestyle=":", alpha=0.6)

    ax = plt.gca()
    if len(nav_dict) > 0:
        first_nav = next(iter(nav_dict.values()))
        if isinstance(first_nav.index, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Main experiment skeleton
# ============================================================

if __name__ == "__main__":
    dataset = load_data()
    features = build_weekly_features(dataset)

    returns = features["ret_w"]
    pe_raw = features["pe_raw_w"]
    amount = features["amount20_w"]

    raw_factor_dict = {
        "size": features["size_w"],
        "value": features["value_w"],
        "pe": features["pe_w"],
        "momentum": features["mom_w"],
        "volatility": features["vol_w"],
    }

    # Students may choose whether to use industry-neutralized factors.
    neutralizer = IndustryNeutral(ind="l1_code")
    factor_dict = neutralizer(raw_factor_dict)

    # Direction convention: +1 means larger is better; -1 means smaller is better.
    # After preprocessing, these choices should be explained in the report.
    factor_directions = {
        "size": +1,
        "value": +1,
        "pe": +1,
        "momentum": +1,
        "volatility": -1,
    }

    top_n_list = [10, 20, 30, 50, 100]
    rebalance_freq_list = ["D", "W", "M"]
    transaction_cost_list = [0.0, 0.001, 0.002]

    # ========================================================
    # TODO for students:
    # After completing the TODO functions above, uncomment and run the loop below.
    # ========================================================

    # composite_score = rank_aggregation(
    #     factor_dict=factor_dict,
    #     directions=factor_directions,
    #     weights=None,
    #     min_non_missing=2,
    # )
    #
    # all_results = {}
    # for n in top_n_list:
    #     for freq in rebalance_freq_list:
    #         for tc in transaction_cost_list:
    #             key = f"N={n}, freq={freq}, cost={tc}"
    #             all_results[key] = run_single_backtest(
    #                 score=composite_score,
    #                 pe=pe_raw,
    #                 returns=returns,
    #                 amount=amount,
    #                 n=n,
    #                 rebalance_freq=freq,
    #                 transaction_cost=tc,
    #             )
    #
    # summary_table = pd.DataFrame({
    #     key: result["summary"] for key, result in all_results.items()
    # }).T
    # print(summary_table)
    #
    # # Example: compare Alpha NAV across different N values under one setting.
    # alpha_nav_by_n = {
    #     f"N={n}": all_results[f"N={n}, freq=M, cost=0.001"]["alpha_nav"]
    #     for n in top_n_list
    # }
    # plot_nav_comparison(alpha_nav_by_n, title="Alpha NAV Comparison across N")

    print("Framework loaded successfully.")
    print("Please complete the TODO functions for Assignment 5.")

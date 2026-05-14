# -*- coding: utf-8 -*-
"""
Assignment 5 solution: Candidate stock selection, rebalance, and alpha portfolio
===============================================================================

This is the reference solution file.  It completes the functions in the student
framework for Assignment 5.

Assignment focus
----------------
1. Build a multi-factor pool and combine factors by Rank Aggregation.
2. Construct the investable stock universe.
3. Apply PE lowest 30% filtering.  Do NOT use PB as an additional candidate
   selection filter in this revised version.
4. Apply Top-N selection for N in {10, 20, 30, 50, 100}.
5. Build equal-weight long-only portfolios.
6. Load benchmark data and compute alpha returns.
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


def load_data(startdate: str, enddate: str) -> dict[str, pd.DataFrame]:
    """
    Load Shanghai and Shenzhen market data and align all fields.

    Parameters
    ----------
    startdate : str
        Start date in a format accepted by ``pd.to_datetime``.
    enddate : str
        End date in a format accepted by ``pd.to_datetime``.

    Returns
    -------
    dict[str, pd.DataFrame]
        A dictionary containing close, pb, pe_ttm, total_mv, amount, and adj_factor.
    """
    dataset: dict[str, pd.DataFrame] = {}
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)

    for field in FIELDS:
        sh = read_one(field, "sh")
        sz = read_one(field, "sz")
        merged = pd.concat([sh, sz], axis=1).sort_index()
        dataset[field] = merged.loc[(merged.index >= start) & (merged.index <= end)]
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


def load_benchmark(startdate: str, enddate: str) -> pd.DataFrame:
    """
    Load benchmark data from DATA_DIR/benchmark.csv and keep the requested period.
    """
    benchmark = read_one("benchmark", None)
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    benchmark = benchmark.loc[(benchmark.index >= start) & (benchmark.index <= end)]
    return benchmark


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
# 3. Factor-test utilities for direction inference
# ============================================================

def _safe_pearson(x: pd.Series, y: pd.Series, min_obs: int = 10) -> float:
    """
    Pearson correlation after dropping missing values.
    """
    tmp = pd.concat([x, y], axis=1).dropna()

    if tmp.shape[0] < min_obs:
        return np.nan

    if tmp.iloc[:, 0].std(ddof=0) == 0 or tmp.iloc[:, 1].std(ddof=0) == 0:
        return np.nan

    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="pearson")


def _safe_spearman(x: pd.Series, y: pd.Series, min_obs: int = 10) -> float:
    """
    Spearman rank correlation after dropping missing values.
    """
    tmp = pd.concat([x, y], axis=1).dropna()

    if tmp.shape[0] < min_obs:
        return np.nan

    if tmp.iloc[:, 0].nunique() < 2 or tmp.iloc[:, 1].nunique() < 2:
        return np.nan

    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1], method="spearman")


def _safe_ir(x: pd.Series) -> float:
    """
    Mean divided by standard deviation.
    """
    x = x.dropna()

    if len(x) == 0:
        return np.nan

    sd = x.std(ddof=1)

    if sd == 0 or np.isnan(sd):
        return np.nan

    return x.mean() / sd


def _split_sorted_assets(f: pd.Series, num_groups: int) -> list[pd.Index]:
    """
    Sort assets by factor value and split them into groups.
    """
    sorted_assets = f.sort_values(ascending=True).index.to_numpy()
    groups = np.array_split(sorted_assets, num_groups)

    return [pd.Index(g) for g in groups]


def factor_test(
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    num_groups: int = 5,
    comsn: float = 0.0,
    min_obs: int = 10,
    name: str = "factor",
    plot: bool = True,
    display: bool = True,
) -> dict[str, object]:
    """
    Test a cross-sectional factor by IC, RankIC, grouped returns, and long-short NAV.

    Important time alignment
    ------------------------
    At date t, the factor value F_t is matched with next-period return R_{t+1}.
    Therefore, the last factor row cannot be used because R_{T+1} is unavailable.
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame.")

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    common_index = factors.index.intersection(returns.index)
    common_columns = factors.columns.intersection(returns.columns)

    factors = factors.reindex(index=common_index, columns=common_columns).sort_index()
    returns = returns.reindex(index=common_index, columns=common_columns).sort_index()

    dates = factors.index

    if len(dates) < 2:
        raise ValueError("At least two dates are required for factor testing.")

    test_dates = dates[:-1]

    ic = pd.Series(index=test_dates, dtype=float, name="IC")
    rank_ic_s = pd.Series(index=test_dates, dtype=float, name="RankIC")

    group_returns = pd.DataFrame(
        index=test_dates,
        columns=[f"G{q + 1}" for q in range(num_groups)],
        dtype=float,
    )

    turnover_low = pd.Series(index=test_dates, dtype=float, name="turnover_low")
    turnover_high = pd.Series(index=test_dates, dtype=float, name="turnover_high")

    prev_low_group: set[str] | None = None
    prev_high_group: set[str] | None = None

    for pos, t in enumerate(test_dates):
        next_t = dates[pos + 1]

        f = factors.loc[t]
        r_next = returns.loc[next_t]

        valid = f.notna() & r_next.notna()
        f_valid = f.loc[valid]
        r_valid = r_next.loc[valid]

        if len(f_valid) < max(min_obs, num_groups):
            continue

        ic.loc[t] = _safe_pearson(f_valid, r_valid, min_obs=min_obs)
        rank_ic_s.loc[t] = _safe_spearman(f_valid, r_valid, min_obs=min_obs)

        groups = _split_sorted_assets(f_valid, num_groups=num_groups)

        for q, group_assets in enumerate(groups):
            if len(group_assets) == 0:
                group_returns.loc[t, f"G{q + 1}"] = np.nan
            else:
                group_returns.loc[t, f"G{q + 1}"] = r_valid.loc[group_assets].mean()

        low_group = set(groups[0])
        high_group = set(groups[-1])

        if prev_low_group is None:
            turnover_low.loc[t] = 1.0
        elif len(low_group) == 0:
            turnover_low.loc[t] = np.nan
        else:
            turnover_low.loc[t] = 1.0 - len(low_group & prev_low_group) / len(low_group)

        if prev_high_group is None:
            turnover_high.loc[t] = 1.0
        elif len(high_group) == 0:
            turnover_high.loc[t] = np.nan
        else:
            turnover_high.loc[t] = 1.0 - len(high_group & prev_high_group) / len(high_group)

        prev_low_group = low_group
        prev_high_group = high_group

    avg_group_return = group_returns.mean(axis=0, skipna=True)

    if avg_group_return.iloc[-1] >= avg_group_return.iloc[0]:
        long_short_direction = f"G{num_groups} - G1"
        gross_long_short = group_returns.iloc[:, -1] - group_returns.iloc[:, 0]
        excessive = group_returns.iloc[:, -1] - group_returns.mean(axis=1)
    else:
        long_short_direction = f"G1 - G{num_groups}"
        gross_long_short = group_returns.iloc[:, 0] - group_returns.iloc[:, -1]
        excessive = group_returns.iloc[:, 0] - group_returns.mean(axis=1)

    cost_adjustment = 0.5 * comsn * (turnover_low + turnover_high)
    long_short = gross_long_short - cost_adjustment
    long_short.name = "LongShort"

    excessive = excessive - cost_adjustment
    excessive.name = "Excessive"

    group_nav = (1.0 + group_returns.fillna(0.0)).cumprod()
    long_short_nav = (1.0 + long_short.fillna(0.0)).cumprod()
    excessive_nav = (1.0 + excessive.fillna(0.0)).cumprod()

    ic_stats = pd.Series(
        {
            "IC_mean": ic.mean(skipna=True),
            "IC_std": ic.std(skipna=True, ddof=1),
            "ICIR": _safe_ir(ic),
            "IC_win_rate(%)": (ic.dropna() > 0).mean() * 100,
            "RankIC_mean": rank_ic_s.mean(skipna=True),
            "RankIC_std": rank_ic_s.std(skipna=True, ddof=1),
            "RankICIR": _safe_ir(rank_ic_s),
            "RankIC_win_rate(%)": (rank_ic_s.dropna() > 0).mean() * 100,
            "LongShort_mean(%)": long_short.mean(skipna=True) * 100,
            "LongShort_std(%)": long_short.std(skipna=True, ddof=1) * 100,
            "LongShort_final_nav": long_short_nav.iloc[-1],
            "Excessive_mean(%)": excessive.mean(skipna=True) * 100,
            "Excessive_final_nav": excessive_nav.iloc[-1],
        }
    )

    result = {
        "name": name,
        "ic": ic,
        "rank_ic": rank_ic_s,
        "ic_stats": ic_stats,
        "group_returns": group_returns,
        "avg_group_return": avg_group_return,
        "group_nav": group_nav,
        "long_short": long_short,
        "long_short_nav": long_short_nav,
        "long_short_direction": long_short_direction,
        "excessive": excessive,
        "excessive_nav": excessive_nav,
        "turnover_low": turnover_low,
        "turnover_high": turnover_high,
    }

    if display:
        print("=" * 70)
        print(f"Factor test result: {name}")
        print("-" * 70)
        print(ic_stats)
        print("-" * 70)
        print("Average group returns:")
        print(avg_group_return)
        print(f"Long-short direction: {long_short_direction}")
        print("=" * 70)

    if plot:
        plot_factor_test_result(result)

    return result


def plot_factor_test_result(result: dict[str, object]) -> None:
    """
    Plot IC, RankIC, average group return, and long-short NAV.
    """
    name = result["name"]
    ic = result["ic"]
    rank_ic_s = result["rank_ic"]
    avg_group_return = result["avg_group_return"]
    long_short_nav = result["long_short_nav"]
    group_nav = result["group_nav"]
    long_short_direction = result["long_short_direction"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Factor Test: {name}", fontsize=16)

    axes[0, 0].bar(rank_ic_s.index, rank_ic_s.values, width=5)
    axes[0, 0].axhline(rank_ic_s.mean(skipna=True), linestyle="--", linewidth=1.5)
    axes[0, 0].axhline(0.0, linestyle="-", linewidth=1)
    axes[0, 0].set_title("RankIC")
    axes[0, 0].set_ylabel("RankIC")

    axes[0, 1].bar(ic.index, ic.values, width=5)
    axes[0, 1].axhline(ic.mean(skipna=True), linestyle="--", linewidth=1.5)
    axes[0, 1].axhline(0.0, linestyle="-", linewidth=1)
    axes[0, 1].set_title("Normal IC")
    axes[0, 1].set_ylabel("IC")

    x = np.arange(1, len(avg_group_return) + 1)
    axes[1, 0].bar(x, avg_group_return.values)
    axes[1, 0].axhline(0.0, linestyle="-", linewidth=1)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(avg_group_return.index)
    axes[1, 0].set_title("Average Group Return")
    axes[1, 0].set_ylabel("Average return")

    axes[1, 1].plot(long_short_nav.index, long_short_nav.values, label=long_short_direction)
    axes[1, 1].axhline(1.0, linestyle="--", linewidth=1)
    axes[1, 1].set_title("Long-Short NAV")
    axes[1, 1].set_ylabel("NAV")
    axes[1, 1].legend(loc="best")

    for ax in axes.ravel():
        ax.grid(linestyle=":", alpha=0.6)
        if isinstance(ic.index, pd.DatetimeIndex):
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for col in group_nav.columns:
        plt.plot(group_nav.index, group_nav[col], label=col)

    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.title(f"Grouped NAV: {name}")
    plt.ylabel("NAV")
    plt.legend(loc="best")
    plt.grid(linestyle=":", alpha=0.6)

    ax = plt.gca()
    if isinstance(group_nav.index, pd.DatetimeIndex):
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.show()


def infer_factor_directions(
    factor_dict: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    min_obs: int = 30,
) -> tuple[dict[str, int], pd.Series]:
    """
    Infer factor directions from the sign of mean IC.
    """
    directions: dict[str, int] = {}
    ic_mean_dict: dict[str, float] = {}

    for factor_name, factor_matrix in factor_dict.items():
        result = factor_test(
            factor_matrix,
            returns,
            min_obs=min_obs,
            name=factor_name,
            plot=False,
            display=False,
        )
        ic_mean = float(result["ic_stats"]["RankIC_mean"])
        if np.isnan(ic_mean):
            raise ValueError(f"RankIC_mean for factor {factor_name} is NaN; cannot infer direction.")

        directions[factor_name] = +1 if ic_mean >= 0 else -1
        ic_mean_dict[factor_name] = ic_mean

    return directions, pd.Series(ic_mean_dict, name="IC_mean")


# ============================================================
# 4. Assignment 5 solution functions
# ============================================================

def rank_aggregation(
    factor_dict: dict[str, pd.DataFrame],
    directions: dict[str, int],
    weights: dict[str, float] | None = None,
    min_non_missing: int = 2,
) -> pd.DataFrame:
    """
    Combine multiple factor matrices into one composite score matrix.

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
    if not factor_dict:
        raise ValueError("factor_dict must contain at least one factor matrix.")

    missing_dirs = set(factor_dict) - set(directions)
    if missing_dirs:
        raise ValueError(f"Missing factor directions for: {sorted(missing_dirs)}")

    # Align all factors by common dates and common assets.
    common_index = None
    common_columns = None
    for fac in factor_dict.values():
        if common_index is None:
            common_index = fac.index
            common_columns = fac.columns
        else:
            common_index = common_index.intersection(fac.index)
            common_columns = common_columns.intersection(fac.columns)

    assert common_index is not None and common_columns is not None
    common_index = common_index.sort_values()
    common_columns = common_columns.sort_values()

    if weights is None:
        weights = {name: 1.0 for name in factor_dict}

    # Keep only factors with positive weights.
    used_factors = [name for name in factor_dict if weights.get(name, 0.0) > 0]
    if not used_factors:
        raise ValueError("At least one factor must have a positive weight.")

    numerator = pd.DataFrame(0.0, index=common_index, columns=common_columns)
    denominator = pd.DataFrame(0.0, index=common_index, columns=common_columns)
    non_missing_count = pd.DataFrame(0, index=common_index, columns=common_columns)

    for name in used_factors:
        direction = directions[name]
        if direction not in (+1, -1):
            raise ValueError(f"Direction for {name} must be +1 or -1.")

        fac = factor_dict[name].reindex(index=common_index, columns=common_columns)

        # Convert each factor into cross-sectional percentile ranks.
        # After multiplying by direction, larger transformed values are better.
        transformed = fac * direction
        rank_score = transformed.rank(axis=1, method="average", pct=True, ascending=True)

        w = float(weights.get(name, 0.0))
        valid = rank_score.notna()
        numerator = numerator.add(rank_score.fillna(0.0) * w, fill_value=0.0)
        denominator = denominator.add(valid.astype(float) * w, fill_value=0.0)
        non_missing_count = non_missing_count.add(valid.astype(int), fill_value=0).astype(int)

    composite = numerator / denominator.replace(0.0, np.nan)
    composite = composite.mask(non_missing_count < min_non_missing)
    return composite


def build_investable_universe(
    score: pd.DataFrame,
    returns: pd.DataFrame,
    pe: pd.DataFrame,
    amount: pd.DataFrame | None = None,
    min_amount: float | None = None,
) -> pd.DataFrame:
    """
    Construct the investable stock universe mask.

    Parameters
    ----------
    score : pd.DataFrame
        Composite score matrix.
    returns : pd.DataFrame
        Return matrix.  At date t, the user should ensure that the next-period
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
    common_index = score.index.intersection(returns.index).intersection(pe.index).sort_values()
    common_columns = score.columns.intersection(returns.columns).intersection(pe.columns).sort_values()

    score_a = score.reindex(index=common_index, columns=common_columns)
    returns_a = returns.reindex(index=common_index, columns=common_columns)
    pe_a = pe.reindex(index=common_index, columns=common_columns)

    # At signal date t, require next-period return to be available.
    next_ret_available = returns_a.shift(-1).notna()

    mask = score_a.notna() & pe_a.notna() & (pe_a > 0) & next_ret_available

    if amount is not None:
        amount_a = amount.reindex(index=common_index, columns=common_columns)
        mask = mask & amount_a.notna()
        if min_amount is not None:
            mask = mask & (amount_a >= min_amount)

    return mask.astype(bool)


def pe_quantile_filter(
    pe: pd.DataFrame,
    investable_mask: pd.DataFrame,
    quantile: float = 0.30,
) -> pd.DataFrame:
    """
    Keep stocks with PE in the lowest given quantile.

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
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1].")

    pe_a = pe.reindex_like(investable_mask)
    investable = investable_mask.astype(bool)

    out = pd.DataFrame(False, index=investable.index, columns=investable.columns)

    for t in investable.index:
        valid = investable.loc[t] & pe_a.loc[t].notna() & (pe_a.loc[t] > 0)
        pe_valid = pe_a.loc[t, valid]

        if pe_valid.empty:
            continue

        cutoff = pe_valid.quantile(quantile)
        out.loc[t, valid] = pe_valid <= cutoff

    return out.astype(bool)


def select_top_n(
    score: pd.DataFrame,
    candidate_mask: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    Select the Top-N stocks from the PE-filtered candidate universe.

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
    if n <= 0:
        raise ValueError("n must be positive.")

    score_a = score.reindex_like(candidate_mask)
    candidate = candidate_mask.astype(bool)
    out = pd.DataFrame(False, index=candidate.index, columns=candidate.columns)

    for t in candidate.index:
        valid = candidate.loc[t] & score_a.loc[t].notna()
        scores = score_a.loc[t, valid]
        if scores.empty:
            continue

        selected = scores.nlargest(min(n, len(scores))).index
        out.loc[t, selected] = True

    return out.astype(bool)


def build_equal_weight_weights(selection_mask: pd.DataFrame) -> pd.DataFrame:
    """
    Convert selected stocks into equal-weight long-only portfolio weights.

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
    mask = selection_mask.astype(bool)
    count = mask.sum(axis=1).replace(0, np.nan)
    weights = mask.astype(float).div(count, axis=0).fillna(0.0)
    return weights


def compute_benchmark_returns(
    benchmark: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convert benchmark close prices into weekly returns aligned to target_index.
    """
    if "close" not in benchmark.columns:
        raise ValueError("benchmark data must contain a 'close' column.")

    idx = pd.DatetimeIndex(target_index).sort_values()
    close = benchmark["close"].copy().sort_index()
    close = close.resample("W").last()
    close = close.reindex(idx).ffill()

    benchmark_returns = close.pct_change().fillna(0.0)
    benchmark_returns.name = "benchmark_return"
    return benchmark_returns


def get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    Choose rebalance dates from a trading-date index.

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
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    idx = index.dropna().sort_values().unique()
    freq = freq.upper()

    if freq in {"D", "DAILY"}:
        return pd.DatetimeIndex(idx)

    s = pd.Series(pd.DatetimeIndex(idx), index=pd.DatetimeIndex(idx))

    if freq in {"W", "WEEKLY"}:
        return pd.DatetimeIndex(s.groupby(s.index.to_period("W")).max().values)

    if freq in {"M", "MONTHLY"}:
        return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)

    raise ValueError("freq must be one of 'D', 'W', or 'M'.")


def apply_rebalance_schedule(weights: pd.DataFrame, rebalance_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """
    Convert rebalance-date weights into daily/weekly holding weights.

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
    rb_dates = pd.DatetimeIndex(rebalance_dates)
    rb_dates = rb_dates.intersection(weights.index).sort_values()

    scheduled = weights.copy() * np.nan
    if len(rb_dates) > 0:
        scheduled.loc[rb_dates] = weights.loc[rb_dates]

    # Before the first rebalance date, the portfolio has no position.
    holding_weights = scheduled.ffill().fillna(0.0)
    return holding_weights


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute one-way portfolio turnover.

    A common definition is
        turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|.

    Returns
    -------
    pd.Series
        Turnover series indexed by date.
    """
    w = weights.fillna(0.0)
    prev = w.shift(1).fillna(0.0)
    turnover = 0.5 * (w - prev).abs().sum(axis=1)
    turnover.name = "turnover"
    return turnover


def compute_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """
    Compute net portfolio returns from weights and asset returns.

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
    common_index = weights.index.intersection(returns.index).sort_values()
    common_columns = weights.columns.intersection(returns.columns).sort_values()

    w = weights.reindex(index=common_index, columns=common_columns).fillna(0.0)
    r = returns.reindex(index=common_index, columns=common_columns).fillna(0.0)

    # Use weights formed at t-1 to earn returns over period t.
    holding = w.shift(1).fillna(0.0)
    gross = (holding * r).sum(axis=1)
    gross.name = "gross_return"

    # Cost paid when the new target portfolio is formed at the previous date.
    turnover = compute_turnover(w).shift(1).fillna(0.0)
    net = gross - transaction_cost * turnover
    net.name = "net_return"
    return net


def compute_nav(portfolio_returns: pd.Series) -> pd.Series:
    """
    Compute cumulative NAV from a return series.
    """
    ret = portfolio_returns.fillna(0.0)
    nav = (1.0 + ret).cumprod()
    nav.name = "NAV"
    return nav


def compute_alpha_returns(
    long_returns_net: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """
    Compute net alpha return.

    Simplified assignment convention
    --------------------------------
    net alpha return = net long-only return - benchmark return.

    This avoids double-counting benchmark-side transaction costs unless the user
    explicitly chooses a different convention and explain it in the report.
    """
    common_index = long_returns_net.index.intersection(benchmark_returns.index).sort_values()
    alpha = long_returns_net.reindex(common_index).fillna(0.0) - benchmark_returns.reindex(common_index).fillna(0.0)
    alpha.name = "alpha_return_net"
    return alpha


def performance_summary(ret: pd.Series, nav: pd.Series | None = None, periods_per_year: int = 52) -> pd.Series:
    """
    Report main performance statistics.

    Suggested metrics
    -----------------
    - annualized return
    - annualized volatility
    - Sharpe ratio
    - maximum drawdown
    - final NAV
    """
    ret = ret.dropna()
    if nav is None:
        nav = compute_nav(ret)
    else:
        nav = nav.reindex(ret.index).dropna()

    if len(ret) == 0 or len(nav) == 0:
        return pd.Series(
            {
                "ann_return": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "final_nav": np.nan,
            }
        )

    final_nav = float(nav.iloc[-1])
    n_periods = len(ret)

    if final_nav > 0 and n_periods > 0:
        ann_return = final_nav ** (periods_per_year / n_periods) - 1.0
    else:
        ann_return = np.nan

    ann_vol = ret.std(ddof=1) * np.sqrt(periods_per_year) if len(ret) > 1 else np.nan
    sd = ret.std(ddof=1)
    sharpe = ret.mean() / sd * np.sqrt(periods_per_year) if (sd is not None and sd > 0 and not np.isnan(sd)) else np.nan

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = drawdown.min()

    return pd.Series(
        {
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "final_nav": final_nav,
        }
    )


def run_single_backtest(
    score: pd.DataFrame,
    pe: pd.DataFrame,
    returns: pd.DataFrame,
    amount: pd.DataFrame | None,
    benchmark: pd.DataFrame,
    n: int,
    rebalance_freq: str,
    transaction_cost: float,
) -> dict[str, object]:
    """
    Run one complete configuration of the Assignment 5 backtest.

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
    benchmark : pd.DataFrame
        Benchmark price table loaded from benchmark.csv.
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
        - 'alpha_weights' (None when benchmark constituent weights are unavailable)
        - 'long_returns_net'
        - 'benchmark_returns'
        - 'alpha_returns_net'
        - 'long_nav'
        - 'benchmark_nav'
        - 'alpha_nav'
        - 'turnover'
        - 'summary'
    """
    # 1. Build investable universe.
    investable = build_investable_universe(
        score=score,
        returns=returns,
        pe=pe,
        amount=amount,
        min_amount=None,
    )

    # 2. PE lowest 30% filter.
    pe_mask = pe_quantile_filter(pe=pe, investable_mask=investable, quantile=0.30)

    # 3. Top-N selection.
    selection = select_top_n(score=score, candidate_mask=pe_mask, n=n)

    # 4. Long-only weights on all signal dates.
    long_weights_target = build_equal_weight_weights(selection)

    # 5. Apply rebalance schedule.
    rb_dates = get_rebalance_dates(long_weights_target.index, rebalance_freq)
    long_weights = apply_rebalance_schedule(long_weights_target, rb_dates)
    alpha_weights = None

    # 6. Returns and turnover.
    long_returns_net = compute_portfolio_returns(long_weights, returns, transaction_cost=transaction_cost)
    benchmark_returns = compute_benchmark_returns(benchmark, long_returns_net.index)
    alpha_returns_net = compute_alpha_returns(long_returns_net, benchmark_returns)

    turnover = compute_turnover(long_weights)

    # 7. NAV and summary.
    long_nav = compute_nav(long_returns_net)
    benchmark_nav = compute_nav(benchmark_returns)
    alpha_nav = compute_nav(alpha_returns_net)

    long_summary = performance_summary(long_returns_net, long_nav, periods_per_year=52)
    benchmark_summary = performance_summary(benchmark_returns, benchmark_nav, periods_per_year=52)
    alpha_summary = performance_summary(alpha_returns_net, alpha_nav, periods_per_year=52)

    summary = pd.concat(
        {
            "long_only": long_summary,
            "benchmark": benchmark_summary,
            "alpha": alpha_summary,
        },
        axis=0,
    )
    summary.loc[("long_only", "avg_turnover")] = turnover.mean(skipna=True)
    summary.loc[("long_only", "total_turnover")] = turnover.sum(skipna=True)

    return {
        "investable_mask": investable,
        "pe_mask": pe_mask,
        "selection_mask": selection,
        "rebalance_dates": rb_dates,
        "long_weights": long_weights,
        "alpha_weights": alpha_weights,
        "long_returns_net": long_returns_net,
        "benchmark_returns": benchmark_returns,
        "alpha_returns_net": alpha_returns_net,
        "long_nav": long_nav,
        "benchmark_nav": benchmark_nav,
        "alpha_nav": alpha_nav,
        "turnover": turnover,
        "summary": summary,
    }


def plot_nav_comparison(nav_dict: dict[str, pd.Series], title: str) -> None:
    """
    Plot several NAV curves in one figure.

    This plotting function is provided for convenience and can be used after the
    solution backtest functions are completed.
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
# %% 5. Main experiment skeleton
# ============================================================

if __name__ == "__main__":
    startdate = "2020-01-02"
    enddate = "2024-12-31"# alternative: "2025-12-26"

    dataset = load_data(startdate=startdate, enddate=enddate)
    features = build_weekly_features(dataset)

    returns = features["ret_w"]
    pe_raw = features["pe_raw_w"]
    amount = features["amount20_w"]
    benchmark = load_benchmark(
        startdate=startdate,
        enddate=enddate,  
    )

    raw_factor_dict = {
        "size": features["size_w"],
        "value": features["value_w"],
        "momentum": features["mom_w"],
        "volatility": features["vol_w"],
    }

    # Students may choose whether to use industry-neutralized factors.
    neutralizer = IndustryNeutral(ind="l1_code")
    factor_dict = neutralizer(raw_factor_dict)

    factor_directions, factor_ic_mean = infer_factor_directions(
        factor_dict,
        returns,
        min_obs=30,
    )
    print("\nInferred factor directions from IC_mean:")
    print(pd.DataFrame({"IC_mean": factor_ic_mean, "direction": pd.Series(factor_directions)}))

    top_n_list = [10, 20, 30, 50, 100]
    rebalance_freq_list = ["M"]
    transaction_cost_list = [ 0.002]

    composite_score = rank_aggregation(
        factor_dict=factor_dict,
        directions=factor_directions,
        weights=None,
        min_non_missing=2,
    )

    all_results = {}
    for n in top_n_list:
        for freq in rebalance_freq_list:
            for tc in transaction_cost_list:
                key = f"N={n}, freq={freq}, cost={tc}"
                print(f"Running {key} ...")
                all_results[key] = run_single_backtest(
                    score=composite_score,
                    pe=pe_raw,
                    returns=returns,
                    amount=amount,
                    benchmark=benchmark,
                    n=n,
                    rebalance_freq=freq,
                    transaction_cost=tc,
                )

    summary_table = pd.DataFrame({
        key: result["summary"] for key, result in all_results.items()
    }).T

    print("\nAssignment 5 solution summary table:")
    print(summary_table)

    # Example: compare Long-Only NAV across different N values under one setting.
    long_only_nav_by_n = {
        f"N={n}": all_results[f"N={n}, freq=M, cost=0.002"]["long_nav"]
        for n in top_n_list
    }
    plot_nav_comparison(long_only_nav_by_n, title="Long-Only NAV Comparison across N")

    # Example: compare Alpha NAV across different N values under one setting.
    alpha_nav_by_n = {
        f"N={n}": all_results[f"N={n}, freq=M, cost=0.002"]["alpha_nav"]
        for n in top_n_list
    }
    plot_nav_comparison(alpha_nav_by_n, title="Alpha NAV Comparison across N")

    sharpe_comparison = pd.DataFrame(
        {
            "long_only": {
                f"N={n}": all_results[f"N={n}, freq=M, cost=0.002"]["summary"].loc[("long_only", "sharpe")]
                for n in top_n_list
            },
            "alpha": {
                f"N={n}": all_results[f"N={n}, freq=M, cost=0.002"]["summary"].loc[("alpha", "sharpe")]
                for n in top_n_list
            },
        }
    )

    print("\nSharpe ratio comparison:")
    print(sharpe_comparison)

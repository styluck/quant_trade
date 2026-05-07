# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:19:21 2026

@author: lich5

Reference solution for the Factor Testing assignment.

This script reuses the data-loading and weekly feature-construction pipeline
from the previous risk-parity assignment, and reorganizes the old factor_test
function into a cleaner version.

Main tasks
----------
1. Load and clean the required CSV files.
2. Build weekly style factors: size, value, momentum, volatility.
3. Choose one factor matrix F and one return matrix R.
4. Evaluate the factor by:
   - Normal IC
   - RankIC
   - grouped return analysis
   - long-short group return
   - long-short NAV
5. Compare several factors in the main section.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. Data loading and feature construction
# ============================================================

DATA_DIR = Path(__file__).resolve().parent / "dataset"
FIELDS = ["close", "pb", "total_mv", "adj_factor"]


def read_one(field: str, market: str | None) -> pd.DataFrame:
    """
    Read one CSV file.

    Parameters
    ----------
    field : str
        Field name, such as 'close', 'pb', 'total_mv', 'adj_factor'.
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
        A dictionary containing close, pb, total_mv, and adj_factor.
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
        dataset[field] = dataset[field].reindex(
            index=common_index,
            columns=common_columns,
        )

    # Remove invalid observations.
    dataset["close"] = dataset["close"].mask(dataset["close"] <= 0)
    dataset["adj_factor"] = dataset["adj_factor"].mask(dataset["adj_factor"] <= 0)
    dataset["pb"] = dataset["pb"].mask(dataset["pb"] <= 0)
    dataset["total_mv"] = dataset["total_mv"].mask(dataset["total_mv"] <= 0)

    return dataset


def winsorize_row(row: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """
    Cross-sectional winsorization for one date.
    """
    valid = row.dropna()

    if valid.empty:
        return row

    lo = valid.quantile(lower)
    hi = valid.quantile(upper)

    return row.clip(lower=lo, upper=hi)


def zscore_row(row: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score standardization for one date.
    """
    valid = row.dropna()

    if valid.shape[0] < 2:
        return row * np.nan

    mu = valid.mean()
    sd = valid.std(ddof=0)

    if sd == 0 or np.isnan(sd):
        return row * np.nan

    return (row - mu) / sd


def preprocess_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize and standardize factor exposures cross-sectionally.
    """
    out = df.copy()
    out = out.apply(winsorize_row, axis=1)
    out = out.apply(zscore_row, axis=1)
    return out


def build_weekly_features(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build weekly return and style factor matrices.

    Returns
    -------
    dict[str, pd.DataFrame]
        Weekly close, weekly return, and weekly factor matrices.
    """
    close = dataset["close"] * dataset["adj_factor"]
    pb = dataset["pb"]
    total_mv = dataset["total_mv"]

    daily_ret = close.pct_change()

    mom20 = close / close.shift(20) - 1
    value = (1.0 / pb).replace([np.inf, -np.inf], np.nan)
    size = -np.log(total_mv)
    vol20 = daily_ret.rolling(20).std()

    weekly = {
        "close_w": close.resample("W").last(),
        "ret_w": close.resample("W").last().pct_change(),
        "size_w": size.resample("W").last(),
        "value_w": value.resample("W").last(),
        "mom_w": mom20.resample("W").last(),
        "vol_w": vol20.resample("W").last(),
    }

    weekly["size_w"] = preprocess_exposure(weekly["size_w"])
    weekly["value_w"] = preprocess_exposure(weekly["value_w"])
    weekly["mom_w"] = preprocess_exposure(weekly["mom_w"])
    weekly["vol_w"] = preprocess_exposure(weekly["vol_w"])

    return weekly


# ============================================================
# 2. Factor-test utilities
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

    Parameters
    ----------
    f : pd.Series
        One-date factor cross section.
    num_groups : int
        Number of groups.

    Returns
    -------
    list[pd.Index]
        Asset indices for each group.
    """
    sorted_assets = f.sort_values(ascending=True).index.to_numpy()
    groups = np.array_split(sorted_assets, num_groups)

    return [pd.Index(g) for g in groups]


# ============================================================
# 3. Clean factor_test function
# ============================================================

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

    Parameters
    ----------
    factors : pd.DataFrame
        Factor matrix indexed by date and columns by assets.
    returns : pd.DataFrame
        Return matrix indexed by date and columns by assets.
    num_groups : int, default=5
        Number of factor-sorted groups.
    comsn : float, default=0.0
        One-way transaction cost parameter used in long-short spread adjustment.
    min_obs : int, default=10
        Minimum number of valid stocks in one cross section.
    name : str, default='factor'
        Factor name.
    plot : bool, default=True
        Whether to plot diagnostic figures.
    display : bool, default=True
        Whether to print summary statistics.

    Returns
    -------
    dict[str, object]
        A dictionary containing:
        - 'ic'
        - 'rank_ic'
        - 'ic_stats'
        - 'group_returns'
        - 'avg_group_return'
        - 'group_nav'
        - 'long_short'
        - 'long_short_nav'
        - 'long_short_direction'
        - 'turnover_low'
        - 'turnover_high'
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame.")

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    # Align dates and assets.
    common_index = factors.index.intersection(returns.index)
    common_columns = factors.columns.intersection(returns.columns)

    factors = factors.reindex(index=common_index, columns=common_columns).sort_index()
    returns = returns.reindex(index=common_index, columns=common_columns).sort_index()

    dates = factors.index

    if len(dates) < 2:
        raise ValueError("At least two dates are required for factor testing.")

    # Use factor dates t = 0, ..., T-2.
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

        # IC and RankIC.
        ic.loc[t] = _safe_pearson(f_valid, r_valid, min_obs=min_obs)
        rank_ic_s.loc[t] = _safe_spearman(f_valid, r_valid, min_obs=min_obs)

        # Grouped return.
        groups = _split_sorted_assets(f_valid, num_groups=num_groups)

        for q, group_assets in enumerate(groups):
            if len(group_assets) == 0:
                group_returns.loc[t, f"G{q + 1}"] = np.nan
            else:
                group_returns.loc[t, f"G{q + 1}"] = r_valid.loc[group_assets].mean()

        # Turnover for extreme groups.
        low_group = set(groups[0])
        high_group = set(groups[-1])

        if prev_low_group is None:
            turnover_low.loc[t] = 1.0
        else:
            if len(low_group) == 0:
                turnover_low.loc[t] = np.nan
            else:
                turnover_low.loc[t] = 1.0 - len(low_group & prev_low_group) / len(low_group)

        if prev_high_group is None:
            turnover_high.loc[t] = 1.0
        else:
            if len(high_group) == 0:
                turnover_high.loc[t] = np.nan
            else:
                turnover_high.loc[t] = 1.0 - len(high_group & prev_high_group) / len(high_group)

        prev_low_group = low_group
        prev_high_group = high_group

    # Average grouped return.
    avg_group_return = group_returns.mean(axis=0, skipna=True)

    # Decide long-short direction according to average group returns.
    # If high-factor group outperforms low-factor group, use GQ - G1.
    # Otherwise use G1 - GQ.
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

    # NAV.
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


# ============================================================
# 4. Plotting
# ============================================================

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

    # RankIC.
    axes[0, 0].bar(rank_ic_s.index, rank_ic_s.values, width=5)
    axes[0, 0].axhline(rank_ic_s.mean(skipna=True), linestyle="--", linewidth=1.5)
    axes[0, 0].axhline(0.0, linestyle="-", linewidth=1)
    axes[0, 0].set_title("RankIC")
    axes[0, 0].set_ylabel("RankIC")

    # Normal IC.
    axes[0, 1].bar(ic.index, ic.values, width=5)
    axes[0, 1].axhline(ic.mean(skipna=True), linestyle="--", linewidth=1.5)
    axes[0, 1].axhline(0.0, linestyle="-", linewidth=1)
    axes[0, 1].set_title("Normal IC")
    axes[0, 1].set_ylabel("IC")

    # Average group return.
    x = np.arange(1, len(avg_group_return) + 1)
    axes[1, 0].bar(x, avg_group_return.values)
    axes[1, 0].axhline(0.0, linestyle="-", linewidth=1)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(avg_group_return.index)
    axes[1, 0].set_title("Average Group Return")
    axes[1, 0].set_ylabel("Average return")

    # Long-short NAV.
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

    # Plot grouped NAV separately.
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


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    dataset = load_data()
    features = build_weekly_features(dataset)

    returns = features["ret_w"]

    factor_dict = {
        "size": features["size_w"],
        "value": features["value_w"],
        "momentum": features["mom_w"],
        "volatility": features["vol_w"],
    }

    num_groups = 5
    comsn = 0.001
    min_obs = 30

    all_stats = {}

    # Run detailed factor test for each factor.
    # To avoid producing too many figures, only plot one representative factor.
    plot_factor_name = "value"

    for factor_name, factor_matrix in factor_dict.items():
        result = factor_test(
            factor_matrix,
            returns,
            num_groups=num_groups,
            comsn=comsn,
            min_obs=min_obs,
            name=factor_name,
            plot=(factor_name == plot_factor_name),
            display=True,
        )

        all_stats[factor_name] = result["ic_stats"]

    summary_table = pd.DataFrame(all_stats).T

    print("\nFactor testing assignment solution")
    print("\nSummary table:")
    print(summary_table)

    # Optional: compare average group returns of all factors.
    # Students may comment this part out if they only need one factor.
    plt.figure(figsize=(10, 5))

    for factor_name, factor_matrix in factor_dict.items():
        result = factor_test(
            factor_matrix,
            returns,
            num_groups=num_groups,
            comsn=comsn,
            min_obs=min_obs,
            name=factor_name,
            plot=False,
            display=False,
        )
        avg_group_return = result["avg_group_return"]
        plt.plot(
            np.arange(1, num_groups + 1),
            avg_group_return.values,
            marker="o",
            label=factor_name,
        )

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(np.arange(1, num_groups + 1), [f"G{i}" for i in range(1, num_groups + 1)])
    plt.xlabel("Factor group")
    plt.ylabel("Average next-period return")
    plt.title("Average Group Returns Across Factors")
    plt.legend(loc="best")
    plt.grid(linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()
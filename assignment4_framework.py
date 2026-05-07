# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:00:06 2026

@author: lich5 

Assignment framework for the Factor Testing assignment.

Students should complete the missing parts marked by TODO.

Main tasks
----------
1. Load and clean required CSV files.
2. Build weekly style factors: size, value, momentum, volatility.
3. Apply industry neutralization to factor exposures.
4. Evaluate one or several factors by:
   - Normal IC
   - RankIC
   - grouped return analysis
   - long-short group return
   - long-short NAV
5. Compare factor testing results.
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

# If the industry information file is stored elsewhere, modify this path.
INDUSTRY_FILE = DATA_DIR / "stk_company_info.csv"


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

        # TODO:
        # 1. Concatenate Shanghai and Shenzhen data by columns.
        # 2. Sort the index.
        # 3. Store the result in dataset[field].
        raise NotImplementedError("Complete data concatenation.")

    # TODO:
    # Find the common date index and common stock columns across all fields.
    raise NotImplementedError("Complete common index and column alignment.")

    # TODO:
    # Reindex all fields to the common index and common columns.
    raise NotImplementedError("Complete field alignment.")

    # TODO:
    # Mask invalid observations:
    # close <= 0, adj_factor <= 0, pb <= 0, total_mv <= 0.
    raise NotImplementedError("Complete invalid value cleaning.")

    return dataset


def winsorize_row(row: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    """
    Cross-sectional winsorization for one date.

    Students should clip extreme factor values by lower and upper quantiles.
    """
    # TODO:
    # 1. Drop missing values.
    # 2. If the row is empty, return the original row.
    # 3. Compute lower and upper quantiles.
    # 4. Clip the row to the quantile range.
    raise NotImplementedError("Complete winsorization.")


def zscore_row(row: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score standardization for one date.
    """
    # TODO:
    # 1. Drop missing values.
    # 2. If fewer than two observations remain, return NaN values.
    # 3. Compute mean and standard deviation.
    # 4. Return (row - mean) / std.
    raise NotImplementedError("Complete z-score standardization.")


def preprocess_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize and standardize factor exposures cross-sectionally.
    """
    # TODO:
    # Apply winsorize_row and zscore_row row by row.
    raise NotImplementedError("Complete factor exposure preprocessing.")


def build_weekly_features(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Build weekly return and style factor matrices.

    Returns
    -------
    dict[str, pd.DataFrame]
        Weekly close, weekly return, and weekly factor matrices.
    """
    # TODO:
    # 1. Compute adjusted close price:
    #       close = raw close * adj_factor
    # 2. Compute daily returns.
    # 3. Construct style factors:
    #       momentum: close / close.shift(20) - 1
    #       value: 1 / pb
    #       size: -log(total_mv)
    #       volatility: rolling 20-day std of daily returns
    # 4. Convert all variables to weekly frequency using resample("W").last().
    # 5. Apply preprocess_exposure to each factor matrix.
    raise NotImplementedError("Complete weekly feature construction.")


# ============================================================
# 2. Industry neutralization
# ============================================================

def get_industries(ind: str = "l1_code") -> pd.DataFrame:
    """
    Load industry classification information.

    The industry file should contain at least:
        ts_code, in_date, and the selected industry column.

    Parameters
    ----------
    ind : str
        Industry classification column, such as 'l1_code'.

    Returns
    -------
    pd.DataFrame
        Industry information indexed by stock code.
    """
    # TODO:
    # 1. Read INDUSTRY_FILE.
    # 2. Set ts_code as index.
    # 3. Return columns [ind, "in_date"].
    raise NotImplementedError("Complete industry data loading.")


class IndustryNeutral:
    """
    Industry neutralization by within-industry z-score.

    For each date t and each industry g,

        F_neutral[t, i] = (F[t, i] - mean_g[t]) / std_g[t],

    where stock i belongs to industry g.
    """

    def __init__(self, ind: str = "l1_code"):
        self.ind = ind
        self.industry = get_industries(ind)

    def __call__(self, factor: pd.DataFrame | dict[str, pd.DataFrame]):
        """
        Apply industry neutralization to one factor matrix or a dictionary of factors.
        """
        # TODO:
        # If factor is a dictionary, neutralize each factor matrix.
        # If factor is a DataFrame, neutralize it directly.
        raise NotImplementedError("Complete industry neutralization call logic.")

    def _neutralize_one_factor(self, factor: pd.DataFrame) -> pd.DataFrame:
        """
        Industry-neutralize one factor matrix.
        """
        # TODO:
        # 1. Copy the factor matrix.
        # 2. Build a stock-to-industry mapping for factor.columns.
        # 3. For each industry:
        #       select stocks in that industry;
        #       compute row-wise industry mean;
        #       compute row-wise industry std;
        #       replace factor values by within-industry z-scores.
        # 4. Return the neutralized factor matrix.
        raise NotImplementedError("Complete one-factor industry neutralization.")


# ============================================================
# 3. Factor-test utilities
# ============================================================

def _safe_pearson(x: pd.Series, y: pd.Series, min_obs: int = 10) -> float:
    """
    Pearson correlation after dropping missing values.
    """
    # TODO:
    # 1. Combine x and y.
    # 2. Drop missing values.
    # 3. Check min_obs and zero standard deviation.
    # 4. Return Pearson correlation.
    raise NotImplementedError("Complete safe Pearson correlation.")


def _safe_spearman(x: pd.Series, y: pd.Series, min_obs: int = 10) -> float:
    """
    Spearman rank correlation after dropping missing values.
    """
    # TODO:
    # Similar to _safe_pearson, but use Spearman correlation.
    raise NotImplementedError("Complete safe Spearman correlation.")


def _safe_ir(x: pd.Series) -> float:
    """
    Compute information ratio: mean divided by standard deviation.
    """
    # TODO:
    # Drop NaN values and return mean / std.
    raise NotImplementedError("Complete IR calculation.")


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
    # TODO:
    # 1. Sort assets by factor values in ascending order.
    # 2. Split the sorted assets into num_groups groups.
    # 3. Return a list of asset index groups.
    raise NotImplementedError("Complete factor-sorted grouping.")


# ============================================================
# 4. Factor test
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
    At date t, factor value F_t is matched with next-period return R_{t+1}.

    Therefore, if the sample contains dates 1, 2, ..., T, the available test dates are
    1, 2, ..., T-1.
    """
    if not isinstance(factors, pd.DataFrame):
        raise TypeError("factors must be a pandas DataFrame.")

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    # TODO:
    # Align factor matrix and return matrix by common dates and common stock columns.
    raise NotImplementedError("Complete factor-return alignment.")

    # TODO:
    # Define test dates. The last date cannot be used because R_{T+1} is unavailable.
    raise NotImplementedError("Complete test date construction.")

    # TODO:
    # Initialize:
    #   ic
    #   rank_ic_s
    #   group_returns
    #   turnover_low
    #   turnover_high
    raise NotImplementedError("Complete output containers.")

    # TODO:
    # Loop over each test date t:
    #   1. Set next_t as the next date.
    #   2. Extract f = factors.loc[t].
    #   3. Extract r_next = returns.loc[next_t].
    #   4. Keep stocks with both non-missing factor and future return.
    #   5. Compute Normal IC.
    #   6. Compute RankIC.
    #   7. Sort stocks by factor value and divide into groups.
    #   8. Compute each group's next-period equal-weighted return.
    #   9. Compute turnover for the lowest and highest groups.
    raise NotImplementedError("Complete main factor testing loop.")

    # TODO:
    # Compute average grouped return.
    raise NotImplementedError("Complete average grouped return calculation.")

    # TODO:
    # Decide long-short direction:
    #   if top group average return >= bottom group average return:
    #       use GQ - G1
    #   else:
    #       use G1 - GQ
    raise NotImplementedError("Complete long-short direction selection.")

    # TODO:
    # Apply transaction cost adjustment:
    #   cost_adjustment = 0.5 * comsn * (turnover_low + turnover_high)
    raise NotImplementedError("Complete transaction cost adjustment.")

    # TODO:
    # Compute:
    #   group_nav
    #   long_short_nav
    #   excessive_nav
    raise NotImplementedError("Complete NAV calculation.")

    # TODO:
    # Build ic_stats as a pd.Series containing:
    #   IC_mean, IC_std, ICIR, IC_win_rate(%)
    #   RankIC_mean, RankIC_std, RankICIR, RankIC_win_rate(%)
    #   LongShort_mean(%), LongShort_std(%), LongShort_final_nav
    #   Excessive_mean(%), Excessive_final_nav
    raise NotImplementedError("Complete summary statistics.")

    # TODO:
    # Build result dictionary.
    raise NotImplementedError("Complete result dictionary.")

    if display:
        # TODO:
        # Print the factor test result.
        pass

    if plot:
        plot_factor_test_result(result)

    return result


# ============================================================
# 5. Plotting
# ============================================================

def plot_factor_test_result(result: dict[str, object]) -> None:
    """
    Plot IC, RankIC, average group return, long-short NAV, and grouped NAV.
    """
    # TODO:
    # Extract objects from result:
    #   name, ic, rank_ic, avg_group_return, long_short_nav,
    #   group_nav, long_short_direction.
    raise NotImplementedError("Complete result extraction for plotting.")

    # TODO:
    # Create a 2-by-2 figure:
    #   subplot 1: RankIC time series and mean line
    #   subplot 2: Normal IC time series and mean line
    #   subplot 3: average group return bar plot
    #   subplot 4: long-short NAV
    raise NotImplementedError("Complete diagnostic plotting.")

    # TODO:
    # Create another figure for grouped NAV curves.
    raise NotImplementedError("Complete grouped NAV plotting.")


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Step 1. Load raw data
    # --------------------------------------------------------
    # TODO:
    # Call load_data().
    raise NotImplementedError("Load dataset.")

    # --------------------------------------------------------
    # Step 2. Build weekly features
    # --------------------------------------------------------
    # TODO:
    # Call build_weekly_features(dataset).
    raise NotImplementedError("Build weekly features.")

    # --------------------------------------------------------
    # Step 3. Select return matrix
    # --------------------------------------------------------
    # TODO:
    # Select weekly return matrix from features.
    # Example:
    # returns = features["ret_w"]
    raise NotImplementedError("Select return matrix.")

    # --------------------------------------------------------
    # Step 4. Build raw factor dictionary
    # --------------------------------------------------------
    # TODO:
    # Build a dictionary containing:
    #   size, value, momentum, volatility
    raise NotImplementedError("Build raw factor dictionary.")

    # --------------------------------------------------------
    # Step 5. Apply industry neutralization
    # --------------------------------------------------------
    # TODO:
    # 1. Create IndustryNeutral(ind="l1_code").
    # 2. Apply it to raw_factor_dict.
    raise NotImplementedError("Apply industry neutralization.")

    # --------------------------------------------------------
    # Step 6. Set factor testing parameters
    # --------------------------------------------------------
    # TODO:
    # Set:
    #   num_groups
    #   comsn
    #   min_obs
    #   plot_factor_name
    raise NotImplementedError("Set factor testing parameters.")

    # --------------------------------------------------------
    # Step 7. Run factor_test for each factor
    # --------------------------------------------------------
    # TODO:
    # For each factor in the neutralized factor dictionary:
    #   call factor_test(...)
    #   store result["ic_stats"] into all_stats
    raise NotImplementedError("Run factor tests.")

    # --------------------------------------------------------
    # Step 8. Build and print summary table
    # --------------------------------------------------------
    # TODO:
    # Convert all_stats into a DataFrame and print it.
    raise NotImplementedError("Build summary table.")

    # --------------------------------------------------------
    # Step 9. Optional raw vs. industry-neutral comparison
    # --------------------------------------------------------
    # TODO:
    # Choose one factor, for example "value".
    # Compare raw factor and industry-neutralized factor by:
    #   1. Running factor_test on raw factor.
    #   2. Running factor_test on neutralized factor.
    #   3. Printing their summary statistics side by side.
    #   4. Plotting their average group returns.
    raise NotImplementedError("Complete raw vs. neutralized comparison.")
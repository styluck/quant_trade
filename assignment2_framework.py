# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:58:42 2026

@author: lich5
"""
# -*- coding: utf-8 -*-
"""
Reference solution for assignment1.

This script follows the assignment steps directly:
1. Load and clean the required CSV files.
2. Build the daily factors mom5 and bp.
3. Convert close and factors to weekly frequency.
4. Select stocks with the provided utility functions.
5. Run the backtest and report summary statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def calc_nav(Pctchg, w, **kwargs):
    '''
        计算净值、pnl、换手率
    '''
    
    if 'comsn' in kwargs.keys():
        comsn = kwargs['comsn'] #加入交易费用
    else:
        comsn = 0
        
    Times = pd.to_datetime(Pctchg.index)
    # 时间期数
    T = len(Times)
    # 从价格序列得到的收益矩阵
    Pctchg = Pctchg.values
    w = w.fillna(0)
    w = w.values
    # 定义pnl
    pnl = np.zeros((T, ))
    # 定义净值
    nav = np.ones((T, ))
    # 定义换手率
    turnover = np.zeros((T, ))
    # 循环计算每期净值以及换手
    
    for i in range(1, T):
        turnover[i] = np.sum(np.abs(w[i] - ((1 + Pctchg[i]) * w[i-1]))) / 2
        pnl[i] = np.dot(w[i-1], Pctchg[i]) - (turnover[i] * comsn)
        nav[i] = (1 + pnl[i]) * nav[i-1]
        
    # 转换成pandas格式
    nav = pd.Series(nav, index = Times)
    pnl = pd.Series(pnl, index = Times)
    turnover = pd.Series(turnover, index = Times)
    
    return {'nav':nav, 'pnl':pnl, 'turnover':turnover}


def plot_equity(nav, bench_stats=None, label='Strategy', color='blue', ax=None):
    def format_two_dec(x, pos):
        return '%.2f' % x

    # -------- clean nav index --------
    nav = nav.copy()
    nav.index = pd.to_datetime(nav.index, errors='coerce')
    nav = nav[~nav.index.isna()].sort_index()

    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(format_two_dec))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

    nav.plot(lw=2, color=color, alpha=0.7, x_compat=False, label=label, ax=ax)

    xmin = nav.index.min()
    xmax = nav.index.max()

    if bench_stats is not None:
        benchmark = bench_stats.copy()
        benchmark.index = pd.to_datetime(benchmark.index, errors='coerce')
        benchmark = benchmark[~benchmark.index.isna()].sort_index()

        if len(benchmark) > 0:
            benchmark = benchmark.pct_change()
            benchmark.iloc[0] = 0
            benchmark_nav = (1 + benchmark).cumprod()
            benchmark_nav = benchmark_nav.reindex(
                pd.date_range(benchmark_nav.index[0], benchmark_nav.index[-1], freq='D')
            )
            benchmark_nav = benchmark_nav.ffill().dropna()

            benchmark_nav.plot(
                lw=2, color='gray', alpha=0.7,
                x_compat=False, label='Benchmark', ax=ax
            )

            xmin = min(xmin, benchmark_nav.index.min())
            xmax = max(xmax, benchmark_nav.index.max())

    # 显式限制 x 轴范围，避免被异常索引拖到 1970
    ax.set_xlim(xmin, xmax)

    ax.axhline(1.0, linestyle='--', color='black', lw=1)
    ax.set_ylabel('Cumulative returns')
    ax.set_xlabel('')
    ax.legend(loc='best')
    plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
    return ax


def stock_selection(finalfac: pd.DataFrame, n: int, thres: float) -> pd.DataFrame:
    mat = pd.DataFrame(False, index=finalfac.index, columns=finalfac.columns)
    for i, time in enumerate(finalfac.index):

        maxi = finalfac.loc[time].max()
        if i == 0:
            # 第一期直接选前n个最大因子值资产
            selected = finalfac.loc[time].nlargest(n).index
            mat.loc[time, selected] = True
        else:
            prev_time = finalfac.index[i-1]
            prev_selected = mat.loc[prev_time]
            # 保留T-1期被选中且因子值 > maxi*0.95的资产
            retained = prev_selected[(prev_selected) & (finalfac.loc[time] > maxi * thres)].index
            mat.loc[time, retained] = True
            # 计算还需选入的资产数量
            remain_num = n - mat.loc[time].sum()
            if remain_num > 0:
                # 剩余未选资产，按因子值排序选入
                candidates = finalfac.loc[time][~mat.loc[time]].sort_values(ascending=False)
                selected_candidates = candidates.head(remain_num).index
                mat.loc[time, selected_candidates] = True
        
    return mat
    


DATA_DIR = Path(__file__).resolve().parent / "dataset"
FIELDS = ["close", "open", "high", "low", "pb", "pe_ttm", "total_mv" ]
PRICE_FIELDS = ["open", "close", "high", "low"]


def read_one(field: str, market: str) -> pd.DataFrame: 
    
    if market== None:
        path = DATA_DIR / f"{field}.csv"
    else:
        path = DATA_DIR / f"{field}_{market}.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.sort_index()
    if market == "sz":
        df = df[~df.index.duplicated(keep="first")]
    return df
 

def load_data() -> dict[str, pd.DataFrame]:
    dataset: dict[str, pd.DataFrame] = {}

    for field in FIELDS:
        sh = read_one(field, "sh")
        sz = read_one(field, "sz")
        dataset[field] = pd.concat([sh, sz], axis=1).sort_index()
        print(f'Data {field} is loaded')

    # Keep only common dates across all required variables.
    common_index = dataset["close"].index
    for field in FIELDS[1:]:
        common_index = common_index.intersection(dataset[field].index)

    common_columns = dataset["close"].columns
    for field in FIELDS[1:]:
        common_columns = common_columns.intersection(dataset[field].columns)

    for field in FIELDS:
        dataset[field] = dataset[field].reindex(index=common_index, columns=common_columns)

    for field in PRICE_FIELDS:
        dataset[field] = dataset[field].mask(dataset[field] <= 0)
  
    return dataset


def build_factors(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = dataset["close"]
    pb = dataset["pb"]
    mv = dataset["total_mv"]
    mom5 = close / close.shift(5) - 1
    bp = (1 / pb).replace([np.inf, -np.inf], np.nan)
    
    weekly = {
        "close_w": close.resample("W").last(),
        "mom5_w": mom5.resample("W").last(),
        "bp_w": bp.resample("W").last(),
        "mv_w": mv.resample("W").last(),
    }
    weekly["Pctchg"] = weekly["close_w"].pct_change()

    return {
        "mom5": mom5,
        "bp": bp,
        "mv": mv,
        **weekly,
    }


def estimate_factor_returns(features,factor_names):
    ret_next = features['ret_w'].shift(-1)
    factor_returns = pd.DataFrame(
        index=features['ret_w'].index, columns=factor_names, dtype=float
    )
    exposures_by_time = {}

    # 先对因子做时间方向前向填充
    size_w = features['size_w'].ffill()
    value_w = features['value_w'].ffill()
    mom_w = features['mom_w'].ffill()
    vol_w = features['vol_w'].ffill()

    for t in features['ret_w'].index:
        X = pd.DataFrame({
            'size': size_w.loc[t],
            'value': value_w.loc[t],
            'mom': mom_w.loc[t],
            'vol': vol_w.loc[t],
        })
        y = ret_next.loc[t]

        valid_mask = y.notna() & X.notna().all(axis=1)
        if valid_mask.sum() < 10:
            continue

        X_valid = X.loc[valid_mask, factor_names]
        y_valid = y.loc[valid_mask]

        X_use = np.column_stack([np.ones(len(X_valid)), X_valid.to_numpy()])
        y_use = y_valid.to_numpy()

        beta, *_ = np.linalg.lstsq(X_use, y_use, rcond=None)

        factor_returns.loc[t, factor_names] = beta[1:]
        exposures_by_time[t] = X_valid.copy()

    return factor_returns, exposures_by_time


def build_predicted_return_matrix(
    factor_returns: pd.DataFrame,
    exposures_by_time: dict[pd.Timestamp, pd.DataFrame],
    lookback: int = 4,
) -> pd.DataFrame:
    factor_names = ['size', 'value', 'mom', 'vol']

    # 先计算滚动平均的因子收益预测
    factor_forecast = factor_returns[factor_names].shift(1).rolling(
        lookback, min_periods=lookback
    ).mean()

    # 转成 numpy，减少循环中的 pandas 索引开销
    forecast_arr = factor_forecast.to_numpy()
    time_index = factor_forecast.index
    time_to_pos = {t: i for i, t in enumerate(time_index)}

    # 用字典收集每期结果，最后一次性转 DataFrame
    out = {}

    for t, X in exposures_by_time.items():
        pos = time_to_pos.get(t)
        if pos is None:
            continue

        fhat = forecast_arr[pos]
        if np.isnan(fhat).any():
            continue

        # X 的列顺序已固定为 ['size','value','mom','vol']
        X_arr = X[factor_names].to_numpy()
        pred_arr = X_arr @ fhat

        out[t] = pd.Series(pred_arr, index=X.index)

    pred = pd.DataFrame.from_dict(out, orient='index')
    pred = pred.reindex(factor_returns.index)
    pred = pred.sort_index(axis=1)
    return pred


def build_equal_weight(mat: pd.DataFrame) -> pd.DataFrame:
    w = mat.astype(float)
    w = w.div(w.sum(axis=1), axis=0)
    w = w.replace([np.inf, -np.inf], 0)
    return w.fillna(0)


def load_benchmark(startdate: str, enddate: str) -> pd.DataFrame:
    
    both = read_one('benchmark', None)
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    both = both.loc[(both.index >= start) & (both.index <= end)]
    
    return both


def summarize_result(result: dict[str, pd.Series]) -> pd.Series:
    nav = result["nav"]
    pnl = result["pnl"]
    turnover = result["turnover"]
    return pd.Series(
        {
            "final_nav": nav.iloc[-1],
            "mean_period_return(%)": pnl.mean()*100,
            "return_volatility(%)": pnl.std(ddof=1)*100,
            "mean_turnover(%)": turnover.mean()*100,
        }
    )
 
# %% ############# main ####################
if __name__ == "__main__": 
    
    dataset = load_data() 
    features = build_factors(dataset)

    factor_name = "mom"  # alternative: "bp"
    finalfac = -features[f"{factor_name}_w"]
    pctchg = features["Pctchg"]

    aligned_index = finalfac.index.intersection(pctchg.index)
    aligned_columns = finalfac.columns.intersection(pctchg.columns)
    finalfac = finalfac.reindex(index=aligned_index, columns=aligned_columns)
    pctchg = pctchg.reindex(index=aligned_index, columns=aligned_columns)

    n = 30
    thres = 1.0
    comsn = 0.001

    mat = stock_selection(finalfac, n, thres)
    w = build_equal_weight(mat)

    result = calc_nav(pctchg.fillna(0), w, comsn=comsn)

    bench_stats = load_benchmark(
        startdate=dataset["close"].index.min().strftime("%Y-%m-%d"),
        enddate=dataset["close"].index.max().strftime("%Y-%m-%d"),
    )
    plot_equity(result["nav"], bench_stats["close"])

    summary = summarize_result(result)

    print(f"Selected factor: {factor_name}")
    print("\nSelection matrix (head):")
    print(mat.head())
    print("\nPerformance summary:")
    print(summary)

# [EOF]

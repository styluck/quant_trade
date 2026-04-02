# -*- coding: utf-8 -*-
"""
Reference solution for the Risk Parity portfolio construction assignment.

This script keeps the main code structure from the previous simplified
Barra multi-factor assignment and extends it to compare:
1. Equal-weight portfolio
2. Simplified Risk Parity portfolio (inverse-volatility weighting)

Main steps
----------
1. Load and clean the required CSV files.
2. Build style factors: size, value, momentum, volatility.
3. Convert data to weekly frequency.
4. Standardize factor exposures cross-sectionally each week.
5. Estimate weekly factor returns via cross-sectional OLS.
6. Forecast next-period factor returns using a rolling mean.
7. Build predicted returns and select stocks.
8. Construct equal-weight and risk-parity portfolios.
9. Run backtests and compare the results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def calc_nav(Pctchg, w, **kwargs):
    """计算净值、pnl、换手率"""
    comsn = kwargs.get('comsn', 0)

    Times = pd.to_datetime(Pctchg.index)
    T = len(Times)
    Pctchg = Pctchg.values
    w = w.fillna(0).values

    pnl = np.zeros((T,))
    nav = np.ones((T,))
    turnover = np.zeros((T,))

    for i in range(1, T):
        turnover[i] = np.sum(np.abs(w[i] - ((1 + Pctchg[i]) * w[i - 1]))) / 2
        pnl[i] = np.dot(w[i - 1], Pctchg[i]) - (turnover[i] * comsn)
        nav[i] = (1 + pnl[i]) * nav[i - 1]

    nav = pd.Series(nav, index=Times)
    pnl = pd.Series(pnl, index=Times)
    turnover = pd.Series(turnover, index=Times)
    return {'nav': nav, 'pnl': pnl, 'turnover': turnover}

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
        current = finalfac.loc[time]
        if current.dropna().empty:
            continue
        maxi = current.max()
        if i == 0:
            selected = current.nlargest(n).index
            mat.loc[time, selected] = True
        else:
            prev_time = finalfac.index[i - 1]
            prev_selected = mat.loc[prev_time]
            retained = prev_selected[(prev_selected) & (current > maxi * thres)].index
            mat.loc[time, retained] = True
            remain_num = n - int(mat.loc[time].sum())
            if remain_num > 0:
                candidates = current[~mat.loc[time]].sort_values(ascending=False)
                selected_candidates = candidates.head(remain_num).index
                mat.loc[time, selected_candidates] = True
    return mat


DATA_DIR = Path(__file__).resolve().parent / 'dataset'
FIELDS = ['close', 'pb', 'total_mv','adj_factor']



def read_one(field: str, market: str | None) -> pd.DataFrame:
    path = DATA_DIR / (f'{field}.csv' if market is None else f'{field}_{market}.csv')
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[~df.index.isna()].sort_index()
    if market == 'sz':
        df = df[~df.index.duplicated(keep='first')]
    return df



def load_data() -> dict[str, pd.DataFrame]:
    dataset: dict[str, pd.DataFrame] = {}
    for field in FIELDS:
        sh = read_one(field, 'sh')
        sz = read_one(field, 'sz')
        dataset[field] = pd.concat([sh, sz], axis=1).sort_index()
        print(f'Data {field} is loaded')

    common_index = dataset['close'].index
    common_columns = dataset['close'].columns
    for field in FIELDS[1:]:
        common_index = common_index.intersection(dataset[field].index)
        common_columns = common_columns.intersection(dataset[field].columns)

    for field in FIELDS:
        dataset[field] = dataset[field].reindex(index=common_index, columns=common_columns)

    dataset['close'] = dataset['close'].mask(dataset['close'] <= 0)
    dataset['adj_factor'] = dataset['adj_factor'].mask(dataset['adj_factor'] <= 0)
    dataset['pb'] = dataset['pb'].mask(dataset['pb'] <= 0)
    dataset['total_mv'] = dataset['total_mv'].mask(dataset['total_mv'] <= 0)
    return dataset



def winsorize_row(row: pd.Series, lower: float = 0.02, upper: float = 0.98) -> pd.Series:
    valid = row.dropna()
    if valid.empty:
        return row
    lo = valid.quantile(lower)
    hi = valid.quantile(upper)
    return row.clip(lower=lo, upper=hi)

def zscore_row(row: pd.Series) -> pd.Series:
    valid = row.dropna()
    if valid.shape[0] < 2:
        return row * np.nan
    mu = valid.mean()
    sd = valid.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return row * np.nan
    return (row - mu) / sd

def preprocess_exposure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.apply(winsorize_row, axis=1)
    out = out.apply(zscore_row, axis=1)
    return out



def build_weekly_features(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close = dataset['close']*dataset['adj_factor']
    pb = dataset['pb']
    total_mv = dataset['total_mv']

    daily_ret = close.pct_change()
    mom20 = close / close.shift(20) - 1
    value = (1.0 / pb).replace([np.inf, -np.inf], np.nan)
    size = -np.log(total_mv)
    vol20 = daily_ret.rolling(20).std()

    weekly = {
        'close_w': close.resample('W').last(),
        'ret_w': close.resample('W').last().pct_change(),
        'size_w': size.resample('W').last(),
        'value_w': value.resample('W').last(),
        'mom_w': mom20.resample('W').last(),
        'vol_w': vol20.resample('W').last(),
    }

    weekly['size_w'] = preprocess_exposure(weekly['size_w'])
    weekly['value_w'] = preprocess_exposure(weekly['value_w'])
    weekly['mom_w'] = preprocess_exposure(weekly['mom_w'])
    weekly['vol_w'] = preprocess_exposure(weekly['vol_w'])
    return weekly


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



def build_risk_parity_weight(mat: pd.DataFrame, pctchg: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Simplified risk parity: inverse-volatility weighting within the selected universe."""
    rolling_vol = pctchg.rolling(window=window, min_periods=window).std()
    inv_vol = 1.0 / rolling_vol
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    # Keep only selected assets each period.
    rp_raw = inv_vol.where(mat, np.nan)
    rp_w = rp_raw.div(rp_raw.sum(axis=1), axis=0)
    rp_w = rp_w.replace([np.inf, -np.inf], 0)
    return rp_w.fillna(0)



def load_benchmark(startdate: str, enddate: str) -> pd.DataFrame:
    both = read_one('benchmark', None)
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)
    both = both.loc[(both.index >= start) & (both.index <= end)]
    return both



def summarize_result(result: dict[str, pd.Series]) -> pd.Series:
    nav = result['nav']
    pnl = result['pnl']
    turnover = result['turnover']
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return pd.Series({
        'final_nav': nav.iloc[-1],
        'mean_period_return(%)': pnl.mean() * 100,
        'return_volatility(%)': pnl.std(ddof=1) * 100,
        'max_drawdown(%)': drawdown.min() * 100,
        'mean_turnover(%)': turnover.mean() * 100,
    })

# %% ----- main -----
if __name__ == '__main__':
    dataset = load_data()
    features = build_weekly_features(dataset)

    # ----- Keep the previous Barra code part -----
    
    factor_names = ['size', 'value', 'mom', 'vol']
    factor_returns, exposures_by_time = estimate_factor_returns(features, factor_names)
    pred_ret = build_predicted_return_matrix(factor_returns, exposures_by_time, lookback=12)

    pctchg = features['ret_w']
    aligned_index = pred_ret.index.intersection(pctchg.index)
    aligned_columns = pred_ret.columns.intersection(pctchg.columns)
    finalfac = pred_ret.reindex(index=aligned_index, columns=aligned_columns)
    pctchg = pctchg.reindex(index=aligned_index, columns=aligned_columns)

    n = 30
    thres = 1.0
    comsn = 0.001
    rp_window = 12

    # ----- Same stock pool from the previous assignment -----
    mat = stock_selection(finalfac, n, thres)

    # Equal-weight portfolio
    w_eq = build_equal_weight(mat)
    result_eq = calc_nav(pctchg.fillna(0), w_eq, comsn=comsn)

    # Risk parity portfolio
    w_rp = build_risk_parity_weight(mat, pctchg, window=rp_window)
    result_rp = calc_nav(pctchg.fillna(0), w_rp, comsn=comsn)

    bench_stats = load_benchmark(
        startdate=dataset['close'].index.min().strftime('%Y-%m-%d'),
        enddate=dataset['close'].index.max().strftime('%Y-%m-%d'),
    )
    
    # %% ---- plotings ----
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plot_equity(result_eq['nav'], bench_stats['close'], label='Equal Weight', color='blue', ax=ax)
    plot_equity(result_rp['nav'], None, label='Risk Parity', color='red', ax=ax)


    summary_eq = summarize_result(result_eq)
    summary_rp = summarize_result(result_rp)
    summary_table = pd.concat([summary_eq, summary_rp], axis=1)
    summary_table.columns = ['EqualWeight', 'RiskParity']

    print('Risk Parity portfolio construction assignment solution')
    print('\nSelection matrix (head):')
    print(mat.head())
    print('\nEqual-weight matrix (head):')
    print(w_eq.head())
    print('\nRisk-parity weight matrix (head):')
    print(w_rp.head())
    print('\nPerformance summary:')
    print(summary_table)
    
    # [EOF]

# -*- coding: utf-8 -*-
"""
Assignment 1 framework script

使用说明
1. 本文件按照 assignment1.pdf 的任务顺序组织代码。
2. 你需要在 `# TODO` 标记处补全代码。 
3. 题目要求使用的函数请直接调用：
   - `stock_selection`
   - `calc_nav`
   - `plot_equity`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_io.io_framework import load_benchmark
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


DATA_DIR = Path(__file__).resolve().parent / "datasets"
FIELDS = ["close", "open", "high", "low", "pb"]
PRICE_FIELDS = ["open", "close", "high", "low"]


def read_one(field: str, market: str) -> pd.DataFrame:
    """
    读取单个原始 CSV 文件，例如：
    - field='close', market='sh' -> close_sh.csv
    - field='pb', market='sz' -> pb_sz.csv

    Task 1 对应要求：
    - 读取文件
    - 将行索引转为日期
    - 按日期升序排序
    - 对深圳市场文件去重，只保留第一条记录
    """
    path = DATA_DIR / f"{field}_{market}.csv"

    # TODO: 使用 pd.read_csv(..., index_col=0) 读取 CSV
    df = None

    # TODO: 将索引转换为 datetime，并删除无法解析的日期

    # TODO: 按日期升序排序

    # TODO: 若 market == "sz"，删除重复日期，只保留第一条

    return df


def load_data() -> dict[str, pd.DataFrame]:
    """
    Task 1
    读取并整理作业所需的原始数据。

    建议输出：
    dataset["close"]
    dataset["open"]
    dataset["high"]
    dataset["low"]
    dataset["pb"]
    """
    dataset: dict[str, pd.DataFrame] = {}

    for field in FIELDS:
        # TODO: 分别读取 sh 和 sz 两个市场的数据
        sh = None
        sz = None

        # TODO: 将同一字段的 sh / sz 数据按列方向合并
        dataset[field] = None

        print(f"Data {field} is loaded.")

    # TODO: 找到所有变量共同拥有的日期索引 common_index
    common_index = None

    # TODO: 找到所有变量共同拥有的股票代码 common_columns
    common_columns = None

    # TODO: 将每个字段都 reindex 到共同的日期和股票范围

    # TODO: 对价格数据（open, close, high, low），若数值 <= 0，则设为缺失值

    return dataset


def build_factors(dataset: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Task 2
    构造题目要求的基础因子与收益率。

    需要完成：
    1. 构造 5 日动量因子 mom5
    2. 构造账面市净率倒数因子 bp
    3. 将收盘价和因子转换为周度数据
    4. 根据周度收盘价计算周收益率 Pctchg
    """
    close = dataset["close"]
    pb = dataset["pb"]

    # TODO: 根据题目公式计算 5 日动量因子 mom5
    # 提示：可使用 close.shift(5)
    mom5 = None

    # TODO: 计算账面市净率倒数因子 bp = 1 / pb
    # 注意处理 inf 和 -inf
    bp = None

    # TODO: 将收盘价和因子转换为周频数据
    # 提示：可使用 .resample("W").last()
    close_w = None
    mom5_w = None
    bp_w = None

    # TODO: 根据周度收盘价计算周收益率，并命名为 Pctchg
    pctchg = None

    features = {
        "mom5": mom5,
        "bp": bp,
        "close_w": close_w,
        "mom5_w": mom5_w,
        "bp_w": bp_w,
        "Pctchg": pctchg,
    }
    return features


def choose_factor(features: dict[str, pd.DataFrame], factor_name: str) -> pd.DataFrame:
    """
    Task 3.1
    选择一个因子作为选股依据，并整理成宽表 finalfac。

    factor_name 可选：
    - "mom5"
    - "bp"
    """
    candidate_factors = {
        "mom5": features["mom5_w"],
        "bp": features["bp_w"],
    }

    if factor_name not in candidate_factors:
        raise ValueError("factor_name must be 'mom5' or 'bp'.")

    finalfac = candidate_factors[factor_name]

    return finalfac


def align_factor_and_return(
    finalfac: pd.DataFrame, pctchg: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    在回测前，确保因子矩阵和收益率矩阵在日期和股票维度上完全对齐。
    """
    # TODO: 找到共同日期
    aligned_index = None

    # TODO: 找到共同股票代码
    aligned_columns = None

    # TODO: 对 finalfac 和 pctchg 重新索引

    return finalfac, pctchg


def build_equal_weight(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Task 3.3
    根据选股结果矩阵 mat 构造等权持仓矩阵 w。

    要求：
    - 每期被选中的股票权重相等
    - 每期权重和为 1
    - 若某期没有股票被选中，则该期权重设为 0
    """
    # TODO: 将布尔矩阵 mat 转成浮点矩阵
    w = None

    # TODO: 按行归一化，使每期权重之和为 1

    # TODO: 处理 inf / NaN

    return w


def load_benchmark(startdate: str, enddate: str) -> pd.DataFrame:
    """
    读取 benchmark.csv，并截取样本区间。
    """
    path = DATA_DIR / "benchmark.csv"

    # TODO: 读取 benchmark.csv
    benchmark = None

    # TODO: 将索引转为日期，并按日期排序

    # TODO: 按 startdate 和 enddate 截取样本区间

    return benchmark


def summarize_result(result: dict[str, pd.Series]) -> pd.Series:
    """
    Task 3.6
    输出以下绩效指标：
    - 最终净值
    - 平均每期收益率
    - 收益率样本波动率
    - 平均换手率
    """
    # TODO: 从 result 中取出 nav, pnl, turnover
    nav = None
    pnl = None
    turnover = None

    # TODO: 计算四个指标
    summary = pd.Series(
        {
            "final_nav": np.nan,
            "mean_period_return(%)": np.nan,
            "return_volatility(%)": np.nan,
            "mean_turnover(%)": np.nan,
        }
    )

    return summary


###################### 基础代码 ######################
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


def plot_equity(nav, bench_stats=None):

    def format_two_dec(x, pos):
        return '%.2f' % x
    
    equity = nav
    
    plt.figure()
    ax = plt.gca()

    y_axis_formatter = FuncFormatter(format_two_dec)
    ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
    ax.xaxis.set_tick_params(reset=True)
    ax.yaxis.grid(linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.grid(linestyle=':')
    
    equity.plot(lw=2, color='blue', alpha=0.6, x_compat=False,
                label='Strategy', ax=ax)

    benchmark = bench_stats.pct_change()
    benchmark.iloc[0] = 0
    benchmark_nav = (1 + benchmark).cumprod()
    benchmark_nav = benchmark_nav.reindex(pd.date_range(benchmark_nav.index[0], benchmark_nav.index[-1], freq='D'))
    benchmark_nav = benchmark_nav.fillna(method = 'ffill')
    benchmark_nav = benchmark_nav.dropna()
    
    benchmark_nav.plot(lw=2, color='gray', alpha=0.6, x_compat=False,
                   label='Benchmark', ax=ax)
    ax.axhline(1.0, linestyle='--', color='black', lw=1)
    ax.set_ylabel('Cumulative returns')
    ax.legend(loc='best')
    ax.set_xlabel('')
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

###################### 基础代码 ######################

 
if __name__ == "__main__":
    # =========================
    # Task 1: 基础清洗
    # =========================
    dataset = load_data()

    # =========================
    # Task 2: 因子与收益率构造
    # =========================
    features = build_factors(dataset)

    # =========================
    # Task 3.1: 选择因子
    # =========================
    factor_name = "mom5"  # TODO: 改成 "mom5" 或 "bp"
    finalfac = choose_factor(features, factor_name)
    pctchg = features["Pctchg"]

    # =========================
    # Task 3.2: 选股
    # =========================
    finalfac, pctchg = align_factor_and_return(finalfac, pctchg)

    n = 30
    thres = 1.0
    comsn = 0.001

    # TODO: 调用 stock_selection(finalfac, n, thres)
    mat = None

    # =========================
    # Task 3.3 - 3.5: 持仓、回测、画图
    # =========================
    w = build_equal_weight(mat)

    # TODO: 调用 calc_nav(pctchg.fillna(0), w, comsn=comsn)
    result = None

    # TODO: 根据样本起止日期读取 benchmark
    bench_stats = None

    # TODO: 调用 plot_equity(result["nav"], bench_stats["close"])

    # =========================
    # Task 3.6 - 3.7: 指标汇总与结果说明
    # =========================
    summary = summarize_result(result)

    print(f"Selected factor: {factor_name}")
    print("\nSelection matrix (head):")
    print(mat.head())
    print("\nPerformance summary:")
    print(summary)

    # TODO:
    # 根据净值曲线和 summary，
    # 用 2~3 句话对策略表现做简要说明。

# [EOF]
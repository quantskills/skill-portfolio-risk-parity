"""风险平价组合月度 rebalance 回测 —— 无未来函数

口径：
    1. 交易日按月分组，每月【最后一个交易日】为调仓日
    2. 每次调仓：用截至当日（含）的过去 LOOKBACK_DAYS 个交易日估 LW 收缩协方差 → 求 ERC 权重
    3. 下一交易日起到下次调仓前，持有该固定权重（权重只用过去信息，无未来函数）

对比基准：等权组合（1/N，验证风险平价是否真把风险摊平）+ 沪深300 指数 benchmark。

关键指标：样本外实现波动率 / 年化收益 / 最大回撤 / 夏普 /
        样本外风险贡献离散度（验证 ERC 在 test 段是否仍近似等风险）/ 调仓次数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor import (
    LOOKBACK_DAYS,
    TRADING_DAYS,
    build_return_matrix,
    ledoit_wolf_shrinkage,
    risk_contributions,
    solve_erc,
)

BENCHMARK_SYMBOL = "000300.SH"   # 默认 benchmark（沪深300）


# === 基础纯函数（供 validate / report 复用）=====================================
def annualized_return(returns: np.ndarray) -> float:
    """年化收益率（几何口径）"""
    if len(returns) == 0:
        return 0.0
    cum = float(np.prod(1.0 + returns) - 1.0)
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return cum
    return float((1.0 + cum) ** (1.0 / years) - 1.0)


def annualized_vol(returns: np.ndarray) -> float:
    """年化波动率"""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    """最大回撤（负值）"""
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min())


def sharpe_ratio(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """夏普比率（年化）"""
    if len(returns) < 2:
        return 0.0
    excess = returns - rf_daily
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(TRADING_DAYS))


def portfolio_daily_returns(ret: pd.DataFrame, weights: dict) -> pd.Series:
    """给定权重字典，算组合日收益序列"""
    w = np.array([weights.get(s, 0.0) for s in ret.columns])
    return pd.Series(ret.values @ w, index=ret.index, name="portfolio_return")


# === 月度 rebalance 调仓权重序列 =================================================
def monthly_rebalance_weights(ret: pd.DataFrame,
                              lookback: int = LOOKBACK_DAYS) -> dict:
    """逐月调仓：每月末用过去 lookback 日算 ERC 权重

    Returns: {rebalance_date_str: {symbol: weight}}，按时间升序。
    首个 lookback 期内不调仓（数据不足，无法估协方差）。
    """
    dates = ret.index.tolist()
    dt_idx = pd.to_datetime([str(d) for d in dates])
    ser = pd.Series(dates, index=dt_idx)
    # 每月最后一个交易日
    rebal_dates = ser.groupby(dt_idx.to_period("M")).apply(lambda x: x.iloc[-1]).tolist()

    weights_by_date: dict = {}
    for d in rebal_dates:
        idx = dates.index(d)
        if idx < lookback:                      # 数据不足，跳过
            continue
        window = ret.iloc[idx - lookback + 1: idx + 1]   # 截至当日（含）的过去 lookback 日
        try:
            Sigma, _, _ = ledoit_wolf_shrinkage(window)
            w, _ = solve_erc(Sigma)
        except Exception as e:
            print(f"[WARN] 调仓日 {d} 求解失败 ({e})，跳过")
            continue
        weights_by_date[d] = dict(zip(ret.columns, w))
    return weights_by_date


def backtest_with_curves(ret: pd.DataFrame, weights_by_date: dict,
                         benchmark_series: pd.Series | None = None) -> dict:
    """根据调仓权重序列，逐日算 RP 组合与等权组合的收益/净值曲线"""
    dates = ret.index.tolist()
    sorted_rebal = sorted(weights_by_date.keys())
    n_assets = ret.shape[1]

    rp_ret, eq_ret = [], []
    for d in dates:
        # 生效权重 = 最近一个 ≤d 的调仓日权重
        applicable = [rd for rd in sorted_rebal if rd <= d]
        if not applicable:
            continue
        w = weights_by_date[applicable[-1]]
        w_arr = np.array([w.get(s, 0.0) for s in ret.columns])
        day_ret = ret.loc[d].values
        rp_ret.append((d, float(w_arr @ day_ret)))
        eq_ret.append((d, float(np.full(n_assets, 1.0 / n_assets) @ day_ret)))

    rp_s = pd.Series(dict(rp_ret))
    eq_s = pd.Series(dict(eq_ret))
    rp_curve = (1.0 + rp_s).cumprod()
    eq_curve = (1.0 + eq_s).cumprod()
    rp_dd = rp_curve / np.maximum.accumulate(rp_curve) - 1.0

    out = {
        "rp_daily": rp_s, "eq_daily": eq_s,
        "rp_curve": rp_curve, "eq_curve": eq_curve,
        "rp_drawdown": rp_dd,
        "dates": list(rp_s.index),
    }
    if benchmark_series is not None:
        bench = benchmark_series.reindex(rp_s.index).dropna()
        out["bench_daily"] = bench
        out["bench_curve"] = (1.0 + bench).cumprod()
    return out


def run_backtest(price_df: pd.DataFrame, future_meta: dict | None = None,
                 lookback: int = LOOKBACK_DAYS,
                 benchmark_symbol: str = BENCHMARK_SYMBOL) -> dict:
    """月度 rebalance 完整回测，返回指标 + 时序数据（供 report 用）

    Args:
        price_df: 三类资产合并长表（date/symbol/close/asset_type）
        future_meta: 期货元信息（报告资金占用）
        lookback: 协方差估计窗口（默认 60 交易日）
        benchmark_symbol: 基准指数代码（需在 price_df 的指数资产中）
    """
    ret = build_return_matrix(price_df)
    weights_by_date = monthly_rebalance_weights(ret, lookback)
    if not weights_by_date:
        raise ValueError(f"无有效调仓（数据长度不足 lookback={lookback}）")

    # benchmark：从 price_df 取指数 benchmark 的日收益
    bench_series = None
    bench_df = price_df[price_df["symbol"] == benchmark_symbol]
    if not bench_df.empty:
        bs = bench_df.sort_values("date").set_index("date")["close"].ffill().pct_change(fill_method=None).dropna()
        bs.index = bs.index.astype(str)
        bench_series = bs

    curves = backtest_with_curves(ret, weights_by_date, bench_series)
    rp_r = curves["rp_daily"].values
    eq_r = curves["eq_daily"].values

    # 样本外风险贡献离散度：用全样本协方差 + 末次权重看均衡性（1.0=完全等风险）
    last_w = weights_by_date[sorted(weights_by_date.keys())[-1]]
    Sigma_full, _, _ = ledoit_wolf_shrinkage(ret)
    rc = risk_contributions(np.array([last_w.get(s, 0.0) for s in ret.columns]), Sigma_full)
    rc_disp = float(rc.max() / rc.min()) if rc.min() > 0 else float("inf")

    bench_r = curves.get("bench_daily")
    metrics = {
        "RP_年化波动": round(annualized_vol(rp_r), 6),
        "等权_年化波动": round(annualized_vol(eq_r), 6),
        "RP_年化收益": round(annualized_return(rp_r), 6),
        "等权_年化收益": round(annualized_return(eq_r), 6),
        "RP_最大回撤": round(max_drawdown(rp_r), 6),
        "等权_最大回撤": round(max_drawdown(eq_r), 6),
        "RP_夏普": round(sharpe_ratio(rp_r), 6),
        "等权_夏普": round(sharpe_ratio(eq_r), 6),
        "波动收敛(RP-等权)": round(annualized_vol(rp_r) - annualized_vol(eq_r), 6),  # 负=RP 波动更低
        "样本外风险贡献离散度": round(rc_disp, 4),
        "调仓次数": len(weights_by_date),
        "lookback": lookback,
        "资产数": ret.shape[1],
        "test_交易日数": len(rp_r),
    }
    if bench_r is not None and len(bench_r) > 1:
        metrics.update({
            "Benchmark_年化收益": round(annualized_return(bench_r.values), 6),
            "Benchmark_年化波动": round(annualized_vol(bench_r.values), 6),
            "超额年化(RP-Bench)": round(annualized_return(rp_r) - annualized_return(bench_r.values), 6),
        })
    metrics["评估口径"] = (
        f"月末调仓，过去 {lookback} 日估 LW 收缩协方差求 ERC 权重，下月持有（无未来函数）。"
        f"对比等权(1/N)与 {benchmark_symbol}。风险贡献离散度=末次权重在全样本协方差下的 max(RC)/min(RC)。"
    )
    return {"metrics": metrics, "curves": curves, "weights_by_date": weights_by_date,
            "symbols": ret.columns.tolist()}


if __name__ == "__main__":
    from data_loader import load_all_assets
    df, meta = load_all_assets()
    result = run_backtest(df, meta)
    print("=== 风险平价月度 rebalance 回测 ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

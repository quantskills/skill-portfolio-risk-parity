"""风险平价组合优化 —— 因子计算层（factor）

"风险平价因子" = 各资产的等风险贡献（ERC）权重。本模块纯因子计算，不含数据加载：
    收益矩阵 → Ledoit-Wolf 常数相关收缩协方差 → scipy SLSQP 求 ERC 权重 → 权重表。

与 Alpha 横截面选股因子不同：风险平价是【组合优化因子】，输出是每资产的配置权重
（而非 IC/IR/分层/score/signal）。因此 SKILL.md 的"因子逻辑/输出结果"按权重口径描述。

数据加载由 data_loader.py 负责（load_index_data / load_fund_data / load_future_data /
load_all_assets），本模块仅 `from data_loader import load_all_assets` 供 main 入口使用。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data_loader import load_all_assets

# === 因子/模型常量 ===============================================================
MODEL_ID = "RP1"                       # 模型编号
MODEL_NAME = "风险平价组合(ERC)"        # 模型名称

LOOKBACK_DAYS = 60                     # 月度 rebalance 估协方差的回溯窗口（~3 个月）
TRADING_DAYS = 252                     # 年化换算
REBALANCE_FREQ = "M"                   # 月末调仓
ERC_FTOL = 1e-12                       # SLSQP 求解精度
ERC_RC_RATIO_TOL = 0.05                # 风险贡献相等性容差：max(RC)/min(RC)−1 ≤ 5%


# === 收益矩阵 ===================================================================
def build_return_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    """长表 → T×N 日收益矩阵（index=date, columns=symbol）

    内连接对齐：只保留三类资产都有数据的交易日（期货夜盘/基金不同步会丢少量日）。
    """
    # 防御：期货日线偶有同日重复行（数据源特性），按 (date,symbol) 去重后再透视
    price_df = price_df.drop_duplicates(subset=["date", "symbol"], keep="last")
    price = price_df.pivot(index="date", columns="symbol", values="close").sort_index()
    # ffill 显式前向填充，等价原 pct_change() 默认 pad 行为（pandas 2.1+ 已弃用 fill_method
    # 默认值，显式 fill_method=None 关闭 pct_change 内部填充，改由 ffill 负责）。
    # ETF/指数交易日不同步产生的中间缺失日用前收盘价填（收益记 0），dropna 再丢前导缺失。
    ret = price.ffill().pct_change(fill_method=None).dropna(how="any")
    if ret.empty or ret.shape[1] < 2:
        raise ValueError(f"收益矩阵为空或资产数<2 (shape={ret.shape})")
    return ret


# === Ledoit-Wolf 收缩协方差（常数相关目标，手写）================================
def ledoit_wolf_shrinkage(ret: np.ndarray | pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Ledoit-Wolf 收缩协方差估计（常数相关目标 Constant-Correlation，Ledoit-Wolf 2003）

    Σ = δ·F + (1-δ)·S，其中
        S = 样本协方差（1/T 归一）
        F = 常数相关目标：F_ij = ρ̄·σ_i·σ_j（i≠j），F_ii = S_ii
        ρ̄ = 样本相关矩阵的非对角平均
        δ = clip( π̂ / (γ̂·T), 0, 1 )
            π̂ = Σ_ij (1/T)·Σ_t [ Xc[t,i]·Xc[t,j] - S[i,j] ]²   （协方差元素渐近方差之和）
            γ̂ = ‖F - S‖_F²                                       （目标与样本的偏离）

    δ 越大收缩越多（样本噪声大/T 小/N 大时 δ→1，用稳定目标；反之 δ→0 用样本）。

    Returns: (Sigma_shrink, delta, rho_bar)
    """
    X = np.asarray(ret, dtype=float)
    T, N = X.shape
    if T < 3:
        raise ValueError(f"样本过少(T={T})，无法估计协方差")

    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / T                       # 样本协方差（MLE，1/T）
    sigma = np.sqrt(np.diag(S))
    sigma[sigma <= 0] = 1e-12                 # 防除零

    # 平均相关系数 ρ̄（非对角元素均值）
    D_inv = np.diag(1.0 / sigma)
    Corr = D_inv @ S @ D_inv
    off_mask = ~np.eye(N, dtype=bool)
    rho_bar = float(Corr[off_mask].mean()) if N > 1 else 0.0

    # 目标 F：常数相关
    F = np.outer(sigma, sigma) * rho_bar
    np.fill_diagonal(F, np.diag(S))

    # π̂：样本协方差各元素的渐近方差之和
    pi_mat = np.zeros((N, N))
    for t in range(T):
        dev = np.outer(Xc[t], Xc[t]) - S
        pi_mat += dev ** 2
    pi_mat /= T
    pi_hat = float(pi_mat.sum())

    # γ̂：目标与样本之差的 Frobenius 范数平方
    gamma_hat = float(((F - S) ** 2).sum())

    # 收缩强度
    delta = 0.0 if gamma_hat <= 0 else max(0.0, min(1.0, pi_hat / (gamma_hat * T)))

    Sigma = delta * F + (1.0 - delta) * S
    # 数值对称化 + 确保 PSD 的小幅正则（对角加 epsilon）
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma, float(delta), rho_bar


# === 风险贡献 + ERC 求解 =========================================================
def risk_contributions(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """资产 i 的风险贡献 RC_i = w_i · (Σw)_i / σ_p

    满足 Σ_i RC_i = σ_p（组合波动率），即风险贡献之和 = 总风险。
    """
    w = np.asarray(w, dtype=float)
    var = float(w @ Sigma @ w)
    if var <= 1e-20:
        return np.zeros_like(w)
    sigma_p = np.sqrt(var)
    mrc = (Sigma @ w) / sigma_p          # 边际风险贡献 ∂σ_p/∂w_i
    return w * mrc                       # 风险贡献


def solve_erc(Sigma: np.ndarray, ftol: float = ERC_FTOL) -> tuple[np.ndarray, bool]:
    """等风险贡献（ERC）权重求解

        min_w  Σ_i ( RC_i - σ_p/N )²        （每项风险贡献趋于相等）
        s.t.   Σ w_i = 1,  w_i ≥ 0          （long-only，名义权重归一）

    初值用逆波动率 w_i ∝ 1/σ_i（风险平价的一阶近似，收敛快）。
    """
    N = Sigma.shape[0]
    sigma = np.sqrt(np.diag(Sigma))
    sigma[sigma <= 0] = 1e-12
    w0 = (1.0 / sigma) / (1.0 / sigma).sum()      # 逆波动率初值

    def obj(w):
        rc = risk_contributions(w, Sigma)
        total = rc.sum()
        if total <= 0:
            return 1.0
        rc_pct = rc / total                        # 风险贡献占比（无量纲，Σ=1）
        # 归一化目标：与 1/N 的偏差平方。无量纲量级~O(1)，SLSQP 收敛稳定
        return float(np.sum((rc_pct - 1.0 / N) ** 2))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * N                     # long-only，下界微小非零避免奇异
    # 过滤 SLSQP 噪声警告（线性搜索 directional derivative / maxiter，成功求解时无意义）；
    # 真正的求解失败由下方 res.success 捕获并显式告警，不在此静默
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": ftol})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()                                # 归一消除数值漂移
    if not res.success:
        # 未收敛不静默——显式告警（调用方可据此跳过/降级），区别于上面的噪声过滤
        print(f"[WARN] ERC-SLSQP 未收敛：{res.message}（已归一兜底，结果可能次优）")
    return w, bool(res.success)


# === 完整单期优化 ================================================================
def optimize_portfolio(price_df: pd.DataFrame | None = None,
                       future_meta: dict | None = None,
                       update_time: str | None = None) -> pd.DataFrame:
    """完整流程：合并数据 → 收益矩阵 → LW 收缩 → ERC 求解 → 权重表

    Args:
        price_df: 三类资产合并长表（date/symbol/close/asset_type）。None 则联网加载。
        future_meta: 期货元信息 {symbol: {multiplier, margin_rate, name, product}}。
        update_time: 生成时间；None 按数据最新日推导（可复现）。

    Returns:
        权重表 DataFrame（每资产一行），组合级指标存 attrs。
    """
    if price_df is None:
        price_df, future_meta = load_all_assets()
    future_meta = future_meta or {}

    ret = build_return_matrix(price_df)
    symbols = ret.columns.tolist()
    Sigma, delta, rho_bar = ledoit_wolf_shrinkage(ret)
    w, success = solve_erc(Sigma)

    # 风险贡献
    rc = risk_contributions(w, Sigma)
    sigma_p = float(np.sqrt(w @ Sigma @ w))
    rc_pct = rc / sigma_p if sigma_p > 0 else rc           # 风险贡献占比（应≈1/N）
    # 资产年化波动（用收缩协方差对角）
    asset_vol = np.sqrt(np.diag(Sigma)) * np.sqrt(TRADING_DAYS)

    # asset_type / 杠杆信息
    sym2type = dict(zip(price_df["symbol"], price_df["asset_type"]))

    def _meta(sym, key, default=np.nan):
        m = future_meta.get(sym)
        return m.get(key, default) if m else default

    latest_date = str(price_df["date"].astype(str).max())
    trade_date = f"{latest_date[:4]}-{latest_date[4:6]}-{latest_date[6:8]}"
    update_time = update_time or f"{trade_date}T15:30:00"

    rows = []
    for i, sym in enumerate(symbols):
        atype = sym2type.get(sym, "")
        margin = _meta(sym, "margin_rate", np.nan) if atype == "future" else np.nan
        # 资金占用权重：期货 = 名义权重×margin_rate，其它 = 名义权重
        capital = (w[i] * margin) if atype == "future" and not np.isnan(margin) else w[i]
        rows.append({
            "trade_date": trade_date,
            "asset_type": atype,
            "symbol": sym,
            "name": _meta(sym, "name", sym),
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "weight": round(float(w[i]), 6),                       # 名义敞口权重
            "risk_contribution": round(float(rc[i]), 6),           # 风险贡献（绝对）
            "risk_contribution_pct": round(float(rc_pct[i]), 6),   # 风险贡献占比
            "volatility": round(float(asset_vol[i]), 6),           # 年化波动率
            "margin_rate": round(float(margin), 4) if not np.isnan(margin) else np.nan,
            "contract_multiplier": (round(float(_meta(sym, "multiplier", np.nan)), 4)
                                    if atype == "future" else np.nan),
            "capital_allocation": round(float(capital), 6),        # 资金占用权重
            "data_version": "real-v1",
            "update_time": update_time,
        })

    out = pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    # 组合级指标
    rc_ratio = float(rc.max() / rc.min()) if rc.min() > 0 else float("inf")
    out.attrs["portfolio_vol_annual"] = round(float(sigma_p * np.sqrt(TRADING_DAYS)), 6)
    out.attrs["portfolio_vol_daily"] = round(float(sigma_p), 6)
    out.attrs["lw_shrinkage"] = round(float(delta), 6)
    out.attrs["mean_correlation"] = round(float(rho_bar), 6)
    out.attrs["rc_max_over_min"] = round(rc_ratio, 6)             # ERC 相等性指标（应≈1）
    out.attrs["erc_solved"] = bool(success)
    out.attrs["n_assets"] = len(symbols)
    out.attrs["target"] = "equal_risk_contribution"
    out.attrs["rebalance_freq"] = REBALANCE_FREQ
    out.attrs["lookback_days"] = LOOKBACK_DAYS
    # 资金口径：总资金占用（期货杠杆后），<1 表示有杠杆空间
    capital_total = float(out["capital_allocation"].fillna(out["weight"]).sum())
    out.attrs["capital_utilization"] = round(capital_total, 6)
    return out


if __name__ == "__main__":
    df, meta = load_all_assets()
    result = optimize_portfolio(df, meta)
    print(f"\n优化完成：{len(result)} 资产，组合年化波动 {result.attrs['portfolio_vol_annual']:.2%}")
    print(f"LW 收缩强度 δ={result.attrs['lw_shrinkage']:.3f}，平均相关 ρ̄={result.attrs['mean_correlation']:.3f}")
    print(f"风险贡献比 max/min={result.attrs['rc_max_over_min']:.4f}（1.0=完全等风险）")
    print(f"资金利用率={result.attrs['capital_utilization']:.2%}（<100% 说明期货带来杠杆空间）")
    print("\n权重与风险贡献：")
    print(result[["symbol", "asset_type", "weight", "risk_contribution_pct",
                  "volatility", "margin_rate"]].to_string(index=False))

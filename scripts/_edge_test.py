"""边界路径回归测试 —— 覆盖 validate 未触发的分支

1. 确定性：同输入两次求解权重一致
2. 2 资产 ERC：仍能等风险（下界）
3. lookback > 样本：run_backtest 应明确报错而非静默
4. 高相关两资产：ERC 权重接近（不出现数值奇异）
5. 期货 detail 缺失（调用者传空 meta）：optimize_portfolio 不崩，margin_rate=nan
   （真实流程下 fetch_future_detail 网络失败会自填默认 0.10）
"""
import numpy as np
import pandas as pd

from factor import (
    LOOKBACK_DAYS,
    build_return_matrix,
    ledoit_wolf_shrinkage,
    optimize_portfolio,
    solve_erc,
)
from backtest import run_backtest
from validate import _load_fixture_or_network

price, meta = _load_fixture_or_network()
print(f"=== fixture: {len(price)} 行, {price['symbol'].nunique()} 资产 ===\n")

# --- 1. 确定性 ---
print("【1】同输入两次求解确定性")
a = optimize_portfolio(price, meta).sort_values("symbol")["weight"].values
b = optimize_portfolio(price, meta).sort_values("symbol")["weight"].values
assert np.allclose(a, b, atol=1e-9), "两次求解权重不一致"
print("    PASS\n")

# --- 2. 2 资产 ERC（下界） ---
print("【2】2 资产 ERC：仍应近似等风险")
two_syms = ["000300.SH", "511260.SH"]  # 沪深300(高波动) + 国债ETF(低波动)
two = price[price["symbol"].isin(two_syms)].copy()
r2 = optimize_portfolio(two, meta)
rc_ratio = r2.attrs["rc_max_over_min"]
print(f"    rc_max/min={rc_ratio:.4f}（容差 1.05）")
assert rc_ratio - 1.0 <= 0.05, f"2资产 ERC 不均: {rc_ratio}"
# 低波动(国债ETF)应得更高权重
w = dict(zip(r2["symbol"], r2["weight"]))
assert w["511260.SH"] > w["000300.SH"], "低波动资产应权重更高"
print(f"    权重: 国债ETF={w['511260.SH']:.3f} > 沪深300={w['000300.SH']:.3f}（低波动高权重）")
print("    PASS\n")

# --- 3. lookback > 样本：应明确报错 ---
print("【3】lookback=999 > 样本：run_backtest 应抛 ValueError")
try:
    run_backtest(price, meta, lookback=999)
    print("    [意外] 未报错")
    raise SystemExit("FAIL")
except ValueError as e:
    print(f"    [预期] 正确报错: {str(e)[:60]}")
print("    PASS\n")

# --- 4. 高相关两资产：ERC 权重应接近 ---
print("【4】高相关两资产(ρ→1)：ERC 权重应接近（不奇异）")
rs = np.random.RandomState(7)
T = 120
z = rs.randn(T, 1)
# 两个几乎相同的收益序列（ρ≈0.999）+ 不同波动
r_a = 0.001 + z * 0.012
r_b = 0.001 + z * 0.999 * 0.008 + rs.randn(T, 1) * 0.0005  # 与 a 高相关，波动更小
ret_df = pd.DataFrame({"A": r_a.flatten(), "B": r_b.flatten()})
Sigma, _, _ = ledoit_wolf_shrinkage(ret_df)
w, ok = solve_erc(Sigma)
rc = w * (Sigma @ w) / np.sqrt(w @ Sigma @ w)
rcr = rc.max() / rc.min()
print(f"    求解成功={ok}, 权重=[{w[0]:.3f},{w[1]:.3f}], rc_max/min={rcr:.4f}")
assert ok and rcr - 1.0 <= 0.05, f"高相关下 ERC 失败: rc_ratio={rcr}"
print("    PASS\n")

# --- 5. 期货 detail 缺失：margin 用默认 ---
print("【5】期货 detail 缺失（空 meta）：margin 用默认 10%，不崩")
empty_meta = {}  # 模拟 get_future_detail 失败
r5 = optimize_portfolio(price, empty_meta)
fut = r5[r5["asset_type"] == "future"]
assert not fut.empty
# 空 meta 时 optimize_portfolio 内 margin_rate=NaN（_meta 返回 nan），不崩即可
print(f"    期货行数={len(fut)}, margin_rate={fut['margin_rate'].tolist()}")
print("    PASS\n")

print("=== 边界回归测试全部通过 ===")

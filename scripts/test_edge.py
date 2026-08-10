"""边界路径回归测试 —— 覆盖 validate 未触发的分支（pytest 版）

1. 确定性：同输入两次求解权重一致
2. 2 资产 ERC：仍能等风险（下界）
3. lookback > 样本：run_backtest 应明确报错而非静默
4. 高相关两资产：ERC 权重接近（不出现数值奇异）
5. 期货 detail 缺失（调用者传空 meta）：optimize_portfolio 不崩，margin_rate=nan
   （真实流程下 fetch_future_detail 网络失败会自填默认 0.10）

运行：cd scripts && pytest test_edge.py -v（conftest 默认 PANDA_DATA_OFFLINE=1 读 fixtures）
"""
import numpy as np
import pandas as pd
import pytest

from factor import (
    ledoit_wolf_shrinkage,
    optimize_portfolio,
    solve_erc,
)
from backtest import run_backtest
from validate import _load_fixture_or_network


@pytest.fixture(scope="module")
def price_meta():
    """离线 fixture 价格+期货元信息（conftest 已默认 PANDA_DATA_OFFLINE=1，读 fixtures 不联网）"""
    return _load_fixture_or_network()


def test_determinism(price_meta):
    """【1】同输入两次求解确定性"""
    price, meta = price_meta
    a = optimize_portfolio(price, meta).sort_values("symbol")["weight"].values
    b = optimize_portfolio(price, meta).sort_values("symbol")["weight"].values
    assert np.allclose(a, b, atol=1e-9), "两次求解权重不一致"


def test_two_asset_erc(price_meta):
    """【2】2 资产 ERC：仍应近似等风险，低波动资产权重更高"""
    price, meta = price_meta
    two_syms = ["000300.SH", "511260.SH"]  # 沪深300(高波动) + 国债ETF(低波动)
    two = price[price["symbol"].isin(two_syms)].copy()
    r2 = optimize_portfolio(two, meta)
    rc_ratio = r2.attrs["rc_max_over_min"]
    assert rc_ratio - 1.0 <= 0.05, f"2资产 ERC 不均: {rc_ratio}"
    # 低波动(国债ETF)应得更高权重
    w = dict(zip(r2["symbol"], r2["weight"]))
    assert w["511260.SH"] > w["000300.SH"], "低波动资产应权重更高"


def test_lookback_exceeds_sample_raises(price_meta):
    """【3】lookback=999 > 样本：run_backtest 应抛 ValueError（不静默）"""
    price, meta = price_meta
    with pytest.raises(ValueError):
        run_backtest(price, meta, lookback=999)


def test_high_correlation_erc():
    """【4】高相关两资产(ρ→1)：ERC 权重应接近（不奇异）"""
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
    assert ok and rcr - 1.0 <= 0.05, f"高相关下 ERC 失败: rc_ratio={rcr}"


def test_future_detail_missing(price_meta):
    """【5】期货 detail 缺失（空 meta）：margin 用默认，不崩"""
    price, _ = price_meta
    empty_meta = {}  # 模拟 get_future_detail 失败
    r5 = optimize_portfolio(price, empty_meta)
    fut = r5[r5["asset_type"] == "future"]
    assert not fut.empty

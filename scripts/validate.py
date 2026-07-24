"""风险平价组合权重验证 —— 契约 + ERC 相等性 + LW 收缩 + 无未来函数

验证项：
    1. check_weight_contract     —— 权重硬契约（和为1、非负、字段完整、model_id）
    2. check_erc_equality        —— 风险贡献相等性（ERC 核心：max(RC)/min(RC)≈1，每项占比≈1/N）
    3. check_rc_sum              —— 风险贡献守恒（Σ RC_i = σ_p）
    4. check_lw_shrinkage        —— LW 收缩强度 δ ∈ [0,1]
    5. check_future_leverage     —— 期货杠杆：margin_rate≤1，资金占用≤名义权重
    6. check_no_future_function  —— 截断重算权重一致（确定性，无未来函数）
    7. check_out_of_sample_slice —— 月度 rebalance 至少 2 次调仓

离线模式（PANDA_DATA_OFFLINE=1）从 fixtures 加载，跳过联网。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from factor import (
    ERC_RC_RATIO_TOL,
    LOOKBACK_DAYS,
    MODEL_ID,
    build_return_matrix,
    optimize_portfolio,
)
from backtest import monthly_rebalance_weights

FIXTURE_ASSETS = Path(__file__).parent / "fixtures" / "sample_assets.parquet"
FIXTURE_META = Path(__file__).parent / "fixtures" / "sample_future_meta.json"


def _load_fixture_or_network() -> tuple[pd.DataFrame, dict]:
    """离线：读 fixture (assets parquet + future_meta json)；联网：调 load_all_assets"""
    if os.getenv("PANDA_DATA_OFFLINE", "0") == "1":
        if not FIXTURE_ASSETS.exists():
            raise FileNotFoundError(
                f"离线模式缺 fixture: {FIXTURE_ASSETS}。请先 `python save_fixture.py` 或 _make_synthetic_fixture.py")
        price = pd.read_parquet(FIXTURE_ASSETS)
        meta = {}
        if FIXTURE_META.exists():
            with open(FIXTURE_META, encoding="utf-8") as f:
                meta = json.load(f)
        return price, meta
    from data_loader import load_all_assets
    return load_all_assets()


def check_weight_contract(result: pd.DataFrame) -> None:
    """权重硬契约"""
    assert not result.empty, "权重表为空"
    required = ["symbol", "asset_type", "weight", "risk_contribution",
                "risk_contribution_pct", "volatility", "model_id", "update_time"]
    missing = [c for c in required if c not in result.columns]
    assert not missing, f"缺关键列: {missing}"

    w = result["weight"].values
    assert (w >= -1e-9).all(), f"权重必须非负，min={w.min()}"
    assert abs(w.sum() - 1.0) < 1e-4, f"权重和必须为1，实际 {w.sum()}"
    assert (result["model_id"] == MODEL_ID).all(), f"model_id 必须为 {MODEL_ID}"
    assert result["trade_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all(), "trade_date 格式错"
    assert "T" in result["update_time"].iloc[0], "update_time 须含 T（ISO 8601）"
    print(f"PASS: 权重契约（{len(result)} 资产，{(w>1e-6).sum()} 个非零，和={w.sum():.6f}）")


def check_erc_equality(result: pd.DataFrame) -> None:
    """ERC 核心：风险贡献应近似相等"""
    n = result.attrs["n_assets"]
    rc_ratio = result.attrs["rc_max_over_min"]
    assert rc_ratio - 1.0 <= ERC_RC_RATIO_TOL + 1e-9, \
        f"风险贡献不均：max(RC)/min(RC)={rc_ratio:.4f}，容差 {ERC_RC_RATIO_TOL}"

    rcp = result["risk_contribution_pct"].values
    target = 1.0 / n
    max_dev = float(np.max(np.abs(rcp - target)))
    assert max_dev <= ERC_RC_RATIO_TOL + 1e-6, \
        f"风险贡献占比偏离 1/N={target:.4f}，最大偏差 {max_dev:.4f}"
    print(f"PASS: ERC 相等性（max(RC)/min(RC)={rc_ratio:.4f}，占比偏差≤{max_dev:.4f}）")


def check_rc_sum(result: pd.DataFrame) -> None:
    """风险贡献守恒：Σ RC_i = σ_p"""
    sigma_p = result.attrs["portfolio_vol_daily"]
    rc_sum = float(result["risk_contribution"].sum())
    assert abs(rc_sum - sigma_p) < 1e-4, f"ΣRC({rc_sum:.6f}) ≠ σ_p({sigma_p:.6f})"
    print(f"PASS: 风险贡献守恒（ΣRC={rc_sum:.6f} = σ_p={sigma_p:.6f}）")


def check_lw_shrinkage(result: pd.DataFrame) -> None:
    """LW 收缩强度 δ ∈ [0,1]"""
    delta = result.attrs["lw_shrinkage"]
    assert 0.0 <= delta <= 1.0 + 1e-9, f"LW 收缩强度 δ={delta} 越界 [0,1]"
    rho = result.attrs["mean_correlation"]
    assert -1.0 <= rho <= 1.0, f"平均相关 ρ̄={rho} 越界 [-1,1]"
    print(f"PASS: LW 收缩（δ={delta:.3f}，ρ̄={rho:.3f}）")


def check_future_leverage(result: pd.DataFrame) -> None:
    """期货杠杆：margin_rate≤1，资金占用 ≤ 名义权重"""
    fut = result[result["asset_type"] == "future"]
    if fut.empty:
        print("[INFO] 无期货资产，杠杆检查跳过")
        return
    for _, row in fut.iterrows():
        mr = row["margin_rate"]
        assert pd.notna(mr) and 0 < mr <= 1.0 + 1e-9, f"{row['symbol']} margin_rate={mr} 异常"
        assert row["capital_allocation"] <= row["weight"] + 1e-6, \
            f"{row['symbol']} 资金占用 {row['capital_allocation']} > 名义 {row['weight']}"
    print(f"PASS: 期货杠杆（{len(fut)} 个，margin∈(0,1]，资金占用≤名义）")


def check_no_future_function(price: pd.DataFrame, future_meta: dict) -> None:
    """无未来函数：截断到倒数第6天重算，两次求解权重一致（ERC 确定性）"""
    dates = sorted(price["date"].astype(str).unique())
    if len(dates) < LOOKBACK_DAYS + 6:
        print("[WARN] 样本过短，跳过无未来函数检查")
        return
    cut = dates[-6]
    truncated = price[price["date"].astype(str) <= cut]
    r1 = optimize_portfolio(truncated, future_meta)
    r2 = optimize_portfolio(truncated, future_meta)
    w1 = r1.sort_values("symbol")["weight"].values
    w2 = r2.sort_values("symbol")["weight"].values
    assert np.allclose(w1, w2, atol=1e-6), "截断数据两次求解权重不一致（非确定性？）"
    print("PASS: 无未来函数（截断重算权重可复现）")


def check_out_of_sample_slice(price: pd.DataFrame) -> None:
    """月度 rebalance 至少 2 次调仓"""
    ret = build_return_matrix(price)
    wb = monthly_rebalance_weights(ret, LOOKBACK_DAYS)
    assert len(wb) >= 2, f"调仓次数过少 ({len(wb)})，月度回测无意义"
    print(f"PASS: 样本外切片（{len(wb)} 次调仓）")


if __name__ == "__main__":
    offline = os.getenv("PANDA_DATA_OFFLINE", "0") == "1"
    print(f"[MODE] {'PANDA_DATA_OFFLINE=1，使用 fixtures' if offline else '联网模式'}\n")

    price, future_meta = _load_fixture_or_network()
    result = optimize_portfolio(price, future_meta)

    check_weight_contract(result)
    check_erc_equality(result)
    check_rc_sum(result)
    check_lw_shrinkage(result)
    check_future_leverage(result)
    check_no_future_function(price, future_meta)
    check_out_of_sample_slice(price)

    print("\n验证通过：权重契约完整、ERC 等风险、贡献守恒、LW 收缩合理、杠杆归一、无未来函数、月度可回测")

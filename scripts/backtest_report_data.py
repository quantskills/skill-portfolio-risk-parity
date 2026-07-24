"""风险平价回测数据准备 —— 保留时序 + 离线 stub/parquet fallback 注入

职责：跑月度 rebalance 回测，把曲线/权重/相关矩阵转成 JSON 友好结构供 report.py 渲染。
离线模式（PANDA_DATA_OFFLINE=1）注入 panda_data 存根 + parquet fallback，无需凭证。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor import (  # noqa: E402
    LOOKBACK_DAYS,
    build_return_matrix,
    ledoit_wolf_shrinkage,
    optimize_portfolio,
)
from backtest import run_backtest  # noqa: E402
from validate import _load_fixture_or_network  # noqa: E402

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _is_offline() -> bool:
    return os.getenv("PANDA_DATA_OFFLINE", "0") == "1"


def _inject_panda_data_stub_for_offline() -> None:
    """离线模式注入 panda_data 存根，绕过联网（策略同 CVaR skill）

    fixture 已含 date/symbol/close/asset_type，存根不会被实际调用（数据走 parquet 缓存）。
    """
    if not _is_offline():
        return

    import types
    if "panda_data" not in sys.modules:
        sys.modules["panda_data"] = types.ModuleType("panda_data")

    pd_mod = sys.modules["panda_data"]

    def _noop(*args, **kwargs):
        return pd.DataFrame()

    for name in ("init_token", "get_index_daily", "get_future_daily",
                 "get_stock_daily", "get_future_detail"):
        if not hasattr(pd_mod, name):
            setattr(pd_mod, name, _noop)


def _patch_pd_read_parquet_fallback() -> None:
    """pyarrow 读旧版 parquet 失败时回退 fastparquet（策略同 CVaR skill）"""
    _orig = pd.read_parquet

    def _patched(path, *args, **kwargs):
        try:
            return _orig(path, *args, **kwargs)
        except Exception:
            return _orig(path, *args, engine="fastparquet")

    pd.read_parquet = _patched


def _series_to_points(s: pd.Series, value_key: str = "value") -> list[dict]:
    """pd.Series → [{date, value_key}, ...]（便于 JSON 序列化）"""
    return [{"date": str(d), value_key: float(v)} for d, v in s.items()]


def run_backtest_with_series(offline: bool | None = None) -> dict:
    """跑回测，保留时序曲线 + 末次权重表 + 样本相关矩阵"""
    if offline is None:
        offline = _is_offline()
    if offline:
        os.environ["PANDA_DATA_OFFLINE"] = "1"
        _inject_panda_data_stub_for_offline()
        _patch_pd_read_parquet_fallback()

    price, future_meta = _load_fixture_or_network()
    bt = run_backtest(price, future_meta, lookback=LOOKBACK_DAYS)
    curves = bt["curves"]

    # 末次权重表（带风险贡献）—— 单期 optimize_portfolio 给出全样本 ERC 权重
    weight_table = optimize_portfolio(price, future_meta)

    # 样本相关矩阵（LW 收缩前）—— 用于热图展示资产相关性结构
    ret = build_return_matrix(price)
    corr = ret.corr()

    data = {
        "metrics": bt["metrics"],
        "rp_curve": _series_to_points(curves["rp_curve"], "cum_return"),
        "eq_curve": _series_to_points(curves["eq_curve"], "cum_return"),
        "rp_drawdown": _series_to_points(curves["rp_drawdown"], "drawdown"),
        "n_rebalances": len(bt["weights_by_date"]),
        "weights": [
            {"symbol": r["symbol"], "asset_type": r["asset_type"], "name": r["name"],
             "weight": r["weight"], "risk_contribution_pct": r["risk_contribution_pct"],
             "volatility": r["volatility"],
             "margin_rate": (None if pd.isna(r["margin_rate"]) else r["margin_rate"])}
            for _, r in weight_table.iterrows()
        ],
        "portfolio_vol_annual": weight_table.attrs["portfolio_vol_annual"],
        "lw_shrinkage": weight_table.attrs["lw_shrinkage"],
        "mean_correlation": weight_table.attrs["mean_correlation"],
        "rc_max_over_min": weight_table.attrs["rc_max_over_min"],
        "capital_utilization": weight_table.attrs["capital_utilization"],
        "corr_matrix": corr.values.tolist(),
        "corr_labels": list(corr.columns),
    }
    if "bench_curve" in curves:
        data["bench_curve"] = _series_to_points(curves["bench_curve"], "cum_return")
    return data


def save_backtest_result(data: dict, json_path: Path | str) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    d = run_backtest_with_series()
    out = REPORTS_DIR / "backtest_result.json"
    save_backtest_result(d, out)
    print(f"[OK] 保存回测中间结果 → {out}")

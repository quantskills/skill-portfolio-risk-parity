"""合成 fixture（无需联网，供离线端到端验证）

构造 9 资产 × 260 交易日的合成日线，使 ERC 有意义：
    - 不同资产类别波动率差异显著（债券低、权益/商品高），低波动资产应获更高权重
    - 同标的簇高相关（沪深300指数+ETF、黄金期货+ETF ρ≈0.92）—— 模拟标的重叠
    - 跨类别低相关
    - 期货元信息（margin_rate / multiplier）单存 JSON
固定 RandomState(42) 保证可复现。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# (symbol, asset_type, 日波动率, 日均值)
ASSETS = [
    ("000300.SH", "index", 0.012, 0.00030),    # 沪深300（权益，高波动）
    ("000905.SH", "index", 0.013, 0.00030),    # 中证500
    ("000016.SH", "index", 0.010, 0.00030),    # 上证50（超大盘权益）
    ("CU_DOMINANT.SHF", "future", 0.015, 0.00020),  # 铜（商品）
    ("RB_DOMINANT.SHF", "future", 0.016, 0.00020),  # 螺纹钢
    ("AU_DOMINANT.SHF", "future", 0.010, 0.00030),  # 黄金
    ("510300.SH", "fund", 0.012, 0.00030),      # 沪深300ETF
    ("511260.SH", "fund", 0.003, 0.00010),      # 国债ETF
    ("518880.SH", "fund", 0.010, 0.00030),      # 黄金ETF
]
N = len(ASSETS)
T = 260

# 同标的对（高相关）：沪深300(0,6)、黄金(5,8)；上证50无对应 ETF，不配对
SAME_UNDERLYING = {0: 6, 6: 0, 5: 8, 8: 5}


def _trade_dates(start: date, n: int) -> list[str]:
    """从 start 起的 n 个交易日（跳过周末），返回 YYYYMMDD 字符串列表"""
    out, d, cnt = [], start, 0
    while cnt < n:
        if d.weekday() < 5:  # 周一~周五
            out.append(d.strftime("%Y%m%d"))
            cnt += 1
        d += timedelta(days=1)
    return out


def build_corr() -> np.ndarray:
    """构造相关矩阵：同标的 ρ=0.92，同类 ρ=0.50，跨类 ρ=0.15"""
    C = np.eye(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if j == SAME_UNDERLYING.get(i):
                C[i, j] = 0.92
            elif ASSETS[i][1] == ASSETS[j][1]:
                C[i, j] = 0.50
            else:
                C[i, j] = 0.15
    C = (C + C.T) / 2.0
    # 最近 PSD 投影（防数值非正定导致 Cholesky 失败）
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C = (V * w) @ V.T
    C = (C + C.T) / 2.0
    return C


def main() -> None:
    rs = np.random.RandomState(42)
    C = build_corr()
    L = np.linalg.cholesky(C)
    sigmas = np.array([a[2] for a in ASSETS])
    mus = np.array([a[3] for a in ASSETS])

    # 生成相关日收益
    Z = rs.randn(T, N)
    ret = mus + (Z @ L.T) * sigmas

    # 累计成价格（起点 100）
    prices = np.zeros((T, N))
    prices[0] = 100.0
    for t in range(1, T):
        prices[t] = prices[t - 1] * (1.0 + ret[t])

    dates = _trade_dates(date(2025, 7, 21), T)
    rows = []
    for t in range(T):
        for i, (sym, atype, _, _) in enumerate(ASSETS):
            rows.append({"date": dates[t], "symbol": sym,
                         "close": round(float(prices[t, i]), 4), "asset_type": atype})
    df = pd.DataFrame(rows)

    future_meta = {
        "CU_DOMINANT.SHF": {"multiplier": 5.0, "margin_rate": 0.10,
                            "name": "沪铜主力", "product": "Commodity"},
        "RB_DOMINANT.SHF": {"multiplier": 10.0, "margin_rate": 0.10,
                            "name": "螺纹钢主力", "product": "Commodity"},
        "AU_DOMINANT.SHF": {"multiplier": 1000.0, "margin_rate": 0.08,
                            "name": "黄金主力", "product": "Commodity"},
    }

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FIXTURE_DIR / "sample_assets.parquet", index=False)
    with open(FIXTURE_DIR / "sample_future_meta.json", "w", encoding="utf-8") as f:
        json.dump(future_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 合成 fixture 已生成：{len(df)} 行, {N} 资产, {T} 交易日")
    print(f"     波动率: {[a[2] for a in ASSETS]}")
    print(f"     → {FIXTURE_DIR / 'sample_assets.parquet'}")
    print(f"     → {FIXTURE_DIR / 'sample_future_meta.json'}")


if __name__ == "__main__":
    main()

"""联网拉三类资产真实样本，存 fixtures（供离线验证复现）

用法（需配置凭证）：
    export PANDA_DATA_USERNAME=...
    export PANDA_DATA_PASSWORD=...
    python save_fixture.py

产物：
    fixtures/sample_assets.parquet       三类资产合并长表（date/symbol/close/asset_type）
    fixtures/sample_future_meta.json    期货元信息 {symbol: {multiplier, margin_rate, name, product}}
"""
from __future__ import annotations

import json
from pathlib import Path

from data_loader import load_all_assets

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def main() -> None:
    # 默认资产池 + 默认回溯 1 年（够月度 rebalance 的 60 日窗口）
    price, future_meta = load_all_assets()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    assets_path = FIXTURE_DIR / "sample_assets.parquet"
    meta_path = FIXTURE_DIR / "sample_future_meta.json"
    price.to_parquet(assets_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(future_meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] fixture 已保存：")
    print(f"     {assets_path}  ({len(price)} 行, {price['symbol'].nunique()} 资产)")
    print(f"     {meta_path}    ({len(future_meta)} 个期货元信息)")
    print(f"     资产类型分布: {price['asset_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

"""风险平价组合优化 —— 数据加载层（data_loader）

从 panda_data SDK 拉取三类资产日线 + 期货元信息，统一为长表(date/symbol/close/asset_type)。
含本地 parquet 缓存（先查本地→未命中联网→写缓存）、交易日历（get_trade_cal）定日期、
字段标准化（防御接口字段名差异）。本层只负责"取数与清洗"，因子计算见 factor.py。

数据源：
    get_index_daily   —— 指数日线（权益/债券大盘）
    get_future_daily  —— 期货主力日线（商品/金融期货）
    get_fund_daily    —— 场内 ETF 日线（510300/511260/518880，OHLCV；⚠️日期跨度 ≤1年）
    get_future_detail —— 期货合约元信息（contract_multiplier / margin_rate）
    get_trade_cal     —— 交易日历（确定最近交易日，精确处理周末/假日）

PandaAI data 实现说明详见 references/data_guide.md。
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import panda_data

# === 全局常量 ===================================================================
BATCH_SIZE = 1                  # 单次 API 调用品种数（对齐其它 skill，避免服务器超限）

# === 默认资产池（3+3+3 九资产，跨权益/债券/商品）=================================
# 三类资产接口：指数走 get_index_daily，场内 ETF 走 get_fund_daily，商品期货走 get_future_daily。
INDEX_POOL = [
    "000300.SH",   # 沪深300（大盘权益）
    "000905.SH",   # 中证500（中盘权益）
    "000016.SH",   # 上证50（超大盘权益；原 000012 国债指数波动 0.72% 过低，
                   #   ERC 给它 65.7% 权重致组合债券化，改权益标的让固收仅剩 511260 国债ETF）
]
FUTURE_POOL = [
    "CU_DOMINANT.SHF",  # 铜主力（有色/商品）
    "RB_DOMINANT.SHF",  # 螺纹钢主力（黑色/商品）
    "AU_DOMINANT.SHF",  # 黄金主力（贵金属）
]
# 基金类：场内 ETF 走 get_fund_daily（510300/511260/518880 均可取，返回 OHLCV）。
# ⚠️ get_fund_daily 日期跨度 ≤1年（index/future 为 ≤5年），故回测窗口 ~1年。
FUND_POOL = [
    "510300.SH",   # 沪深300ETF（权益）
    "511260.SH",   # 国债ETF（固收）
    "518880.SH",   # 黄金ETF（黄金/商品）
]

# === 本地缓存路径 ===============================================================
_CACHE_DIR = Path(__file__).parent / "fixtures"
_CACHE_FILES = {
    "index": _CACHE_DIR / "cache_index.parquet",
    "future": _CACHE_DIR / "cache_future.parquet",
    "fund": _CACHE_DIR / "cache_fund.parquet",
}


def _get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"请先设置环境变量 {name}")
    return value


# === 本地缓存层（策略同 CVaR skill：先查本地 parquet → 未命中联网 → 写缓存）======
def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_parquet(path, engine="fastparquet")  # pyarrow 不兼容时回退
        except Exception:
            return None


def _load_with_cache(cache_path, start_date_8, end_date_8, symbols, fetch_fn) -> pd.DataFrame:
    """通用缓存：命中(日期覆盖+品种覆盖)直接返回，否则联网获取后合并写缓存"""
    cached = _read_cache(cache_path)
    if cached is not None and not cached.empty:
        cdates = cached["date"].astype(str)
        if (cdates.min() <= start_date_8 and cdates.max() >= end_date_8
                and set(symbols).issubset(set(cached["symbol"].unique()))):
            mask = ((cached["date"].astype(str) >= start_date_8)
                    & (cached["date"].astype(str) <= end_date_8)
                    & cached["symbol"].isin(symbols))
            result = cached[mask]
            if not result.empty:
                print(f"[CACHE] 命中本地缓存 ({len(result)} 行) → {cache_path.name}")
                return result.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"[CACHE] 未命中，联网获取 → {cache_path.name}")
    result = fetch_fn()
    if result is not None and not result.empty:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        merged = (pd.concat([cached, result], ignore_index=True).drop_duplicates(
            subset=["date", "symbol"], keep="last") if cached is not None and not cached.empty else result)
        merged.to_parquet(cache_path, index=False)
        print(f"[CACHE] 已写入 ({len(merged)} 行) → {cache_path}")
    return result.sort_values(["symbol", "date"]).reset_index(drop=True)


def _chunked(items, size=BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i:i + size]


_TRADE_DATE_CACHE: str | None = None


def _latest_trade_date(exchange: str = "SH") -> str:
    """从交易日历 get_trade_cal 获取最近可用交易日，返回 'YYYY-MM-DD'。

    16:00 收盘后含当天；收盘前取严格早于今天的最近交易日（当天数据未完成）。
    用官方交易日历精确处理周末与法定假日（优于本地 weekday 推断）。
    结果模块级缓存，单次运行内复用，避免重复联网。
    """
    global _TRADE_DATE_CACHE
    if _TRADE_DATE_CACHE is not None:
        return _TRADE_DATE_CACHE
    _init_token()
    now = datetime.now()
    today8 = now.strftime("%Y%m%d")
    start8 = (now - timedelta(days=20)).strftime("%Y%m%d")  # 窗口覆盖长假
    cal = panda_data.get_trade_cal(start_date=start8, end_date=today8,
                                    exchange=exchange, is_trading_day=1, fields=[])
    trades = sorted(cal["nature_date"].astype(str).tolist())
    after_close = now >= now.replace(hour=16, minute=0, second=0, microsecond=0)
    # 收盘前排除今天（数据未完成）；收盘后含今天
    cand = trades if after_close else [d for d in trades if d < today8]
    if not cand:  # 兜底
        cand = trades
    latest = cand[-1]
    _TRADE_DATE_CACHE = f"{latest[:4]}-{latest[4:6]}-{latest[6:8]}"
    return _TRADE_DATE_CACHE


def get_default_end_date() -> str:
    """默认结束日期：最近交易日（交易日历 get_trade_cal 获取，精确处理周末/假日）"""
    return _latest_trade_date()


def _resolve_date_range(start_date, end_date, lookback_days=350):
    # 350 天适配 get_fund_daily ≤1年限制；index/future 虽可取 5 年，但回测窗口
    # 由最短资产（ETF）决定，内连接后仍 ~350 天，统一短窗避免白拉多取。
    end_date = end_date or os.getenv("PANDA_DATA_END_DATE", get_default_end_date())
    if start_date is None:
        start_date = os.getenv("PANDA_DATA_START_DATE")
        if not start_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    return start_date, end_date


# === 字段标准化（防御接口字段名差异）============================================
def _normalize_price(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """统一出 date / symbol / close / asset_type 四列

    期货接口字段名为 date/symbol（已对齐），股票/指数字段可能是 code/ts_code、trade_date。
    """
    df = df.copy()
    rename = {}
    for src in ("code", "ts_code", "sec_code"):
        if src in df.columns and "symbol" not in df.columns:
            rename[src] = "symbol"
    for src in ("trade_date", "trade_dt", "datetime"):
        if src in df.columns and "date" not in df.columns:
            rename[src] = "date"
    if rename:
        print(f"[INFO] 字段映射: {rename}")
        df = df.rename(columns=rename)

    for col in ("date", "symbol", "close"):
        if col not in df.columns:
            raise ValueError(f"{asset_type} 数据缺字段 {col}，实际: {sorted(df.columns)}")

    df["date"] = df["date"].astype(str).str.replace("-", "", regex=False)
    df["symbol"] = df["symbol"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["asset_type"] = asset_type
    return df[["date", "symbol", "close", "asset_type"]]


# === 数据加载（三类资产）========================================================
def _init_token():
    panda_data.init_token(username=_get_env("PANDA_DATA_USERNAME"),
                          password=_get_env("PANDA_DATA_PASSWORD"))


def load_index_data(symbols=None, start_date=None, end_date=None) -> pd.DataFrame:
    """指数日线 → date/symbol/close/asset_type=index"""
    _init_token()
    symbols = symbols or INDEX_POOL
    start_date, end_date = _resolve_date_range(start_date, end_date)
    s8, e8 = start_date.replace("-", ""), end_date.replace("-", "")
    print(f"获取指数: {symbols}, {start_date} ~ {end_date}")

    def _fetch():
        parts = []
        for batch in _chunked(symbols):
            df = panda_data.get_index_daily(start_date=s8, end_date=e8, symbol=batch)
            if df is not None and not df.empty:
                parts.append(_normalize_price(df, "index"))
        if not parts:
            raise ValueError(f"未获取到指数数据 ({symbols})")
        return pd.concat(parts, ignore_index=True)

    return _load_with_cache(_CACHE_FILES["index"], s8, e8, symbols, _fetch)


def load_fund_data(symbols=None, start_date=None, end_date=None) -> pd.DataFrame:
    """场内 ETF 日线（510300/511260/518880，走 get_fund_daily）

    get_fund_daily 返回 ETF 的 OHLCV（symbol/date/open/high/low/close/volume 等 15 列），
    _normalize_price 直接兼容（date 已为 8 位字符串）。
    ⚠️ get_fund_daily 日期跨度 ≤1年（错误码 100008），_resolve_date_range 默认 350 天已适配。
    """
    _init_token()
    symbols = symbols or FUND_POOL
    start_date, end_date = _resolve_date_range(start_date, end_date)
    s8, e8 = start_date.replace("-", ""), end_date.replace("-", "")
    print(f"获取基金(ETF): {symbols}, {start_date} ~ {end_date}")

    def _fetch():
        parts = []
        for batch in _chunked(symbols):
            df = panda_data.get_fund_daily(start_date=s8, end_date=e8, symbol=batch, fields=[])
            if df is not None and not df.empty:
                parts.append(_normalize_price(df, "fund"))
        if not parts:
            raise ValueError(f"未获取到基金(ETF)数据 ({symbols})")
        return pd.concat(parts, ignore_index=True)

    return _load_with_cache(_CACHE_FILES["fund"], s8, e8, symbols, _fetch)


def load_future_data(symbols=None, start_date=None, end_date=None) -> tuple[pd.DataFrame, dict]:
    """期货主力日线 → date/symbol/close/asset_type=future + margin/multiplier 元信息

    期货是杠杆资产：实际占用资金 = 名义敞口 × margin_rate。风险平价权重统一为
    【名义敞口权重】（pct_change 天然是名义收益率口径），margin 仅用于报告资金占用。
    contract_multiplier / margin_rate 通过 get_future_detail 获取。
    """
    _init_token()
    symbols = symbols or FUTURE_POOL
    start_date, end_date = _resolve_date_range(start_date, end_date)
    s8, e8 = start_date.replace("-", ""), end_date.replace("-", "")
    print(f"获取期货: {symbols}, {start_date} ~ {end_date}")

    def _fetch():
        parts = []
        for batch in _chunked(symbols):
            df = panda_data.get_future_daily(start_date=s8, end_date=e8, symbol=batch)
            if df is not None and not df.empty:
                parts.append(_normalize_price(df, "future"))
        if not parts:
            raise ValueError(f"未获取到期货数据 ({symbols})")
        return pd.concat(parts, ignore_index=True)

    price_df = _load_with_cache(_CACHE_FILES["future"], s8, e8, symbols, _fetch)

    # 期货合约元信息（乘数 / 保证金率）—— 杠杆归一化所需
    future_meta = fetch_future_detail(symbols)
    return price_df, future_meta


def fetch_future_detail(symbols: list[str]) -> dict:
    """调 get_future_detail 拿 contract_multiplier / margin_rate / name / product

    Returns: {symbol: {multiplier, margin_rate, name, product}}
    """
    _init_token()
    meta: dict[str, dict] = {}
    try:
        df = panda_data.get_future_detail(symbol=symbols, fields=[])
    except Exception as e:
        print(f"[WARN] get_future_detail 失败 ({e})，期货保证金/乘数用默认值 10%/1")
        return {s: {"multiplier": 1.0, "margin_rate": 0.10, "name": s, "product": ""}
                for s in symbols}

    if df is None or df.empty:
        return {s: {"multiplier": 1.0, "margin_rate": 0.10, "name": s, "product": ""}
                for s in symbols}

    for _, row in df.iterrows():
        sym = str(row.get("symbol", ""))
        if sym not in symbols:
            continue
        meta[sym] = {
            "multiplier": float(row.get("contract_multiplier", 1.0) or 1.0),
            "margin_rate": float(row.get("margin_rate", 0.10) or 0.10),
            "name": str(row.get("name", sym)),
            "product": str(row.get("product", "")),
        }
    # 未命中的补默认
    for s in symbols:
        if s not in meta:
            meta[s] = {"multiplier": 1.0, "margin_rate": 0.10, "name": s, "product": ""}
    return meta


def load_all_assets(index_pool=None, future_pool=None, fund_pool=None,
                    start_date=None, end_date=None) -> tuple[pd.DataFrame, dict]:
    """加载三类资产并纵向合并为一张长表 + 期货元信息"""
    index_df = load_index_data(index_pool, start_date, end_date)
    fund_df = load_fund_data(fund_pool, start_date, end_date)
    future_df, future_meta = load_future_data(future_pool, start_date, end_date)

    merged = pd.concat([index_df, future_df, fund_df], ignore_index=True)
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True), future_meta

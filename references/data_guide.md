# 数据指南（data_guide）

本 skill 数据层 `scripts/data_loader.py` 通过 **panda_data SDK** 拉取三类资产日线 + 期货元信息。
本文档说明各数据接口、字段、限制与缓存策略，供联网取数与离线验证参考。

> 因子计算见 `scripts/factor.py`；本文档只覆盖"取数与清洗"（data_loader 职责）。

---

## 1. 数据源接口总览

| 接口 | 用途 | 日期跨度上限 | 关键返回字段 |
|---|---|---|---|
| `get_index_daily` | 指数日线（权益/债券大盘） | ≤5年 | symbol / date / open / high / low / close / volume |
| `get_future_daily` | 期货主力日线（商品/金融期货） | ≤5年 | symbol / date / open / high / low / close |
| `get_fund_daily` | **场内 ETF 日线** | **≤1年**（错误码 100008） | symbol / date / open / high / low / close / volume（OHLCV 15 列） |
| `get_future_detail` | 期货合约元信息 | — | symbol / name / contract_multiplier / margin_rate / product |
| `get_trade_cal` | 交易日历 | — | exchange / is_trade / nature_date / pretrade_date / next_trade_date |

> ⚠️ **关键约束**：`get_fund_daily` 日期跨度 **≤1年**，是整个回测窗口的硬约束
> （最短资产决定窗口）。`_resolve_date_range` 默认 `lookback_days=350`（约 1 年自然日 ≈ 233 交易日）已适配。

---

## 2. 调用方式

```python
import panda_data
panda_data.init_token(username=..., password=...)

# 指数（5年）
df = panda_data.get_index_daily(start_date="20250101", end_date="20260101",
                                symbol=["000300.SH"])
# 场内 ETF（⚠️ ≤1年！）
df = panda_data.get_fund_daily(start_date="20250101", end_date="20260101",
                               symbol=["510300.SH"], fields=[])
# 期货主力
df = panda_data.get_future_daily(start_date="20250101", end_date="20260101",
                                 symbol=["CU_DOMINANT.SHF"])
# 期货合约元信息（乘数 / 保证金率）
detail = panda_data.get_future_detail(symbol=["CU_DOMINANT.SHF"], fields=[])
# 交易日历
cal = panda_data.get_trade_cal(start_date="20250101", end_date="20260101",
                               exchange="SH", is_trading_day=1, fields=[])
```

### 凭证（环境变量）

```bash
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
```

- `_get_env` 缺凭证时抛 `RuntimeError("请先设置环境变量 ...")`
- `_init_token` 在每个加载函数入口调用，无需手动 init

---

## 3. 默认资产池（3+3+3 九资产）

| 类 | 接口 | 标的 |
|---|---|---|
| 指数 | `get_index_daily` | 000300 沪深300 / 000905 中证500 / **000016 上证50** |
| 期货 | `get_future_daily` + `get_future_detail` | CU 铜 / RB 螺纹钢 / AU 黄金（主力） |
| ETF | `get_fund_daily` | 510300 沪深300ETF / **511260 国债ETF** / 518880 黄金ETF |

> 指数与 ETF 标的局部重叠（沪深300 / 黄金），形成高相关块 —— 正是 Ledoit-Wolf 收缩
> 要稳定的对象。资产池可经 `load_all_assets(index_pool=..., future_pool=..., fund_pool=...)` 覆盖。

**资产池演进**（详见 notes.md）：曾用国债指数 000012（波动 0.72%），ERC 给它 65.7% 权重致组合债券化；
改用 000016 上证50 后，固收仅剩 511260 国债ETF，组合波动从 2.19% 升至 6.39%（更均衡）。

---

## 4. 字段标准化

`_normalize_price(df, asset_type)` 将任意接口返回统一为 **4 列长表**：

| 列 | 类型 | 说明 |
|---|---|---|
| `date` | str（YYYYMMDD，8位） | 交易日期 |
| `symbol` | str | 资产代码 |
| `close` | float | 收盘价 |
| `asset_type` | str | `index` / `future` / `fund` |

防御接口字段名差异（自动 rename）：
- `code` / `ts_code` / `sec_code` → `symbol`
- `trade_date` / `trade_dt` / `datetime` → `date`

期货接口字段名已对齐（date/symbol），股票/指数接口字段名可能为 code/ts_code、trade_date。

---

## 5. 期货杠杆元信息

`fetch_future_detail(symbols)` 返回 `{symbol: {multiplier, margin_rate, name, product}}`：

| 字段 | 说明 |
|---|---|
| `contract_multiplier` | 合约乘数（仅"转手数"时需要；风险平价输出权重不强制） |
| `margin_rate` | 保证金率（**资金占用 = 名义权重 × margin_rate**） |
| `name` / `product` | 合约中文名 / 品种类别 |

- 网络失败或字段缺失时自填默认 `multiplier=1.0 / margin_rate=0.10`，保证流程不中断
- **权重口径**：风险平价输出**名义敞口权重**（`pct_change` 天然是名义收益率口径，与乘数无关），
> `margin_rate` 仅用于报告"资金占用"，不参与 ERC 求解

---

## 6. 本地缓存策略

`_load_with_cache(cache_path, start, end, symbols, fetch_fn)` —— 先查本地 → 未命中联网 → 写缓存：

- **缓存路径**：`scripts/fixtures/cache_{index,future,fund}.parquet`
- **命中条件**：缓存日期区间 ⊇ 请求区间 **且** 品种 ⊇ 请求品种
- **未命中**：联网获取 → 与旧缓存合并去重（按 date+symbol） → 写回 parquet
- **读取容错**：默认 pyarrow，失败回退 fastparquet

---

## 7. 交易日历定日期

`_latest_trade_date(exchange="SH")` 用 `get_trade_cal` 取最近可用交易日（精确处理周末/假日）：

- 16:00 收盘后含当天；收盘前取**严格早于今天**的最近交易日（当天数据未完成）
- 结果模块级缓存（`_TRADE_DATE_CACHE`），单次运行复用，避免重复联网
- `get_default_end_date()` 直接返回此值
- （曾因周日 7/19 取数失败，改用交易日历修正，详见 notes.md）

---

## 8. 离线 fixture 模式

`PANDA_DATA_OFFLINE=1` 时（`validate.py` / `report.py` / `analysis_report.py` / `_edge_test.py`）：

- 数据从 `fixtures/sample_assets.parquet` + `sample_future_meta.json` 读取
- `sample_assets.parquet` 由 `save_fixture.py` **联网拉真实数据**生成（9 资产 × 231 交易日）
- `_make_synthetic_fixture.py` 生成**合成 fixture**（RandomState=42）作离线基准备用
- 无需凭证，适合 CI / 端到端验证

---

## 9. 输出 Parquet 字段（对齐 Alpha 因子开发规则 V2 §9）

风险平价是**组合优化因子**（输出配置权重），非横截面选股因子（IC/IR/分层/score/signal 不适用）。
权重表字段与 V2 §9 标准的映射：

| V2 §9 标准字段 | 本 skill 权重表字段 | 说明 |
|---|---|---|
| `trade_date` | `trade_date` | 权重生成日 |
| `asset_type` | `asset_type` | index / future / fund |
| `ts_code` / `symbol` | `symbol` | 资产代码 |
| `factor_id` | `model_id` | RP1 |
| `factor_name` | `model_name` | 风险平价组合(ERC) |
| `factor_value` | `weight` | **名义敞口权重**（风险平价的"因子值"） |
| `data_version` | `data_version` | real-v1 |
| `update_time` | `update_time` | 生成时间（ISO 8601，可复现） |
| `score` / `rank` / `signal` / `confidence` | — | 横截面选股概念，组合优化不适用 |

额外字段（组合优化特有）：`risk_contribution` / `risk_contribution_pct` / `volatility` /
`margin_rate` / `contract_multiplier` / `capital_allocation`。

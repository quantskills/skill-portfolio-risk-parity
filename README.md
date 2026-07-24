# skill-portfolio-risk-parity

**简体中文** | [English](README.en.md)

> 风险平价（ERC）组合优化：手写 Ledoit-Wolf 收缩协方差稳定相关性，scipy SLSQP 求等风险贡献权重，支持指数/期货/ETF 三类资产与月度 rebalance。

<p align="center">
  <img alt="libraries" src="https://img.shields.io/badge/libraries-Risk%20Parity-blue">
  <img alt="model" src="https://img.shields.io/badge/model-RP1-brightgreen">
  <img alt="type" src="https://img.shields.io/badge/type-portfolio--risk-blue">
  <img alt="platform" src="https://img.shields.io/badge/platform-PandaAI-9cf">
  <img alt="solver" src="https://img.shields.io/badge/solver-scipy%20SLSQP-orange">
  <img alt="covariance" src="https://img.shields.io/badge/covariance-Ledoit--Wolf%20shrinkage-red">
  <img alt="status" src="https://img.shields.io/badge/status-active-brightgreen">
  <img alt="license" src="https://img.shields.io/badge/license-GPLv3-blue">
</p>

`skill-portfolio-risk-parity` 是一个风险平价组合优化 Skill，基于等风险贡献（ERC, Equal Risk Contribution）方法，
求每类资产对组合风险贡献相等的权重，并完成契约验证、月度 rebalance 回测与离线复现。

这个 Skill 适合用于：

- 跨资产配置（权益/债券/商品/贵金属）的均衡组合构建
- 需要稳定低波动、对相关性结构敏感的多资产组合
- 验证风险平价是否真的把风险均匀分摊到各类资产
- Claude Code 对话中触发确定性组合优化与回测

本 Skill 通过 `panda_data` SDK 拉取指数 / 期货 / ETF 三类资产日线，输出权重表、回测指标（年化波动/夏普/MDD）、
HTML 可视化报告以及离线 fixture，便于 CI 集成。

## 核心方法

- **等风险贡献（ERC）**：求权重使每个资产的风险贡献 RC_i 相等（= σ_p/N），`scipy.optimize.minimize(method="SLSQP")` 求解
- **Ledoit-Wolf 收缩协方差（手写）**：常数相关目标，稳定样本不足/高相关下的协方差矩阵，避免极端权重
- **杠杆资产归一**：期货按名义敞口权重统一（保证金仅用于资金占用报告），三类资产风险可比
- **月度 rebalance**：月末调仓，协方差窗口默认 60 交易日，只用过去信息（无未来函数）

## 仓库内容

| 文件 | 说明 |
|---|---|
| `SKILL.md` | Skill 契约文档（Agent 内部使用，含完整公式） |
| `scripts/factor.py` | ERC 权重求解入口（收益矩阵 → LW 收缩 → SLSQP → 权重表，可独立运行） |
| `scripts/data_loader.py` | 数据加载层（指数/期货/ETF 日线 + 期货元信息 + parquet 缓存 + 交易日历） |
| `scripts/validate.py` | 权重验证（7 项检查：契约/ERC 相等/贡献守恒/LW/杠杆/无未来函数/切片） |
| `scripts/backtest.py` | 月度 rebalance 回测（年化波动/夏普/MDD + 等权对比） |
| `scripts/backtest_report_data.py` | 回测时序数据包装层（HTML 报告数据源） |
| `scripts/report.py` | HTML 报告生成入口（4 图：净值/风险贡献/权重/相关性热图） |
| `scripts/analysis_report.py` | lookback 扫描 + LW 收缩效果对比 |
| `scripts/save_fixture.py` | 一次性生成离线测试 fixture（联网拉真实样本） |
| `scripts/_make_synthetic_fixture.py` | 生成合成 fixture（RandomState=42，离线基准备用） |
| `scripts/_edge_test.py` | 边界回归测试 |
| `scripts/fixtures/` | 离线测试数据（Parquet 格式） |
| `requirements.txt` | Python 依赖清单 |
| `references/data_guide.md` | PandaAI 数据接口参考 |
| `LICENSE` | GPLv3 协议 |
| `README.md` / `README.en.md` | 中英文 README |

## 目录结构

```text
skill-portfolio-risk-parity/
├── SKILL.md
├── README.md
├── README.en.md
├── LICENSE
├── requirements.txt
├── notes.md
├── references/
│   └── data_guide.md
├── scripts/
│   ├── factor.py                  # ERC 权重求解（LW 收缩 + SLSQP + optimize_portfolio + main）
│   ├── data_loader.py             # 数据加载层（load_index/fund/future + detail + 缓存 + 交易日历）
│   ├── validate.py                # 7 项检查（契约/ERC/守恒/LW/杠杆/无未来/切片）
│   ├── backtest.py                # 月度 rebalance 回测（年化波动/夏普/MDD + 等权对比）
│   ├── backtest_report_data.py    # 回测时序数据包装层（保留 curve / drawdown / rc 序列）
│   ├── report.py                  # HTML 报告（4 图 + 指标卡 + 权重明细）
│   ├── analysis_report.py         # lookback 扫描 + LW 收缩效果对比
│   ├── save_fixture.py            # 联网生成离线 fixture
│   ├── _make_synthetic_fixture.py # 合成 fixture（测试基线）
│   ├── _edge_test.py              # 边界回归测试
│   └── fixtures/
│       ├── sample_assets.parquet  # 离线资产数据
│       └── sample_future_meta.json # 期货元信息（margin/multiplier）
└── reports/                       # 报告产物（.gitignored）
```

## 数据要求

调用 `panda_data` 的输入契约：

| 资产类 | 接口 | 说明 |
|---|---|---|
| 指数 | `get_index_daily` | 权益/债券大盘（沪深300/中证500/上证50） |
| 期货 | `get_future_daily` + `get_future_detail` | 商品/金融期货主力合约（带 margin_rate / contract_multiplier） |
| ETF | `get_fund_daily` | 场内 ETF（510300/511260/518880），返回 OHLCV；⚠️日期跨度 ≤1年 |

### 默认资产池（3+3+3 九资产）

```
指数：  000300.SH(沪深300)  000905.SH(中证500)  000016.SH(上证50)
期货：  CU_DOMINANT.SHF(铜)  RB_DOMINANT.SHF(螺纹钢)  AU_DOMINANT.SHF(黄金)
ETF：   510300.SH(沪深300ETF)  511260.SH(国债ETF)  518880.SH(黄金ETF)
```

输入契约细则：

- **期货主力合约格式**：`XXX_DOMINANT.EXCHANGE`（如 `CU_DOMINANT.SHF`）
- **三类资产交易日历内连接对齐**（`dropna(how="any")`），期货夜盘/ETF 不同步会丢少量交易日
- **`_normalize_price` 防御字段差异**：`code`/`ts_code`→`symbol`、`trade_date`→`date`
- **默认回溯 1 年**（受 `get_fund_daily` ≤1年限制）；月度 rebalance 协方差窗口默认 60 交易日

## 快速开始

### 环境准备

```bash
export PANDA_DATA_USERNAME=your_username
export PANDA_DATA_PASSWORD=your_password

# 可选：控制数据区间
export PANDA_DATA_START_DATE=2025-08-01
export PANDA_DATA_END_DATE=2026-07-20
```

### 四步运行

```bash
cd scripts/

# 1. ERC 权重求解（单期）
python factor.py
# 输出：权重表（trade_date, symbol, weight, risk_contribution, risk_contribution_pct, ...）

# 2. 权重验证
python validate.py
# 输出：7 项检查（契约/ERC 相等/贡献守恒/LW 收缩/杠杆/无未来函数/切片）

# 3. 月度 rebalance 回测
python backtest.py
# 输出：RP 年化波动、夏普、MDD、调仓次数，及与等权组合对比

# 4. HTML 报告生成
python report.py
# 输出：reports/report.html（净值/风险贡献/权重/相关性热图 4 图）
```

### 离线模式（CI 友好）

```bash
# 1. 一次性生成 fixture（需联网 + 凭证）
python save_fixture.py
# 生成 scripts/fixtures/sample_assets.parquet + sample_future_meta.json

# 或用合成基线（无需联网）
python _make_synthetic_fixture.py

# 2. 后续验证无需联网
export PANDA_DATA_OFFLINE=1
python validate.py
python report.py --offline
python analysis_report.py --offline    # lookback 扫描
```

离线模式下若本地未装 `panda_data` SDK，会自动注入 stub 模块绕过顶层 import；
pyarrow 与旧版 parquet 不兼容时自动 fallback 到 `fastparquet` 引擎。

## 输入配置

| 环境变量 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `PANDA_DATA_USERNAME` | ✓ | — | PandaAI 账号 |
| `PANDA_DATA_PASSWORD` | ✓ | — | PandaAI 密码 |
| `PANDA_DATA_START_DATE` | — | 结束日期往前 350 天 | 起始日期（YYYY-MM-DD） |
| `PANDA_DATA_END_DATE` | — | 最近交易日（交易日历） | 结束日期（YYYY-MM-DD） |
| `PANDA_DATA_OFFLINE` | — | `0` | `1` 时启用离线模式，从 fixture 加载 |
| `PANDA_DATA_DEBUG` | — | `0` | `1` 时首次 API 调用打印 schema 探针 |

## 输出文件

`factor.py` 输出的权重表字段（每个资产一行）：

| 字段 | 说明 |
|---|---|
| `trade_date` | 权重生成日期（数据最新日，YYYY-MM-DD） |
| `asset_type` | 资产类型（index/future/etf） |
| `symbol` | 资产代码 |
| `name` | 名称（期货来自 detail） |
| `model_id` | 模型编号（固定为 `RP1`） |
| `model_name` | 模型名称 |
| `weight` | **名义敞口权重**（Σ=1） |
| `risk_contribution` | 风险贡献 RC_i（绝对值） |
| `risk_contribution_pct` | 风险贡献占比（应≈1/N） |
| `volatility` | 资产年化波动率 √(Σ_ii)·√252 |
| `margin_rate` | 期货保证金率（仅期货，来自 detail） |
| `contract_multiplier` | 合约乘数（仅期货） |
| `capital_allocation` | 资金占用权重（期货=w×margin_rate，其它=w） |
| `data_version` | 数据版本（`real-v1`） |
| `update_time` | 数据最新日期 + A股收盘时间 15:30（ISO 8601） |

组合级指标存于 `DataFrame.attrs`：`portfolio_vol_annual` / `lw_shrinkage`（δ）/ `mean_correlation`（ρ̄）/
`rc_max_over_min`（ERC 相等性指标，应≈1.0）/ `capital_utilization` / `n_assets` 等。

## 验证口径

`validate.py` 的 7 项检查：

1. **权重契约**：非负、和为 1、字段完整
2. **ERC 相等**：max(RC)/min(RC) ≤ 1.05（风险贡献近似相等）
3. **贡献守恒**：Σ RC_i = σ_p（风险贡献之和 = 组合总风险）
4. **LW 收缩**：δ ∈ [0,1]，协方差矩阵 PSD
5. **杠杆**：期货 margin_rate ∈ (0,1]，资金占用 ≤ 名义权重
6. **无未来函数**：月度 rebalance 只用过去数据估协方差
7. **样本外切片**：train/test 切分可复现

## 项目状态（2026-07-20 真实数据全链路通过）

- **真实数据 fixture**：9 资产 × 231 交易日（2025-08 ~ 2026-07）
- **validate.py**：7 项全过 —— ERC max(RC)/min(RC)=1.0000、ΣRC=σ_p、δ=0.176、ρ̄=0.267、9 次调仓
- **report.py**：RP 年化波动 6.39%、夏普 0.21、RC 离散度 1.000
- **analysis_report.py**：lookback 扫描 30/60/90/120 全可行；LW 条件数 462 vs 纯样本 502

### 已知局限

- 权重为名义敞口权重，假设名义敞口=本金（无整体杠杆）；真实风险平价基金常通过目标波动率加杠杆，本 skill 不做
- 指数/ETF 标的局部重叠（沪深300/国债/黄金各两处），形成高相关块，依赖 LW 收缩稳定
- `get_fund_daily` 日期跨度 ≤1 年，回测窗口约 350 交易日

## License

[GPL-3.0](LICENSE) © 2026 PandaTest

---
name: portfolio-risk-parity
description: 当需要开发、计算、验证风险平价（等风险贡献 ERC）组合时使用。手写 Ledoit-Wolf 收缩协方差稳定相关性，scipy SLSQP 求 ERC 权重，支持指数/期货/ETF 三类资产与月度 rebalance。
tags: [quant, portfolio, optimization, risk, risk-parity, erc]
---

# skill-portfolio-risk-parity — 风险平价组合优化

## 适用场景

需要构建一个**每类资产对组合风险的贡献相等**（而非权重相等）的组合时使用。
典型场景：跨资产配置（权益/债券/商品/贵金属）、需要稳定低波动的均衡组合、
对相关性结构敏感的多资产组合。区别于等权（1/N，权重相等但风险高度集中）与
均值-方差（对期望收益/协方差极端敏感），风险平价只依赖协方差，鲁棒性更好。

## 核心原理

**风险贡献（Risk Contribution）**：组合波动率可分解为各资产的风险贡献之和。

```
组合波动率：       σ_p(w) = √( w' Σ w )                         （Σ 为协方差矩阵）
边际风险贡献：     MRC_i = ∂σ_p/∂w_i = (Σw)_i / σ_p
风险贡献：         RC_i  = w_i · MRC_i = w_i · (Σw)_i / σ_p
守恒律：           Σ_i RC_i = σ_p        （所有资产风险贡献之和 = 组合总风险）
```

**等风险贡献（ERC）**：求权重 w 使每个资产的风险贡献相等。

```
RC_i = RC_j = σ_p / N      ∀ i, j       （N 个资产，每个分摊 1/N 的风险）
约束：Σ_i w_i = 1，w_i ≥ 0               （long-only，名义敞口归一）
```

### 求解（非线性优化）

ERC 无解析解（N>3 时），用 `scipy.optimize.minimize(method="SLSQP")`：

```
目标：    min_w  Σ_i ( RC_i - σ_p/N )²
约束：    Σ_i w_i = 1（等式），w_i ∈ [1e-8, 1]（非负 bounds）
初值：    w_i ∝ 1/σ_i    （逆波动率，风险平价的一阶近似，收敛快）
```

## 关键坑 1：相关性矩阵不稳 → Ledoit-Wolf 收缩

样本协方差在样本量不足或资产相关时高度不稳（尤其在 N 接近 T 时奇异），
直接用它求 ERC 会出现极端权重。本模型**手写 Ledoit-Wolf 常数相关收缩**（Ledoit-Wolf 2003）：

```
Σ_shrink = δ · F + (1-δ) · S

样本协方差：    S = (1/T) · Xc' Xc           （Xc 为去均值收益矩阵，1/T 归一）
资产波动：      σ_i = √S_ii
平均相关：      ρ̄ = mean(off-diag of Corr)   （Corr = D^(-1/2) S D^(-1/2)，D=diag(σ_i)）
收缩目标：      F_ij = ρ̄ · σ_i · σ_j   (i≠j)，F_ii = S_ii      （常数相关结构）
收缩强度：      δ = clip( π̂ / (γ̂ · T), 0, 1 )
                π̂ = Σ_ij (1/T) Σ_t [ Xc[t,i]·Xc[t,j] − S[i,j] ]²   （协方差元素渐近方差之和）
                γ̂ = ‖F − S‖²_F                                      （目标与样本的偏离）
```

`δ` 含义：样本噪声越大（π̂ 大）/ 样本越少（T 小）/ 资产越多（N 大）→ δ→1，更依赖稳定目标；
样本充足时 δ→0，用样本协方差。对角加 ε 正则确保 PSD。

> 注：`sklearn.covariance.LedoitWolf` 用单位阵目标，本 skill 用常数相关目标
> （对金融资产更合理，因其相关性结构比"各资产独立"更接近现实）。手写避免引入 sklearn 依赖。

## 关键坑 2：杠杆资产（期货）名义敞口归一化

期货是保证金交易：占用资金 = 名义敞口 × margin_rate。风险平价的权重必须是
**名义敞口权重**（否则不同杠杆资产的"风险"不可比）。

```
名义收益率：   r_i = Δ(close_i) / close_i = close_i.pct_change()
              （pct_change 天然是名义收益率口径，与合约乘数无关）
名义权重：     w_i   ← ERC 求解结果（Σw=1，假设名义敞口=本金，无整体杠杆）
资金占用：     capital_i = w_i × margin_rate_i   （仅期货；指数/ETF 的 capital_i = w_i）
资金利用率：   Σ_i capital_i < 100% → 剩余为杠杆空间（可加仓或持现金）
合约乘数：     contract_multiplier（来自 get_future_detail，仅转"手数"时需要，风险平价输出权重不强制）
```

`margin_rate` / `contract_multiplier` 通过 `get_future_detail(symbol=...)` 获取。

## 关键坑 3：月度 rebalance（无未来函数）

```
1. 交易日按月分组，每月最后一个交易日为调仓日
2. 调仓日 t：用【截至 t 日（含）】的过去 LOOKBACK_DAYS（默认 60）个交易日估 LW 收缩协方差 → 求 ERC 权重
3. t+1 日 ~ 下次调仓前，持有该固定权重
```

权重只用"过去"信息，不窥探未来收益，无未来函数。

## 输入数据

使用 Panda data SDK 拉取三类资产日线：

| 资产类 | 接口 | 说明 |
|---|---|---|
| 指数 | `get_index_daily` | 权益/债券大盘（沪深300/中证500/国债指数） |
| 期货 | `get_future_daily` + `get_future_detail` | 商品/金融期货主力合约（带 margin/multiplier） |
| 基金(ETF) | `get_fund_daily` | 场内 ETF（510300/511260/518880），返回 OHLCV；⚠️日期跨度 ≤1年 |

### 默认资产池（3+3+3 九资产）

```
指数：  000300.SH(沪深300)  000905.SH(中证500)  000016.SH(上证50)
期货：  CU_DOMINANT.SHF(铜)  RB_DOMINANT.SHF(螺纹钢)  AU_DOMINANT.SHF(黄金)
ETF：   510300.SH(沪深300ETF)  511260.SH(国债ETF)  518880.SH(黄金ETF)
```

> ⚠️ 指数与 ETF 标的局部重叠（沪深300/国债/黄金各两处），会出现高相关块。
> 这正是 LW 收缩要稳定的对象。资产池可经参数覆盖；如需完全去重叠，可将 ETF
> 换成创业板/纳指/红利等不同标的。

### 输入契约

- 期货主力合约格式：`XXX_DOMINANT.EXCHANGE`（如 `CU_DOMINANT.SHF`、`RB_DOMINANT.SHF`）
- 三类资产交易日历内连接对齐（`dropna(how="any")`），期货夜盘/ETF 不同步会丢少量交易日
- `_normalize_price` 防御字段差异：`code`/`ts_code`→`symbol`、`trade_date`→`date`
- 默认回溯 1 年；月度 rebalance 协方差窗口默认 60 交易日

## 输出结果

`optimize_portfolio()` 返回权重表 DataFrame，**每个资产一行**：

| 字段 | 说明 |
|---|---|
| trade_date | 权重生成日期（数据最新日，YYYY-MM-DD） |
| asset_type | 资产类型（index/future/etf） |
| symbol | 资产代码 |
| name | 名称（期货来自 detail） |
| model_id | 模型编号（RP1） |
| model_name | 模型名称 |
| weight | **名义敞口权重**（Σ=1） |
| risk_contribution | 风险贡献 RC_i（绝对值） |
| risk_contribution_pct | 风险贡献占比（应≈1/N） |
| volatility | 资产年化波动率 √(Σ_ii)·√252 |
| margin_rate | 期货保证金率（仅期货，来自 detail） |
| contract_multiplier | 合约乘数（仅期货） |
| capital_allocation | 资金占用权重（期货=w×margin_rate，其它=w） |
| data_version | 数据版本（real-v1） |
| update_time | 生成时间（ISO 8601，按数据最新日推导，可复现） |

**组合级指标**存于 `DataFrame.attrs`：

| attrs 键 | 说明 |
|---|---|
| portfolio_vol_annual / portfolio_vol_daily | 组合年化/日频波动率 |
| lw_shrinkage | LW 收缩强度 δ ∈ [0,1] |
| mean_correlation | 平均相关系数 ρ̄ |
| rc_max_over_min | max(RC)/min(RC)（ERC 相等性指标，应≈1.0） |
| erc_solved | ERC 求解是否成功 |
| capital_utilization | 总资金利用率（<100% 表示有杠杆空间） |
| n_assets / target / rebalance_freq / lookback_days | 元信息 |

## 模型评价标准

| 分类 | 指标 | 方向 | 说明 |
|---|---|---|---|
| 核心验证 | 风险贡献离散度 | →1.0 | max(RC)/min(RC)，越接近 1 越等风险 |
| 核心验证 | 风险贡献占比 | →1/N | 每项 RC_i/σ_p 应≈1/N |
| 风险 | RP 年化波动 | 越小越好 | 应显著低于等权组合 |
| 风险 | 波动收敛(RP-等权) | 越负越好 | 负值=风险平价成功降波 |
| 收益 | 年化收益 / 最大回撤 | — | 副指标（风险平价不主动追求收益） |
| 综合 | 夏普 | 越大越好 | 风险调整后收益 |
| 相对 | 超额年化(RP-Bench) | 越大越好 | 相对沪深300 |

## 代码结构（对齐 Alpha 因子开发规则 V2 §4 标准分层）

- `scripts/factor.py` —— **因子计算主脚本，可独立运行**（`python factor.py`）
  收益矩阵 → Ledoit-Wolf 收缩协方差 → scipy SLSQP 求 ERC 权重 → 权重表（含 main 入口）
- `scripts/data_loader.py` —— 数据加载层（取数与清洗，不含因子计算）
  panda_data SDK 拉三类资产日线 + 期货元信息 + parquet 缓存 + 交易日历定日期
- `scripts/backtest.py` —— 月度 rebalance 回测 + 评价指标
- `scripts/validate.py` —— 7 项检查（契约 / ERC 相等 / 贡献守恒 / LW / 杠杆 / 无未来 / 切片）
- `scripts/report.py` / `analysis_report.py` —— HTML 报告 + lookback 扫描
- `references/data_guide.md` —— 数据源接口与字段说明（详见该文档）

> 与横截面选股因子不同：风险平价是**组合优化因子**，输出每资产配置权重
> （非 IC/IR/分层/score/signal）。故 factor.py 的"因子值"= ERC 权重。

## 使用方式

```bash
# 离线验证（合成 fixture，无需凭证）
cd scripts && python _make_synthetic_fixture.py   # 生成 fixture
PANDA_DATA_OFFLINE=1 python validate.py           # 7 项检查
PANDA_DATA_OFFLINE=1 python report.py --offline   # HTML 报告
python analysis_report.py --offline               # lookback 扫描

# 联网真实数据
export PANDA_DATA_USERNAME=...
export PANDA_DATA_PASSWORD=...
python save_fixture.py        # 拉真实样本存 fixture
python factor.py              # 单期 ERC 权重（独立运行，输出权重表）
python validate.py            # 真实数据验证
python report.py              # 真实数据 HTML 报告
```

## 离线验证模式

`PANDA_DATA_OFFLINE=1` 时：
- 数据从 `fixtures/sample_assets.parquet` + `sample_future_meta.json` 读取
- 注入 panda_data 存根 + parquet fallback（pyarrow→fastparquet），无需凭证
- `sample_assets.parquet` 由 `save_fixture.py` 联网拉取的真实数据生成（9资产×231交易日）
- `_make_synthetic_fixture.py` 可生成合成 fixture（RandomState=42）作离线基准备用

## Agent 执行规则

1. 默认走离线模式（`PANDA_DATA_OFFLINE=1`），用 fixture 验证全链路
2. 联网前必须配置 `PANDA_DATA_USERNAME` / `PANDA_DATA_PASSWORD`
3. 每次产出权重表后必须跑 `validate.py` 的 7 项检查全部 PASS
4. 权重必须满足：非负、和为 1、风险贡献近似相等（max(RC)/min(RC) ≈ 1）
5. 期货 margin_rate 必须 ∈ (0,1]，资金占用 ≤ 名义权重
6. LW 收缩强度 δ 必须 ∈ [0,1]
7. 月度 rebalance 必须无未来函数（只用过去数据估协方差）

## 成功标准

- ✅ `validate.py` 7 项检查全 PASS（权重契约/ERC相等/贡献守恒/LW收缩/杠杆/无未来函数/月度可回测）
- ✅ 风险贡献离散度 rc_max_over_min ≤ 1.05（近似等风险）
- ✅ RP 年化波动 < 等权年化波动（风险平价确实降波）
- ✅ HTML 报告正常生成（4 图：净值/风险贡献/权重/相关性热图）

## 验收要求

1. 离线端到端：validate / report / analysis / _edge_test 全链路通过
2. 代码带中文注释，公式与本文档一致
3. `notes.md` 记录关键决策（资产池选择、LW 手写、ETF 代替基金等）

## 依赖

- pandas / numpy（数据处理）
- scipy（SLSQP 求 ERC）
- panda_data（数据源 SDK）
- matplotlib（报告可视化）
- fastparquet（parquet 读 fallback）

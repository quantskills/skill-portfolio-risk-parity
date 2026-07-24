# skill-portfolio-risk-parity

[简体中文](README.md) | **English**

> Risk Parity (ERC) portfolio optimization: hand-written Ledoit-Wolf shrinkage covariance stabilizes correlations, scipy SLSQP solves equal-risk-contribution weights, supporting index/future/ETF assets with monthly rebalance.

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

`skill-portfolio-risk-parity` is a risk-parity portfolio optimization Skill based on the Equal Risk Contribution (ERC) method, solving weights where each asset contributes equally to total portfolio risk, with contract validation, monthly-rebalance backtesting, and offline reproduction.

This Skill is suitable for:

- Balanced portfolio construction across asset classes (equity / bonds / commodities / precious metals)
- Multi-asset portfolios requiring stable low volatility and sensitivity to correlation structure
- Verifying whether risk parity truly spreads risk evenly across asset classes
- Triggering deterministic portfolio optimization and backtesting from Claude Code conversations

This Skill pulls index / futures / ETF daily data via the `panda_data` SDK, outputting weight tables, backtest metrics (annualized volatility / Sharpe / MDD), a self-contained HTML visualization report, and offline fixtures for CI integration.

## Core Methods

- **Equal Risk Contribution (ERC)**: solve weights so each asset's risk contribution RC_i is equal (= σ_p/N), via `scipy.optimize.minimize(method="SLSQP")`
- **Ledoit-Wolf shrinkage covariance (hand-written)**: constant-correlation target stabilizes the covariance matrix under insufficient samples / high correlation, avoiding extreme weights
- **Leveraged asset normalization**: futures unified by notional-exposure weight (margin used only for capital-utilization reporting), making risk comparable across the three asset classes
- **Monthly rebalance**: rebalance at month-end, covariance window defaults to 60 trading days, using only past information (no look-ahead)

## Repository Contents

| File | Description |
|---|---|
| `SKILL.md` | Skill contract document (internal Agent use, full formulas) |
| `scripts/factor.py` | ERC weight solver entry (return matrix → LW shrinkage → SLSQP → weight table, standalone) |
| `scripts/data_loader.py` | Data layer (index/futures/ETF daily + futures metadata + parquet cache + trade calendar) |
| `scripts/validate.py` | Weight validation (7 checks: contract / ERC equality / contribution conservation / LW / leverage / no-look-ahead / split) |
| `scripts/backtest.py` | Monthly-rebalance backtest (annualized vol / Sharpe / MDD + equal-weight comparison) |
| `scripts/backtest_report_data.py` | Backtest timeseries wrapper (HTML report data source) |
| `scripts/report.py` | HTML report generator entry (4 charts: NAV / risk contribution / weights / correlation heatmap) |
| `scripts/analysis_report.py` | lookback sweep + LW shrinkage effect comparison |
| `scripts/save_fixture.py` | One-time offline fixture generator (pulls real samples online) |
| `scripts/_make_synthetic_fixture.py` | Generate synthetic fixture (RandomState=42, offline baseline) |
| `scripts/_edge_test.py` | Edge-case regression tests |
| `scripts/fixtures/` | Offline test data (Parquet format) |
| `requirements.txt` | Python dependencies |
| `references/data_guide.md` | PandaAI data API reference |
| `LICENSE` | GPLv3 license |
| `README.md` / `README.en.md` | Chinese / English README |

## Directory Structure

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
│   ├── factor.py                  # ERC solver (LW shrinkage + SLSQP + optimize_portfolio + main)
│   ├── data_loader.py             # Data layer (load_index/fund/future + detail + cache + trade cal)
│   ├── validate.py                # 7 checks (contract/ERC/conservation/LW/leverage/no-lookahead/split)
│   ├── backtest.py                # Monthly rebalance (annual vol/Sharpe/MDD + equal-weight)
│   ├── backtest_report_data.py    # Timeseries wrapper (curve / drawdown / rc series)
│   ├── report.py                  # HTML report (4 charts + metric cards + weight detail)
│   ├── analysis_report.py         # lookback sweep + LW shrinkage comparison
│   ├── save_fixture.py            # Online fixture generator
│   ├── _make_synthetic_fixture.py # Synthetic fixture (test baseline)
│   ├── _edge_test.py              # Edge-case regression
│   └── fixtures/
│       ├── sample_assets.parquet  # Offline asset data
│       └── sample_future_meta.json # Futures metadata (margin/multiplier)
└── reports/                       # Report artifacts (.gitignored)
```

## Data Requirements

Input contract for calling `panda_data`:

| Asset class | API | Description |
|---|---|---|
| Index | `get_index_daily` | Equity/bond broad indices (CSI 300 / CSI 500 / SSE 50) |
| Futures | `get_future_daily` + `get_future_detail` | Commodity/financial futures main contracts (with margin_rate / contract_multiplier) |
| ETF | `get_fund_daily` | On-exchange ETFs (510300/511260/518880), returns OHLCV; ⚠️ date span ≤1 year |

### Default asset pool (3+3+3, nine assets)

```
Index:   000300.SH (CSI 300)  000905.SH (CSI 500)  000016.SH (SSE 50)
Futures: CU_DOMINANT.SHF (Copper)  RB_DOMINANT.SHF (Rebar)  AU_DOMINANT.SHF (Gold)
ETF:     510300.SH (CSI300 ETF)  511260.SH (Treasury ETF)  518880.SH (Gold ETF)
```

Input contract details:

- **Futures main-contract format**: `XXX_DOMINANT.EXCHANGE` (e.g., `CU_DOMINANT.SHF`)
- **Three asset classes inner-join aligned by trade calendar** (`dropna(how="any")`); futures night session / ETF asynchrony drops a few trading days
- **`_normalize_price` defends field differences**: `code`/`ts_code`→`symbol`, `trade_date`→`date`
- **Default lookback 1 year** (limited by `get_fund_daily` ≤1 year); monthly-rebalance covariance window defaults to 60 trading days

## Quick Start

### Environment Setup

```bash
export PANDA_DATA_USERNAME=your_username
export PANDA_DATA_PASSWORD=your_password

# Optional: control data range
export PANDA_DATA_START_DATE=2025-08-01
export PANDA_DATA_END_DATE=2026-07-20
```

### Four-Step Run

```bash
cd scripts/

# 1. ERC weight solve (single period)
python factor.py
# Output: weight table (trade_date, symbol, weight, risk_contribution, risk_contribution_pct, ...)

# 2. Weight validation
python validate.py
# Output: 7 checks (contract / ERC equality / contribution conservation / LW shrinkage / leverage / no-look-ahead / split)

# 3. Monthly-rebalance backtest
python backtest.py
# Output: RP annualized volatility, Sharpe, MDD, rebalance count, and equal-weight comparison

# 4. HTML report generation
python report.py
# Output: reports/report.html (NAV / risk contribution / weights / correlation heatmap)
```

### Offline Mode (CI Friendly)

```bash
# 1. Generate fixtures once (requires network + credentials)
python save_fixture.py
# Generates scripts/fixtures/sample_assets.parquet + sample_future_meta.json

# Or use synthetic baseline (no network needed)
python _make_synthetic_fixture.py

# 2. Subsequent validation without network
export PANDA_DATA_OFFLINE=1
python validate.py
python report.py --offline
python analysis_report.py --offline    # lookback sweep
```

In offline mode, if `panda_data` SDK is not installed locally, a stub module is auto-injected to bypass the top-level import; when pyarrow is incompatible with legacy parquet, it falls back to the `fastparquet` engine.

## Input Configuration

| Environment Variable | Required | Default | Description |
|---|---|---|---|
| `PANDA_DATA_USERNAME` | ✓ | — | PandaAI account |
| `PANDA_DATA_PASSWORD` | ✓ | — | PandaAI password |
| `PANDA_DATA_START_DATE` | — | 350 days before end date | Start date (YYYY-MM-DD) |
| `PANDA_DATA_END_DATE` | — | Latest trade date (trade calendar) | End date (YYYY-MM-DD) |
| `PANDA_DATA_OFFLINE` | — | `0` | When `1`, enables offline mode loading from fixture |
| `PANDA_DATA_DEBUG` | — | `0` | When `1`, prints schema probe on first API call |

## Output Files

Weight table fields output by `factor.py` (one row per asset):

| Field | Description |
|---|---|
| `trade_date` | Weight generation date (latest data date, YYYY-MM-DD) |
| `asset_type` | Asset class (index/future/etf) |
| `symbol` | Asset code |
| `name` | Name (futures from detail) |
| `model_id` | Model ID (fixed as `RP1`) |
| `model_name` | Model name |
| `weight` | **Notional-exposure weight** (Σ=1) |
| `risk_contribution` | Risk contribution RC_i (absolute) |
| `risk_contribution_pct` | Risk contribution share (should ≈1/N) |
| `volatility` | Asset annualized volatility √(Σ_ii)·√252 |
| `margin_rate` | Futures margin rate (futures only, from detail) |
| `contract_multiplier` | Contract multiplier (futures only) |
| `capital_allocation` | Capital utilization weight (futures=w×margin_rate, others=w) |
| `data_version` | Data version (`real-v1`) |
| `update_time` | Latest data date + A-share close time 15:30 (ISO 8601) |

Portfolio-level metrics are stored in `DataFrame.attrs`: `portfolio_vol_annual` / `lw_shrinkage` (δ) / `mean_correlation` (ρ̄) / `rc_max_over_min` (ERC equality metric, should ≈1.0) / `capital_utilization` / `n_assets`, etc.

## Validation Approach

7 checks in `validate.py`:

1. **Weight contract**: non-negative, sum to 1, complete fields
2. **ERC equality**: max(RC)/min(RC) ≤ 1.05 (risk contributions approximately equal)
3. **Contribution conservation**: Σ RC_i = σ_p (sum of risk contributions = total portfolio risk)
4. **LW shrinkage**: δ ∈ [0,1], covariance matrix PSD
5. **Leverage**: futures margin_rate ∈ (0,1], capital utilization ≤ notional weight
6. **No look-ahead**: monthly rebalance uses only past data to estimate covariance
7. **Out-of-sample split**: train/test split reproducible

## Project Status (2026-07-20, real-data full pipeline passed)

- **Real-data fixture**: 9 assets × 231 trading days (2025-08 ~ 2026-07)
- **validate.py**: all 7 pass — ERC max(RC)/min(RC)=1.0000, ΣRC=σ_p, δ=0.176, ρ̄=0.267, 9 rebalances
- **report.py**: RP annualized volatility 6.39%, Sharpe 0.21, RC dispersion 1.000
- **analysis_report.py**: lookback sweep 30/60/90/120 all feasible; LW condition number 462 vs raw sample 502

### Known Limitations

- Weights are notional-exposure weights, assuming notional = principal (no overall leverage); real risk-parity funds often lever up via target volatility, which this skill does not do
- Index/ETF underlying partially overlap (CSI 300 / treasury / gold each appear twice), forming high-correlation blocks that depend on LW shrinkage for stability
- `get_fund_daily` date span ≤1 year, so the backtest window is about 350 trading days

## License

[GPL-3.0](LICENSE) © 2026 PandaTest

#!/usr/bin/env python3
"""风险平价深度分析报告 —— lookback 扫描 + LW 收缩效果对比

功能：
  1. 扫描多个协方差估计窗口 lookback（30/60/90/120），对比样本外波动/夏普/调仓次数
  2. 对比「LW 收缩协方差」vs「纯样本协方差」下 ERC 权重的风险贡献离散度
     （证明 LW 收缩让权重更稳、风险贡献更均衡）
  3. HTML 列出每个 lookback 的指标 + 权重表
  4. 独立脚本，仅导入 optimize/backtest，不修改它们

用法:
    python analysis_report.py --offline
    python analysis_report.py --lookbacks 20,40,60,90,120
"""
from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from factor import ledoit_wolf_shrinkage, solve_erc  # noqa: E402
from backtest import run_backtest  # noqa: E402
from validate import _load_fixture_or_network  # noqa: E402

REPORTS_DIR = Path(__file__).parent.parent / "reports"
CHART_DPI = 120
_C_PRIMARY = "#3498db"
_C_POS = "#27ae60"
_C_NEG = "#e74c3c"
_C_BG = "#f7f7f8"


def run_lookback_scan(price_df: pd.DataFrame, future_meta: dict,
                      lookbacks: list[int]) -> list[dict]:
    """扫描多个 lookback，每个跑月度 rebalance 回测"""
    results = []
    for lb in lookbacks:
        try:
            bt = run_backtest(price_df, future_meta, lookback=lb)
            m = bt["metrics"]
            results.append({
                "lookback": lb, "feasible": True,
                "rp_vol": m["RP_年化波动"], "eq_vol": m["等权_年化波动"],
                "rp_sharpe": m["RP_夏普"], "rp_return": m["RP_年化收益"],
                "rp_mdd": m["RP_最大回撤"], "n_rebalances": m["调仓次数"],
                "rc_dispersion": m["样本外风险贡献离散度"],
                "vol_converge": m["波动收敛(RP-等权)"],
            })
        except Exception as e:  # noqa: BLE001
            results.append({"lookback": lb, "feasible": False, "error": str(e)[:80]})
    return results


def compare_shrinkage(price_df: pd.DataFrame) -> dict:
    """LW 收缩 vs 纯样本协方差 —— 对比「估计稳定性」

    注意：不能比 ERC 求解后的风险贡献离散度（solve_erc 必然把 RC 拉到相等，两边都≈1.0，
    对比无意义）。LW 的价值在「协方差估计对采样不敏感」，故用两个稳定性指标：
      1. 条件数 cond(Σ)：越小矩阵越良态、数值越稳，LW 应降低它
      2. 权重漂移 L1：前半样本估协方差→ERC 权重，与全样本权重的 L1 距离，
         越小表示「换一段数据权重变化越小」，LW 应让它更小
    """
    from factor import build_return_matrix
    ret = build_return_matrix(price_df)
    X = ret.values
    T = X.shape[0]

    # 全样本：两种协方差 + 条件数（良态度量）
    Sigma_lw, delta, rho = ledoit_wolf_shrinkage(ret)
    Sigma_raw = np.cov(X, rowvar=False)  # 1/(T-1)
    cond_lw = float(np.linalg.cond(Sigma_lw))
    cond_raw = float(np.linalg.cond(Sigma_raw))
    w_lw, _ = solve_erc(Sigma_lw)
    w_raw, _ = solve_erc(Sigma_raw)

    # 权重稳定性：前半样本估协方差 → ERC 权重，与全样本权重比 L1 漂移
    half = T // 2
    ret_first = ret.iloc[:half]
    Sigma_lw_1, _, _ = ledoit_wolf_shrinkage(ret_first)
    Sigma_raw_1 = np.cov(ret_first.values, rowvar=False)
    w_lw_half, _ = solve_erc(Sigma_lw_1)
    w_raw_half, _ = solve_erc(Sigma_raw_1)
    l1_lw = float(np.abs(w_lw_half - w_lw).sum())    # LW 下前半 vs 全样本权重漂移
    l1_raw = float(np.abs(w_raw_half - w_raw).sum())  # 纯样本下前半 vs 全样本权重漂移

    return {"delta": delta, "rho": rho,
            "cond_lw": cond_lw, "cond_raw": cond_raw,
            "l1_lw": l1_lw, "l1_raw": l1_raw,
            "w_lw": w_lw.tolist(), "w_raw": w_raw.tolist(),
            "symbols": list(ret.columns)}


def plot_lookback(results: list[dict]) -> str:
    feas = [r for r in results if r["feasible"]]
    if len(feas) < 2:
        return ""
    lbs = [r["lookback"] for r in feas]
    vols = [r["rp_vol"] * 100 for r in feas]
    shp = [r["rp_sharpe"] for r in feas]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_C_BG)
    ax1.set_facecolor("white")
    ax1.plot(lbs, vols, "o-", color=_C_PRIMARY, linewidth=1.8, markersize=7, label="样本外年化波动(%)")
    ax1.set_xlabel("协方差估计窗口 lookback（交易日）", fontsize=11)
    ax1.set_ylabel("年化波动 (%)", color=_C_PRIMARY, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=_C_PRIMARY)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(lbs, shp, "s--", color=_C_POS, linewidth=1.4, markersize=6, alpha=0.85, label="夏普")
    ax2.set_ylabel("夏普比率", color=_C_POS, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=_C_POS)

    fig.suptitle("lookback 扫描：样本外波动与夏普", fontsize=13, fontweight="bold")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.95), fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight", facecolor=_C_BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def render_html(results: list[dict], shrink: dict, meta: dict) -> str:
    feas = [r for r in results if r["feasible"]]
    infeas = [r for r in results if not r["feasible"]]

    rows = []
    for r in feas:
        rows.append(
            f"<tr><td><strong>{r['lookback']}</strong></td>"
            f'<td class="num">{r["rp_vol"]*100:.2f}%</td>'
            f'<td class="num">{r["eq_vol"]*100:.2f}%</td>'
            f'<td class="num {"pos" if r["vol_converge"]<0 else "neg"}">{r["vol_converge"]*100:+.2f}%</td>'
            f'<td class="num">{r["rp_sharpe"]:.2f}</td>'
            f'<td class="num {"pos" if r["rp_return"]>0 else "neg"}">{r["rp_return"]*100:+.2f}%</td>'
            f'<td class="num neg">{r["rp_mdd"]*100:.2f}%</td>'
            f'<td class="num">{r["rc_dispersion"]:.3f}</td>'
            f'<td class="num">{r["n_rebalances"]}</td></tr>')
    infeas_note = ""
    if infeas:
        infeas_note = '<p class="note">以下 lookback 不可行：' + ", ".join(
            str(r["lookback"]) for r in infeas) + "。</p>"

    scan_img = plot_lookback(results)
    scan_section = f'<img src="data:image/png;base64,{scan_img}"/>' if scan_img else ""

    # LW vs 纯样本 权重对比条形
    syms = [s[:6] for s in shrink["symbols"]]
    x = np.arange(len(syms))
    w_img = ""
    try:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor(_C_BG)
        ax.bar(x - 0.2, np.array(shrink["w_lw"]) * 100, 0.4, color=_C_PRIMARY, label="LW 收缩 ERC")
        ax.bar(x + 0.2, np.array(shrink["w_raw"]) * 100, 0.4, color=_C_NEG, alpha=0.7, label="纯样本 ERC")
        ax.set_xticks(x)
        ax.set_xticklabels(syms, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("名义权重 (%)")
        ax.set_title("LW 收缩 vs 纯样本协方差：ERC 权重对比", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=CHART_DPI, bbox_inches="tight", facecolor=_C_BG)
        plt.close(fig)
        buf.seek(0)
        w_img = base64.b64encode(buf.read()).decode("ascii")
    except Exception:  # noqa: BLE001
        pass

    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>风险平价 — 参数扫描分析报告</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 0 auto; max-width: 1300px;
         background: {_C_BG}; color: #1a1a1a; padding: 32px 24px; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid {_C_PRIMARY}; padding-bottom: 12px; }}
  h2 {{ color: #34495e; margin-top: 36px; padding-bottom: 6px; border-bottom: 1px solid #ecf0f1; }}
  .summary {{ background: #fff3cd; padding: 14px 18px; border-left: 4px solid #f39c12; margin: 16px 0 24px; border-radius: 4px; font-size: 14px; line-height: 1.7; }}
  .note {{ color: #7f8c8d; font-size: 13px; margin: 8px 0 16px; line-height: 1.5; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin: 12px 0; font-size: 13px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #34495e; color: white; font-weight: 600; }}
  tr:hover {{ background: #f8f9fa; }}
  .num {{ text-align: right; font-family: Consolas, monospace; }}
  .pos {{ color: {_C_POS}; font-weight: 600; }} .neg {{ color: {_C_NEG}; font-weight: 600; }}
  img {{ max-width: 100%; border-radius: 6px; margin: 12px 0; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #ecf0f1; color: #95a5a6; font-size: 12px; text-align: center; }}
</style></head><body>
  <h1>风险平价(ERC) — 参数扫描分析报告</h1>
  <div class="summary">
    <strong>数据</strong>：{meta['asset_count']} 资产 × {meta['trade_days']} 交易日
    <strong>LW 收缩</strong>：δ={shrink['delta']:.3f}，平均相关 ρ̄={shrink['rho']:.3f}<br>
    <strong>对比</strong>：月末调仓，过去 lookback 日估协方差求 ERC 权重，下月持有。
  </div>

  <h2>lookback 扫描</h2>
  <p class="note">协方差估计窗口越长 → 估计越稳但对市场变化反应越慢；越短 → 反应快但噪声大。
     "波动收敛"为负表示风险平价比等权波动更低（风险平价的目的）。</p>
  {infeas_note}
  <table><thead><tr>
    <th>lookback</th><th>RP波动</th><th>等权波动</th><th>波动收敛</th>
    <th>夏普</th><th>年化收益</th><th>最大回撤</th><th>RC离散度</th><th>调仓次数</th>
  </tr></thead><tbody>{chr(10).join(rows)}</tbody></table>
  {scan_section}

  <h2>LW 收缩 vs 纯样本协方差</h2>
  <p class="note">LW 收缩的核心价值是让协方差估计对采样不敏感（稳定），而非改变 ERC 求解结果
     （solve_erc 必然把风险贡献拉到相等，故不比求解后的离散度）。
     <strong>协方差条件数</strong>（越小矩阵越良态、数值越稳）：LW={shrink['cond_lw']:.1f}，纯样本={shrink['cond_raw']:.1f}<br>
     <strong>权重漂移 L1</strong>（前半样本估权重 vs 全样本，越小越稳）：LW={shrink['l1_lw']:.4f}，纯样本={shrink['l1_raw']:.4f}</p>
  {f'<img src="data:image/png;base64,{w_img}"/>' if w_img else ''}

  <footer>由 analysis_report.py 独立生成 · 未修改 optimize/backtest · 数据来源 panda_data SDK</footer>
</body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description="风险平价参数扫描分析报告")
    p.add_argument("--lookbacks", "-l", default="30,60,90,120",
                   help="协方差窗口列表，逗号分隔（如 30,60,90）")
    p.add_argument("--offline", action="store_true", help="离线模式：从 fixtures 读数据")
    p.add_argument("--output", "-o", default=None, help="HTML 输出路径")
    args = p.parse_args()

    if args.offline:
        os.environ["PANDA_DATA_OFFLINE"] = "1"
    lookbacks = [int(x.strip()) for x in args.lookbacks.split(",")]
    print(f"lookback 扫描: {lookbacks}")

    price, future_meta = _load_fixture_or_network()
    from factor import build_return_matrix
    ret = build_return_matrix(price)

    results = run_lookback_scan(price, future_meta, lookbacks)
    shrink = compare_shrinkage(price)
    meta = {"asset_count": ret.shape[1], "trade_days": ret.shape[0]}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else REPORTS_DIR / "analysis_report.html"
    out.write_text(render_html(results, shrink, meta), encoding="utf-8")
    print(f"\n[OK] 分析报告: {out} ({out.stat().st_size / 1024:.1f} KB)")

    print("\nlookback 扫描摘要:")
    for r in results:
        if r["feasible"]:
            print(f"  lb={r['lookback']:>3}  RP波动={r['rp_vol']*100:.2f}%  "
                  f"夏普={r['rp_sharpe']:.2f}  RC离散={r['rc_dispersion']:.3f}  "
                  f"调仓{r['n_rebalances']}次")
        else:
            print(f"  lb={r['lookback']:>3}  不可行: {r['error']}")
    print(f"\nLW收缩稳定性对比: 条件数 LW={shrink['cond_lw']:.1f} vs 纯样本={shrink['cond_raw']:.1f}"
          f" | 权重漂移L1 LW={shrink['l1_lw']:.4f} vs 纯样本={shrink['l1_raw']:.4f}")


if __name__ == "__main__":
    main()

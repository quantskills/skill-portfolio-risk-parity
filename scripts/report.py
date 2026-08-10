"""生成风险平价(ERC)组合月度 rebalance 回测的 HTML 报告

用法:
    # 离线模式（用 fixtures 数据，无需凭证）
    python scripts/report.py --offline

    # 联网模式（需 PANDA_DATA_USERNAME / PANDA_DATA_PASSWORD）
    python scripts/report.py

    python scripts/report.py --offline --html-path reports/custom.html --open

输出:
    reports/backtest_result.json — 中间产物，含全部时序数据
    reports/report.html          — 最终 HTML 报告（self-contained，base64 内嵌图表）
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 中文字体（标题/图例含中文，必须设置否则方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from backtest_report_data import (  # noqa: E402
    REPORTS_DIR,
    run_backtest_with_series,
    save_backtest_result,
)

# === 配色 =======================================================================
_C_RP = "#3b82f6"       # 蓝：风险平价组合
_C_EQ = "#f59e0b"       # 橙：等权组合
_C_BENCH = "#94a3b8"    # 灰：benchmark
_C_DD = "#ef4444"       # 红：回撤
_COLOR_BY_TYPE = {"index": "#3b82f6", "future": "#ef4444", "fund": "#22c55e"}

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
               "Microsoft YaHei", sans-serif; line-height: 1.6; color: #1f2937;
        background: #f9fafb; max-width: 1200px; margin: 0 auto; padding: 24px; }
header { padding: 16px 0 24px; border-bottom: 2px solid #e5e7eb; margin-bottom: 24px; }
h1 { font-size: 24px; color: #111827; margin-bottom: 8px; font-weight: 600; }
h2 { font-size: 18px; color: #374151; margin: 32px 0 16px; padding-left: 12px;
     border-left: 4px solid #3b82f6; }
.meta { font-size: 13px; color: #6b7280; }
.cards-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.card { background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-left: 3px solid #3b82f6; }
.card .label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; }
.card .value { font-size: 22px; font-weight: 600; color: #111827; margin-top: 4px; }
.card .unit { font-size: 13px; color: #6b7280; font-weight: normal; margin-left: 2px; }
.card.good { border-left-color: #22c55e; } .card.good .value { color: #16a34a; }
.card.bad { border-left-color: #ef4444; } .card.bad .value { color: #dc2626; }
img { max-width: 100%; height: auto; margin: 12px 0; border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08); display: block; }
.note { font-size: 13px; color: #6b7280; margin: 8px 0; padding: 12px; background: #f3f4f6;
        border-radius: 6px; line-height: 1.7; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
        overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 16px 0; }
th { background: #f3f4f6; padding: 12px; text-align: left; font-size: 13px; color: #374151; font-weight: 600; }
td { padding: 10px 12px; border-top: 1px solid #e5e7eb; font-size: 13px; color: #4b5563; }
tr:hover td { background: #f9fafb; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: #16a34a; font-weight: 600; } .neg { color: #dc2626; font-weight: 600; }
footer { margin-top: 40px; padding: 20px 24px; background: #fef3c7; border-radius: 8px;
         border-left: 4px solid #f59e0b; }
footer h3 { font-size: 14px; color: #92400e; margin-bottom: 8px; font-weight: 600; }
footer p { font-size: 13px; color: #78350f; line-height: 1.7; }
"""


def _fig_to_base64(fig, dpi: int) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _fmt_xdates(ax, dates: list[str]) -> None:
    if len(dates) > 8:
        step = max(1, len(dates) // 8)
        idx = list(range(0, len(dates), step))
        ax.set_xticks(idx)
        ax.set_xticklabels([dates[i] for i in idx], rotation=30, ha="right", fontsize=9)


def plot_equity_curve(data: dict, dpi: int) -> str:
    """图1: 三组合累计净值（上）+ RP 回撤（下）"""
    rp = data["rp_curve"]
    eq = data["eq_curve"]
    bench = data.get("bench_curve") or []
    dd = data["rp_drawdown"]
    dates = [p["date"] for p in rp]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(dates, [p["cum_return"] - 1 for p in rp], color=_C_RP, linewidth=1.6, label="风险平价")
    ax1.plot(dates, [p["cum_return"] - 1 for p in eq], color=_C_EQ, linewidth=1.2, label="等权(1/N)", alpha=0.85)
    if bench:
        bd = [p["date"] for p in bench]
        ax1.plot(bd, [p["cum_return"] - 1 for p in bench], color=_C_BENCH, linewidth=1.2,
                 label="沪深300", alpha=0.85, linestyle="--")
    ax1.axhline(0, color="#9ca3af", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    _fmt_xdates(ax1, dates)

    dd_dates = [p["date"] for p in dd]
    ax2.fill_between(dd_dates, [p["drawdown"] for p in dd], 0, color=_C_DD, alpha=0.4, label="RP 回撤")
    ax2.set_ylabel("Drawdown")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    _fmt_xdates(ax2, dd_dates)
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_risk_contribution(data: dict, dpi: int) -> str:
    """图2: 风险贡献占比条形 vs 1/N 等风险线（ERC 核心）"""
    weights = sorted(data["weights"], key=lambda w: w["risk_contribution_pct"])
    syms = [f"{w['symbol'][:6]}" for w in weights]
    rcp = [w["risk_contribution_pct"] * 100 for w in weights]
    colors = [_COLOR_BY_TYPE.get(w["asset_type"], "#999") for w in weights]
    n = len(weights)

    fig, ax = plt.subplots(figsize=(10, max(3, n * 0.35)))
    y = np.arange(n)
    ax.barh(y, rcp, color=colors, alpha=0.85)
    ax.axvline(100.0 / n, color="#dc2626", linestyle="--", linewidth=1.3,
               label=f"等风险目标 1/N={100/n:.1f}%")
    ax.set_yticks(y)
    ax.set_yticklabels(syms, fontsize=9)
    ax.set_xlabel("Risk Contribution (%)")
    for i, v in enumerate(rcp):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_weights(data: dict, dpi: int) -> str:
    """图3: 名义权重条形（按资产类型着色）"""
    weights = sorted([w for w in data["weights"] if w["weight"] > 1e-6],
                     key=lambda w: w["weight"])
    syms = [w["symbol"][:6] for w in weights]
    vals = [w["weight"] * 100 for w in weights]
    colors = [_COLOR_BY_TYPE.get(w["asset_type"], "#999") for w in weights]

    fig, ax = plt.subplots(figsize=(10, max(3, len(syms) * 0.35)))
    y = np.arange(len(syms))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(syms, fontsize=9)
    ax.set_xlabel("Notional Weight (%)")
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def plot_corr_heatmap(data: dict, dpi: int) -> str:
    """图4: 样本相关矩阵热图（展示 LW 要稳定的对象）"""
    corr = np.array(data["corr_matrix"])
    labels = [l[:6] for l in data["corr_labels"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    return _fig_to_base64(fig, dpi)


def render_summary_cards(data: dict) -> str:
    m = data["metrics"]
    vol_converge = m.get("波动收敛(RP-等权)", 0)
    card_cls = "good" if vol_converge < 0 else "bad"
    return f"""
    <div class="cards-grid">
      <div class="card"><div class="label">组合年化波动</div><div class="value">{m['RP_年化波动']*100:.2f}<span class="unit">%</span></div></div>
      <div class="card {card_cls}"><div class="label">波动收敛(vs等权)</div><div class="value">{vol_converge*100:+.2f}<span class="unit">%</span></div></div>
      <div class="card" title="末次ERC权重在样本外协方差下的RC离散度；>1 反映协方差漂移（非求解失败）。口径：{m.get('样本外RC离散口径', '样本外')}"><div class="label">风险贡献离散度</div><div class="value">{m['样本外风险贡献离散度']:.3f}</div></div>
      <div class="card"><div class="label">LW 收缩 δ</div><div class="value">{data['lw_shrinkage']:.3f}</div></div>
      <div class="card"><div class="label">年化收益</div><div class="value">{m['RP_年化收益']*100:+.2f}<span class="unit">%</span></div></div>
      <div class="card"><div class="label">最大回撤</div><div class="value neg-text" style="color:#dc2626">{m['RP_最大回撤']*100:.2f}<span class="unit">%</span></div></div>
      <div class="card"><div class="label">夏普</div><div class="value">{m['RP_夏普']:.2f}</div></div>
      <div class="card"><div class="label">资金利用率</div><div class="value">{data['capital_utilization']*100:.0f}<span class="unit">%</span></div></div>
    </div>"""


def render_comparison(data: dict) -> str:
    m = data["metrics"]
    bench_ret = m.get("Benchmark_年化收益")
    bench_vol = m.get("Benchmark_年化波动")
    bench_row = ""
    if bench_ret is not None:
        bench_row = (f"<tr><td>沪深300</td><td class='num'>{bench_vol*100:.2f}%</td>"
                     f"<td class='num'>{bench_ret*100:+.2f}%</td>"
                     f"<td class='num'>—</td><td class='num'>—</td></tr>")
    return f"""
    <h2>三方对比</h2>
    <table>
      <thead><tr><th>组合</th><th>年化波动</th><th>年化收益</th><th>最大回撤</th><th>夏普</th></tr></thead>
      <tbody>
        <tr><td><strong>风险平价</strong></td><td class='num'>{m['RP_年化波动']*100:.2f}%</td>
            <td class='num {'pos' if m['RP_年化收益']>0 else 'neg'}'>{m['RP_年化收益']*100:+.2f}%</td>
            <td class='num neg'>{m['RP_最大回撤']*100:.2f}%</td><td class='num'>{m['RP_夏普']:.2f}</td></tr>
        <tr><td>等权(1/N)</td><td class='num'>{m['等权_年化波动']*100:.2f}%</td>
            <td class='num {'pos' if m['等权_年化收益']>0 else 'neg'}'>{m['等权_年化收益']*100:+.2f}%</td>
            <td class='num neg'>{m['等权_最大回撤']*100:.2f}%</td><td class='num'>{m['等权_夏普']:.2f}</td></tr>
        {bench_row}
      </tbody>
    </table>"""


def render_weight_table(data: dict) -> str:
    rows = []
    for w in sorted(data["weights"], key=lambda x: -x["weight"]):
        if w["weight"] <= 1e-6:
            continue
        mr = f"{w['margin_rate']*100:.0f}%" if w["margin_rate"] else "—"
        rows.append(
            f"<tr><td>{w['symbol']}</td><td>{w['asset_type']}</td>"
            f"<td class='num'>{w['weight']*100:.1f}%</td>"
            f"<td class='num'>{w['risk_contribution_pct']*100:.1f}%</td>"
            f"<td class='num'>{w['volatility']*100:.1f}%</td>"
            f"<td class='num'>{mr}</td></tr>")
    n = len(data["weights"])
    return f"""
    <h2>持仓与风险贡献</h2>
    <p class="note">名义权重为风险平价求解结果；风险贡献占比应接近 {100/n:.1f}%（1/N）。
       期货 margin_rate 来自 get_future_detail，资金占用 = 名义权重 × margin_rate。</p>
    <table>
      <thead><tr><th>代码</th><th>类型</th><th>名义权重</th><th>风险贡献</th><th>年化波动</th><th>保证金率</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def render_html(data: dict, dpi: int) -> str:
    m = data["metrics"]
    img_curve = plot_equity_curve(data, dpi)
    img_rc = plot_risk_contribution(data, dpi)
    img_w = plot_weights(data, dpi)
    img_corr = plot_corr_heatmap(data, dpi)

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>风险平价组合(ERC)月度 rebalance 报告</title><style>{_CSS}</style></head>
<body>
  <header>
    <h1>风险平价组合(ERC) · 月度 rebalance 报告</h1>
    <div class="meta">{m['评估口径']}</div>
  </header>

  {render_summary_cards(data)}

  <h2>累计净值与回撤</h2>
  <img src="data:image/png;base64,{img_curve}"/>

  {render_comparison(data)}

  <h2>风险贡献（ERC 核心验证）</h2>
  <p class="note">每类资产对组合风险的真实贡献。风险平价的目标是让所有资产的风险贡献相等（= 1/N 红虚线）。
     若条形高度接近红虚线，说明等风险达成；{'当前离散度 ' + format(data['rc_max_over_min'], '.3f') + '，接近 1.0 即等风险。'}
     注：此处为【训练段】均衡性（≈1.0）；样本外因协方差漂移离散度会升高（见上方"风险贡献离散度"卡片）。</p>
  <img src="data:image/png;base64,{img_rc}"/>

  <h2>名义权重</h2>
  <img src="data:image/png;base64,{img_w}"/>
  {render_weight_table(data)}

  <h2>资产相关矩阵</h2>
  <p class="note">样本相关矩阵（LW 收缩处理前的原始相关结构）。指数与 ETF 标的重叠处会出现高相关块，
     这是 LW 收缩要稳定的对象（δ={data['lw_shrinkage']:.3f}，ρ̄={data['mean_correlation']:.3f}）。</p>
  <img src="data:image/png;base64,{img_corr}"/>

  <footer>
    <h3>口径与注意</h3>
    <p>· 权重为<strong>名义敞口权重</strong>（Σw=1）。期货保证金交易，资金利用率 {data['capital_utilization']*100:.0f}%，
       剩余为杠杆空间。<br>
       · 协方差用<strong>手写 Ledoit-Wolf 常数相关收缩</strong>稳定（δ 越大越依赖收缩目标）。<br>
       · 月末调仓，过去 {m['lookback']} 日估协方差求 ERC 权重，下月持有，无未来函数。<br>
       · 合成 fixture 数据下收益无统计意义，真实策略效果需联网复测。
    </p>
  </footer>
</body></html>"""


def validate_payload(data: dict) -> None:
    assert "metrics" in data and "rp_curve" in data and "weights" in data
    assert len(data["rp_curve"]) > 0
    assert len(data["weights"]) > 0


def main() -> None:
    p = argparse.ArgumentParser(description="风险平价组合 HTML 报告")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json-path", default=str(REPORTS_DIR / "backtest_result.json"))
    p.add_argument("--html-path", default=str(REPORTS_DIR / "report.html"))
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    print("[1/3] 运行回测 " + ("(offline=True)" if args.offline or os.getenv("PANDA_DATA_OFFLINE") == "1" else "") + " ...")
    data = run_backtest_with_series(offline=args.offline)
    validate_payload(data)

    print(f"[2/3] 保存中间结果 → {args.json_path}")
    save_backtest_result(data, args.json_path)

    print(f"[3/3] 渲染 HTML 报告（dpi={args.dpi}）...")
    html = render_html(data, args.dpi)
    Path(args.html_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.html_path).write_text(html, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[OK] 报告已生成")
    print(f"     HTML: {args.html_path}")
    print(f"     JSON: {args.json_path}")
    print(f"     调仓次数: {data['n_rebalances']}")
    print(f"     RP年化波动={data['metrics']['RP_年化波动']*100:.2f}%, "
          f"夏普={data['metrics']['RP_夏普']:.2f}, RC离散度={data['rc_max_over_min']:.3f}")
    print("=" * 60)

    if args.open:
        import webbrowser
        webbrowser.open(Path(args.html_path).resolve().as_uri())


if __name__ == "__main__":
    main()

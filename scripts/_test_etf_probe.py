"""ETF 取数失败根因探测 —— 系统排查为什么取不到 A 股 ETF 日线

⚠️ 历史结论已纠正：本脚本漏测 get_fund_daily，曾误判「数据源不覆盖 ETF」。
   实际 ETF（510300/511260/518880）走 get_fund_daily 可取（返回 OHLCV 15列，
   日期跨度 ≤1年）。保留本脚本作排查记录；正确取数见 data_loader.load_fund_data。

逐项验证：代码格式 / 接口选择 / 区间 / fields / 批量 / 标的是否被识别。
对照能取的真股票(600519)，定位是「数据源不覆盖 ETF」还是「调用方式问题」。

用法（需凭证）：
    python _test_etf_probe.py
"""
import os
import panda_data

panda_data.init_token(username=os.environ["PANDA_DATA_USERNAME"],
                      password=os.environ["PANDA_DATA_PASSWORD"])

# 沪深典型 ETF + 确认可取的真股票参照
ETF_SYMS = ["510300.SH", "518880.SH", "511260.SH", "159915.SZ"]
STOCK_REF = "600519.SH"

# 先用交易日历 get_trade_cal 取最近交易日作为 D2（日期落在真实交易日，排除周末/假日因素）
import datetime as _dt
_now = _dt.datetime.now()
_cal = panda_data.get_trade_cal(
    start_date=(_now - _dt.timedelta(days=30)).strftime("%Y%m%d"),
    end_date=_now.strftime("%Y%m%d"), exchange="SH", is_trading_day=1, fields=[])
_trades = sorted(_cal["nature_date"].astype(str).tolist())
D2 = _trades[-1]
D1 = _trades[-30] if len(_trades) >= 30 else _trades[0]
print(f"[日期] 交易日历最近交易日 D2={D2}，探测区间 D1={D1}（均为真实交易日）\n")


def show(tag, df):
    """统一打印接口返回概要"""
    if df is None:
        print(f"  [{tag:16s}] None")
        return
    print(f"  [{tag:16s}] shape={df.shape} empty={df.empty}")
    if not df.empty:
        print(f"       cols={list(df.columns)}")
        print("       head:\n" + df.head(1).to_string())


def safe(tag, fn):
    """包裹一次接口调用，异常不中断"""
    try:
        show(tag, fn())
    except Exception as e:  # noqa: BLE001
        print(f"  [{tag:16s}] ERR {str(e)[:100]}")


print("=" * 72)
print("【1】ETF 在 stock/index/market 三接口（默认参数，逐代码）")
for sym in ETF_SYMS:
    print(f"--- {sym} ---")
    safe("stock_daily",  lambda s=sym: panda_data.get_stock_daily(symbol=s, start_date=D1, end_date=D2))
    safe("index_daily",  lambda s=sym: panda_data.get_index_daily(symbol=s, start_date=D1, end_date=D2))
    safe("market(stock)", lambda s=sym: panda_data.get_market_data(symbol=s, start_date=D1, end_date=D2, type="stock"))

print("=" * 72)
print(f"【2】参照：真股票 {STOCK_REF}（确认接口本身能工作）")
safe("stock_daily", lambda: panda_data.get_stock_daily(symbol=STOCK_REF, start_date=D1, end_date=D2))

print("=" * 72)
print("【3】标的是否被识别：get_stock_detail / get_index_detail")
for sym in ["510300.SH", "518880.SH"]:
    print(f"--- {sym} ---")
    safe("stock_detail", lambda s=sym: panda_data.get_stock_detail(symbol=s))
    safe("index_detail", lambda s=sym: panda_data.get_index_detail(symbol=s))

print("=" * 72)
print("【4】区间敏感性：510300.SH 换历史区间（排除「近期数据缺失」）")
for a, b in [("20200101", "20201231"), ("20150101", "20251231")]:
    safe(f"stock {a[:4]}~{b[:4]}",
         lambda a=a, b=b: panda_data.get_stock_daily(symbol="510300.SH", start_date=a, end_date=b))

print("=" * 72)
print("【5】参数形态：批量 list 传参 / 指定 fields")
safe("list 批量", lambda: panda_data.get_stock_daily(symbol=["510300.SH", "518880.SH"], start_date=D1, end_date=D2))
safe("fields=close", lambda: panda_data.get_stock_daily(symbol="510300.SH", start_date=D1, end_date=D2, fields="close"))

print("=" * 72)
print("【6】市场总览：get_index_constituent / 概念成分里是否含 ETF 代码（侧面验证覆盖）")
safe("index_detail 000300", lambda: panda_data.get_index_detail(symbol="000300.SH"))

print("\n探测完成。⚠️ 本脚本漏测 get_fund_daily —— 实际 ETF（510300 等）走该接口可取。")
print("历史误判「数据源不覆盖 ETF」已纠正，正确取数见 data_loader.load_fund_data。")

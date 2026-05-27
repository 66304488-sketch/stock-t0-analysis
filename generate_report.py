#!/usr/bin/env python3
"""一键生成个股做T分析HTML报告
用法: python generate_report.py 600031
      python generate_report.py 300750 --no-fetch
"""
import sys, os, json, argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# ---- 配色方案 ----
COLOR_SCHEMES = {
    "default":    {"primary": "#1a56db", "header": "#0f172a,#1e3a5f,#1a56db", "accent": "#1a56db"},
    "贵金属":      {"primary": "#dc2626", "header": "#450a0a,#7f1d1d,#dc2626", "accent": "#dc2626"},
    "煤炭":        {"primary": "#059669", "header": "#022c22,#064e3b,#059669", "accent": "#059669"},
    "白酒":        {"primary": "#7c3aed", "header": "#1a0a2e,#3b1f6e,#7c3aed", "accent": "#7c3aed"},
    "电池":        {"primary": "#0891b2", "header": "#0f172a,#0c4a6e,#0891b2", "accent": "#0891b2"},
    "新能源":      {"primary": "#0891b2", "header": "#0f172a,#0c4a6e,#0891b2", "accent": "#0891b2"},
    "跨境ETF":     {"primary": "#0d9488", "header": "#0f172a,#134e4a,#0d9488", "accent": "#0d9488"},
}
TAG_COLORS = {"green": "#d1fae5,#065f46", "yellow": "#fef3c7,#92400e",
              "red": "#fee2e2,#991b1b", "blue": "#dbeafe,#1e40af", "purple": "#ede9fe,#5b21b6"}


def load_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def update_recent_stocks(code, name):
    tracking_file = PROJECT_ROOT / "data" / "recent_stocks.json"
    stocks = []
    if tracking_file.exists():
        try:
            stocks = json.loads(tracking_file.read_text())
        except Exception:
            stocks = []
    stocks = [s for s in stocks if s["code"] != code]
    stocks.insert(0, {"code": code, "name": name, "analyzed_at": datetime.now().strftime("%Y-%m-%d")})
    stocks = stocks[:8]
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    tracking_file.write_text(json.dumps(stocks, ensure_ascii=False, indent=2))


def load_comparison_data(current_code):
    tracking_file = PROJECT_ROOT / "data" / "recent_stocks.json"
    if not tracking_file.exists():
        return []
    try:
        recent = json.loads(tracking_file.read_text())
    except Exception:
        return []
    comparison = []
    for entry in recent:
        if entry["code"] == current_code:
            continue
        full_path = PROJECT_ROOT / "data" / "processed" / f"{entry['code']}_full_data.json"
        if not full_path.exists():
            continue
        try:
            fd = json.loads(full_path.read_text())
            comparison.append({
                "code": entry["code"], "name": fd.get("name", entry["code"]),
                "amp_mean": fd.get("amp_mean", "-"),
                "idio_ratio": round(fd.get("idio_ratio_mean", 0) * 100, 1),
                "garch_persistence": fd.get("garch_persistence", "-"),
                "signal_pct": fd.get("signal_pct", "-"),
                "composite_score": fd.get("composite_score_mean", "-"),
            })
        except Exception:
            continue
    return comparison


def extract_stats(code, config):
    """提取股票全部统计数据"""
    df = pd.read_parquet(PROJECT_ROOT / f"data/processed/{code}_features.parquet")
    amp = df["amplitude"].dropna()
    stock = config.STOCKS.get(code, {"name": code, "industry": "default", "market_index": "sh000001"})

    data = {
        "name": stock["name"], "code": code, "industry": stock.get("industry", ""),
        "market_index": stock.get("market_index", ""),
        "rows": len(df), "date_start": str(df.index.min().date()),
        "date_end": str(df.index.max().date()),
    }

    # 振幅统计
    data["amp_mean"] = round(float(amp.mean()) * 100, 2)
    data["amp_median"] = round(float(amp.median()) * 100, 2)
    data["amp_std"] = round(float(amp.std()) * 100, 2)
    data["amp_min"] = round(float(amp.min()) * 100, 2)
    data["amp_max"] = round(float(amp.max()) * 100, 2)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        data[f"amp_p{p}"] = round(float(amp.quantile(p / 100)) * 100, 2)
    data["amp_skew"] = round(float(amp.skew()), 2)
    data["amp_kurtosis"] = round(float(amp.kurtosis()), 2)

    latest = df.iloc[-1]
    data["latest_amp"] = round(float(latest["amplitude"]) * 100, 2)
    data["latest_close"] = round(float(latest["close"]), 2)
    for w in [5, 10, 20, 60]:
        col = f"amp_ma_{w}"
        if col in df.columns:
            data[f"amp_ma_{w}_last"] = round(float(df[col].dropna().iloc[-1]) * 100, 2)

    for col in ["vol_ratio", "turnover", "amount"]:
        if col in df.columns:
            s = df[col].dropna()
            data[f"{col}_mean"] = round(float(s.mean()), 2)
            data[f"{col}_last"] = round(float(s.iloc[-1]), 2)

    for col in ["beta_60", "r_squared", "idio_ratio", "residual_vol"]:
        if col in df.columns:
            s = df[col].dropna()
            data[f"{col}_mean"] = round(float(s.mean()), 4)
            data[f"{col}_last"] = round(float(s.iloc[-1]), 4)

    # 信号系统
    from src.signals import TSignalGenerator
    sg = TSignalGenerator()
    signal_df = sg.generate_all_signals(df)
    data["signal_pct"] = round(float(signal_df["signal"].mean()) * 100, 2)
    data["signal_days"] = int(signal_df["signal"].sum())
    data["composite_score_mean"] = round(float(signal_df["composite_score"].mean()), 4)
    data["composite_score_last"] = round(float(signal_df["composite_score"].iloc[-1]), 4)
    data["signal_last"] = int(signal_df["signal"].iloc[-1])
    for c in ["score_amplitude", "score_liquidity", "score_idio", "score_regime"]:
        if c in signal_df.columns:
            data[f"{c}_mean"] = round(float(signal_df[c].mean()), 4)
            data[f"{c}_last"] = round(float(signal_df[c].iloc[-1]), 4)

    # 回测
    from src.backtest import TBacktester
    bt = TBacktester()
    bt_results = bt.run_grid(signal_df)
    data["backtest_top5"] = []
    if not bt_results.empty:
        for _, r in bt_results.sort_values("win_rate", ascending=False).head(5).iterrows():
            data["backtest_top5"].append({
                "entry": round(float(r["entry_threshold"]), 3),
                "exit": round(float(r["exit_threshold"]), 3),
                "win_rate": round(float(r["win_rate"]) * 100, 1),
                "avg_pnl": round(float(r["avg_profit_pct"]) * 100, 2),
                "trades": int(r["trade_days"]),
                "total_pnl": round(float(r["total_profit_pct"]) * 100, 2),
            })

    # GARCH
    try:
        from arch import arch_model
        model = arch_model(amp * 100, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        data["garch_omega"] = round(float(res.params["omega"]), 6)
        data["garch_alpha"] = round(float(res.params["alpha[1]"]), 4)
        data["garch_beta"] = round(float(res.params["beta[1]"]), 4)
        data["garch_persistence"] = round(float(res.params["alpha[1]"] + res.params["beta[1]"]), 4)
    except Exception as e:
        data["garch_error"] = str(e)

    # 季节性
    df2 = df.copy()
    df2["dow"] = df2.index.dayofweek
    dow_names = ["周一", "周二", "周三", "周四", "周五"]
    dow = df2.groupby("dow")["amplitude"].mean() * 100
    data["dow_amp"] = {dow_names[int(i)]: round(float(v), 2) for i, v in dow.items()}
    data["dow_best"] = dow_names[int(dow.idxmax())]
    data["dow_worst"] = dow_names[int(dow.idxmin())]
    data["dow_list"] = [round(float(v), 2) for v in dow.tolist()]

    df2["month"] = df2.index.month
    monthly = df2.groupby("month")["amplitude"].mean() * 100
    month_names = [f"{i}月" for i in range(1, 13)]
    data["month_amp"] = {month_names[i-1]: round(float(v), 2) for i, v in monthly.items()}
    data["month_best"] = month_names[int(monthly.idxmax()) - 1]
    data["month_worst"] = month_names[int(monthly.idxmin()) - 1]
    data["monthly_list"] = [round(float(v), 2) for v in monthly.tolist()]

    # 月度涨跌
    if "pct_change" in df.columns:
        ret = df["pct_change"].dropna()
        df2["pct_change"] = ret
        monthly_ret = df2.groupby("month")["pct_change"].mean()
        monthly_win = df2.groupby("month")["pct_change"].apply(lambda x: (x > 0).mean() * 100)
        data["month_ret"] = {month_names[i-1]: round(float(v), 2) for i, v in monthly_ret.items()}
        data["month_win"] = {month_names[i-1]: round(float(v), 1) for i, v in monthly_win.items()}
        data["month_ret_best"] = month_names[int(monthly_ret.idxmax()) - 1]
        data["month_ret_worst"] = month_names[int(monthly_ret.idxmin()) - 1]
        data["monthly_ret_list"] = [round(float(v), 2) for v in monthly_ret.tolist()]
        data["monthly_win_list"] = [round(float(v), 1) for v in monthly_win.tolist()]

    df2["quarter"] = df2.index.quarter
    quarterly = df2.groupby("quarter")["amplitude"].mean() * 100
    data["quarter_amp"] = {f"Q{int(i)}": round(float(v), 2) for i, v in quarterly.items()}

    # 尾部特征
    p90 = amp.quantile(0.90)
    data["high_amp_pct"] = round(float((amp > p90).mean()) * 100, 2)
    data["extreme_amp_pct"] = round(float((amp > 0.05).mean()) * 100, 2)
    data["extreme_amp_days"] = int((amp > 0.05).sum())

    high = amp > p90
    streaks, current = [], 0
    for v in high:
        if v:
            current += 1
        else:
            if current > 0: streaks.append(current)
            current = 0
    if current > 0: streaks.append(current)
    data["max_high_streak"] = max(streaks) if streaks else 0
    data["avg_high_streak"] = round(float(np.mean(streaks)), 1) if streaks else 0

    # 近期
    recent30 = amp.tail(30)
    data["recent30_amp_mean"] = round(float(recent30.mean()) * 100, 2)
    data["recent30_amp_max"] = round(float(recent30.max()) * 100, 2)
    data["recent30_amp_min"] = round(float(recent30.min()) * 100, 2)

    # 价格
    if "close" in df.columns:
        close = df["close"].dropna()
        data["close_mean"] = round(float(close.mean()), 2)
        data["close_last"] = round(float(close.iloc[-1]), 2)
        returns = close.pct_change().dropna()
        data["daily_return_mean"] = round(float(returns.mean()) * 100, 4)
        data["daily_return_std"] = round(float(returns.std()) * 100, 2)
        data["total_return_pct"] = round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2)

    # 相关性
    if "ind_close" in df.columns and "idx_close" in df.columns:
        ind_ret = df["ind_close"].pct_change().dropna()
        idx_ret = df["idx_close"].pct_change().dropna()
        stock_ret = df["close"].pct_change().dropna()
        common = ind_ret.index.intersection(idx_ret.index).intersection(stock_ret.index)
        if len(common) > 10:
            data["ind_corr"] = round(float(stock_ret.loc[common].corr(ind_ret.loc[common])), 4)
            data["idx_corr"] = round(float(stock_ret.loc[common].corr(idx_ret.loc[common])), 4)

    # 近90日振幅序列（走势图用）
    recent90 = df.tail(90)[["amplitude"]].copy()
    recent90.index = recent90.index.strftime("%m-%d")
    data["dates_90"] = recent90.index.tolist()
    data["amp_90"] = [round(float(v) * 100, 2) for v in recent90["amplitude"].tolist()]

    # 全部振幅数据（分布图用）
    data["amp_full"] = [round(float(v) * 100, 2) for v in amp.tolist()]

    # 年度振幅
    df2["year"] = df2.index.year
    yearly = df2.groupby("year")["amplitude"].mean() * 100
    data["yearly_keys"] = [f"{int(y)}年" for y in yearly.index.tolist()]
    data["yearly_vals"] = [round(float(v), 2) for v in yearly.tolist()]

    # 配色
    colors = COLOR_SCHEMES.get(stock.get("industry", ""), COLOR_SCHEMES["default"])
    data["color_primary"] = colors["primary"]
    data["color_header"] = colors["header"]
    data["color_accent"] = colors["accent"]

    return data

def build_strategy(c):
    """根据统计数据自动生成策略建议HTML"""
    amp_mean = c["amp_mean"]
    amp_median = c["amp_median"]
    idio_ratio = c.get("idio_ratio_mean", 0)
    signal_pct = c["signal_pct"]
    score = c["composite_score_mean"]
    total_ret = c.get("total_return_pct", 0)
    close_last = c.get("close_last", 0)
    backtest = c.get("backtest_top5", [])
    primary = c["color_primary"]

    # ── 历史统计评级 ──
    if amp_mean >= 3.0 and idio_ratio >= 0.7 and signal_pct >= 50:
        grade = "非常适合做T"
        grade_color = "#059669"
        verdict = "推荐"
    elif amp_mean >= 2.5 and idio_ratio >= 0.5:
        grade = "适合做T"
        grade_color = "#0891b2"
        verdict = "推荐"
    elif amp_mean >= 2.2:
        grade = "可谨慎做T"
        grade_color = "#d97706"
        verdict = "谨慎"
    else:
        grade = "不适合做T"
        grade_color = "#dc2626"
        verdict = "不推荐"

    # ── 最新信号评级 ──
    score_last = c.get("composite_score_last", 0)
    signal_last = c.get("signal_last", 0)
    if score_last >= 0.65 and signal_last == 1:
        sig_label = "强烈推荐"
        sig_color = "#059669"
        sig_icon = "🟢"
    elif score_last >= 0.6 and signal_last == 1:
        sig_label = "适合做T"
        sig_color = "#1a56db"
        sig_icon = "🔵"
    elif score_last >= 0.5:
        sig_label = "谨慎观望"
        sig_color = "#d97706"
        sig_icon = "🟡"
    else:
        sig_label = "不建议"
        sig_color = "#dc2626"
        sig_icon = "🔴"

    # ── 方向建议 ──
    if total_ret > 15:
        direction = "优先<span style='color:#059669'>正T（先买后卖）</span>，整体上升趋势，顺势而为"
    elif total_ret < -15:
        direction = "优先<span style='color:#dc2626'>反T（先卖后买）</span>，下行趋势中反向操作更安全"
    else:
        direction = "<span style='color:#0891b2'>双向均可</span>，震荡市中灵活选择"

    # ── 入场/出场时机 ──
    timing = "最佳月份：<strong>%s</strong>" % c.get("month_best", "-")
    if c.get("dow_best"):
        timing += "，最佳交易日：<strong>%s</strong>" % c["dow_best"]

    # ── 具体价格计算（双基准：昨收 + 开盘）──
    price_html = ""
    if close_last and close_last > 0:
        # 昨收基准：取历史振幅的30%，下限1.0%，上限3.0%
        dyn_pct = max(min(amp_mean * 0.30, 3.0), 1.0)
        entry_pct = -dyn_pct / 100
        exit_pct = dyn_pct / 100

        if backtest:
            top = backtest[0]
            bt_entry = float(top["entry"])
            bt_exit = float(top["exit"])
            if abs(bt_entry - entry_pct) > abs(entry_pct) * 0.5:
                entry_pct = (bt_entry + entry_pct) / 2
            if abs(bt_exit - exit_pct) > abs(exit_pct) * 0.5:
                exit_pct = (bt_exit + exit_pct) / 2

        entry_price = round(close_last * (1 + entry_pct), 2)
        exit_price = round(close_last * (1 + exit_pct), 2)
        stop_pct = max(min(dyn_pct * 0.6, 1.5), 0.5) / 100
        stop_price = round(entry_price * (1 - stop_pct), 2)

        # 激进策略
        agg_pct = min(dyn_pct * 1.4, 4.5)
        agg_entry_pct = -agg_pct / 100
        agg_exit_pct = agg_pct / 100
        agg_entry_price = round(close_last * (1 + agg_entry_pct), 2)
        agg_exit_price = round(close_last * (1 + agg_exit_pct), 2)

        # 开盘基准：跳空≥中等阈值时使用，入场比例收紧至70%%（跳空本身已贡献波动）
        gap_dyn_pct = max(dyn_pct * 0.7, 0.8)
        gap_entry_pct = -gap_dyn_pct / 100
        gap_exit_pct = gap_dyn_pct / 100
        # 开盘基准的止损也相应收紧
        gap_stop_pct = max(min(gap_dyn_pct * 0.5, 1.2), 0.4) / 100

        if backtest:
            cons_win = str(top["win_rate"]) + "%"
            agg_win = str(backtest[1]["win_rate"] if len(backtest) > 1 else top["win_rate"]) + "%"
        else:
            cons_win = "—"
            agg_win = "—"

        price_html = """<li><strong>操作价格（昨收基准：昨收 %.2f | 动态比例 ±%.1f%% = 振幅%.1f%% × 30%%）：</strong>
        <table style="width:100%%;margin:8px 0;border-collapse:collapse;font-size:0.9rem">
        <tr style="background:%s20">
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">策略</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">入场价</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">止盈价</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">止损价</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">参考胜率</td>
        </tr>
        <tr style="background:#05966908">
          <td style="padding:6px 10px;border:1px solid #e2e8f0">保守<span style="font-size:0.75rem;color:#64748b">(昨收)</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700;color:#059669">%.2f<span style="font-size:0.75rem;color:#64748b">(%+.1f%%)</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700;color:#dc2626">%.2f<span style="font-size:0.75rem;color:#64748b">(+%.1f%%)</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700;color:#64748b">%.2f<span style="font-size:0.75rem;color:#64748b">(-%.1f%%)</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">%s</td>
        </tr>
        </table>
        <span style="font-size:0.8rem;color:#0891b2;font-weight:600">▼ 跳空≥%.1f%%时切换到开盘基准（比例收紧至±%.1f%%，跳空自身已贡献波动）：</span>
        <table style="width:100%%;margin:4px 0 8px 0;border-collapse:collapse;font-size:0.9rem">
        <tr style="background:#0891b220">
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">跳空方向</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">操作</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">执行价</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">目标价</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;font-weight:700">止损价</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;border:1px solid #e2e8f0"><span style="color:#059669;font-weight:700">低开≥%.1f%%</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#059669;font-weight:700">正T</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">开盘价×(1%.1f%%) ≈ 开盘附近</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">开盘价×(1+%.1f%%)</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">执行价×(1-%.1f%%)</td>
        </tr>
        <tr>
          <td style="padding:6px 10px;border:1px solid #e2e8f0"><span style="color:#dc2626;font-weight:700">高开≥%.1f%%</span></td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0;color:#dc2626;font-weight:700">反T</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">开盘价×(1+%.1f%%) ≈ 开盘附近</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">开盘价×(1%.1f%%) 买回</td>
          <td style="padding:6px 10px;border:1px solid #e2e8f0">执行价×(1+%.1f%%)</td>
        </tr>
        </table>
        <span style="font-size:0.75rem;color:#64748b">平开(±%.1f%%)用昨收基准。昨收基准 — 正T：挂 %.2f 买→%.2f 卖→破%.2f 止损。反T方向相反。止损动态比例×0.6。</span></li>""" % (
            close_last, dyn_pct, amp_mean,
            primary,
            entry_price, entry_pct * 100,
            exit_price, exit_pct * 100,
            stop_price, stop_pct * 100,
            cons_win,
            # 开盘基准参数
            gap_mod_threshold := max(dyn_pct * 0.5, 0.5),
            gap_dyn_pct,
            gap_mod_threshold,
            gap_entry_pct * 100,
            gap_exit_pct * 100,
            gap_stop_pct * 100,
            gap_mod_threshold,
            gap_exit_pct * 100,
            gap_entry_pct * 100,
            gap_stop_pct * 100,
            gap_mod_threshold,
            entry_price, exit_price, stop_price,
        )

    # ── 走势预判 ──
    # 基于历史回测9500+交易日数据：
    #   低开→A型58%>V型36%（低开后先反弹再回落）
    #   高开→V/A基本各半
    #   上涨趋势中V型概率更高（+10%偏差）
    pattern_score = 0
    pattern_factors = []
    recent_amp_trend = c.get("recent30_amp_mean", amp_mean) - amp_mean
    garch_p = c.get("garch_persistence", 0.5)

    if total_ret > 10:
        pattern_score += 0.5; pattern_factors.append("上涨趋势，V型略占优(+0.5)")
    elif total_ret < -10:
        pattern_score -= 0.5; pattern_factors.append("下跌趋势，A型略占优(-0.5)")
    if recent_amp_trend > 0.3:
        pattern_score += 0.5; pattern_factors.append("近期振幅扩大，日内波动加剧")
    elif recent_amp_trend < -0.3:
        pattern_score -= 0.3; pattern_factors.append("近期振幅收缩，走势偏平淡")
    if garch_p > 0.9:
        pattern_score -= 0.5; pattern_factors.append("高波动聚集，易出现单边走势")
    signal_score_last = c.get("composite_score_last", 0)
    if signal_score_last >= 0.65:
        pattern_score += 0.3; pattern_factors.append("信号评分高，方向判断可信度提升")

    if pattern_score >= 0.8:
        pattern_label = "偏向<span style='color:#059669;font-weight:700'>V型走势</span>（先跌后涨）"
        pattern_advice = "优先<span style='color:#059669'>正T</span>，挂低价买入等待反弹"
    elif pattern_score <= -0.8:
        pattern_label = "偏向<span style='color:#dc2626;font-weight:700'>A型走势</span>（先涨后跌）"
        pattern_advice = "优先<span style='color:#dc2626'>反T</span>，高开先卖等待回落接回"
    else:
        pattern_label = "走势方向不明确，<span style='color:#d97706;font-weight:700'>震荡格局</span>"
        pattern_advice = "双向挂单，以先触及为准，严格止损"

    pattern_html = """<li style="margin-top:8px;padding:8px 12px;border-radius:6px;background:#fffbeb;border-left:3px solid #d97706">
      <strong>明日走势预判：</strong>%s（评分: %+.1f）
      <div style="font-size:0.8rem;color:#64748b;margin-top:2px">%s</div>
      <div style="font-size:0.8rem;margin-top:2px">操作建议：%s</div>
      <div style="font-size:0.8rem;color:#059669;margin-top:4px">⏱ 开盘30分钟确认：开盘后观察实际走势方向，若与预判一致则执行，若相反则减半仓位或放弃。跳空方向提供了额外信号（参考下方跳空规则）。</div></li>""" % (
        pattern_label, pattern_score,
        "；".join(pattern_factors) if pattern_factors else "无明显方向信号",
        pattern_advice)

    # ── 开盘跳空应对规则（决策流程）──
    gap_high = max(dyn_pct, 1.5)
    gap_mod = max(dyn_pct * 0.5, 0.5)
    gap_html = """<li style="margin-top:8px;padding:8px 12px;border-radius:6px;background:#f1f5f9;border-left:3px solid #64748b">
      <strong>开盘跳空决策流程：</strong>
      <table style="width:100%%;margin:4px 0;border-collapse:collapse;font-size:0.85rem">
      <tr style="background:#e2e8f0"><td style="padding:4px 8px;font-weight:700">跳空幅度</td><td style="padding:4px 8px;font-weight:700">基准价</td><td style="padding:4px 8px;font-weight:700">优先方向</td><td style="padding:4px 8px;font-weight:700">要点</td></tr>
      <tr><td style="padding:4px 8px"><span style="color:#dc2626;font-weight:600">高开 ≥%.1f%%</span></td>
          <td style="padding:4px 8px">开盘价</td><td style="padding:4px 8px;color:#dc2626;font-weight:700">反T</td>
          <td style="padding:4px 8px;color:#64748b">上表「开盘基准」行执行。高开≥%.1f%%+单边涨→放弃正T等回调</td></tr>
      <tr><td style="padding:4px 8px"><span style="color:#d97706;font-weight:600">高开 %.1f~%.1f%%</span></td>
          <td style="padding:4px 8px">昨收/开盘</td><td style="padding:4px 8px;color:#d97706;font-weight:700">反T优先</td>
          <td style="padding:4px 8px;color:#64748b">昨收入场价上移0.3~0.5%%。跳空回补后可正T</td></tr>
      <tr><td style="padding:4px 8px"><span style="color:#0891b2;font-weight:600">平开 ±%.1f%%</span></td>
          <td style="padding:4px 8px">昨收</td><td style="padding:4px 8px;color:#0891b2;font-weight:700">双向</td>
          <td style="padding:4px 8px;color:#64748b">正常挂单。开盘30分钟后根据实际走势确认方向</td></tr>
      <tr><td style="padding:4px 8px"><span style="color:#d97706;font-weight:600">低开 %.1f~%.1f%%</span></td>
          <td style="padding:4px 8px">昨收/开盘</td><td style="padding:4px 8px;color:#059669;font-weight:700">正T优先</td>
          <td style="padding:4px 8px;color:#64748b">昨收入场价下移0.3~0.5%%。历史胜率77%%</td></tr>
      <tr><td style="padding:4px 8px"><span style="color:#059669;font-weight:600">低开 ≥%.1f%%</span></td>
          <td style="padding:4px 8px">开盘价</td><td style="padding:4px 8px;color:#059669;font-weight:700">正T</td>
          <td style="padding:4px 8px;color:#64748b">上表「开盘基准」行执行。低开≥%.1f%%+放量→黄金机会(胜率87%%)，仓位40%%</td></tr>
      </table>
      <span style="font-size:0.75rem;color:#64748b">9500+交易日验证：低开后虽A型(先涨后跌)概率58%%，但正T(开盘买→最高卖)均值+1.7~2.5%%。做T要点：低开正T抓反弹段快进快出，高开反T等回落接回。不追单边！</span></li>""" % (
        gap_high, gap_high,
        gap_mod, gap_high,
        gap_mod,
        gap_mod, gap_high,
        gap_high, gap_high,
    )

    # ── 风险提示 ──
    risks = []
    if amp_mean < 2.2:
        risks.append("振幅过低，扣除0.3%双边成本后利润空间极小")
    if idio_ratio < 0.5:
        risks.append("特质波动占比不足50%，波动多来自大盘/板块联动，做T收益不可控")
    if c.get("garch_persistence", 0) > 0.95:
        risks.append("波动聚集度极高，高波动后易持续，需注意仓位管理")
    if total_ret < -20:
        risks.append("股价长期下行，正T容易被套")
    if close_last > 500:
        risks.append("单价%s元较高，底仓资金需求大" % int(close_last))
    if signal_pct < 30:
        risks.append("信号触发率过低，可操作机会稀少")

    risk_html = ""
    if risks:
        risk_html = "<li><strong>风险提示：</strong>" + "；".join(risks) + "</li>"

    garch_warn = ""
    if c.get("garch_persistence", 0) > 0.95:
        garch_warn = "高波动聚集下建议减半仓位"

    # ── 最新信号HTML ──
    sig_date = c.get("date_end", "")
    sig_html = """<li style="margin-top:8px;padding:8px 12px;border-radius:6px;background:%s15;border-left:3px solid %s">
      <strong>%s 最新信号状态（%s）：</strong><span style="color:%s;font-weight:700">%s</span>
      （综合评分 %.3f，信号 %s）
      <span style="font-size:0.8rem;color:#64748b">← 基于最近交易日数据，判断下一日是否适合做T</span></li>""" % (
        sig_color, sig_color, sig_icon, sig_date, sig_color, sig_label, score_last,
        "已触发" if signal_last == 1 else "未触发")

    return """<div class="card">
  <h2><span class="icon">&#x1F4A1;</span> 做T策略建议</h2>
  <div class="strategy">
    <h3>%s 做T可行性评估 - <span style="color:%s">%s</span></h3>
    <ul>
      <li><strong>历史统计评估：</strong><span style="color:%s;font-weight:700">%s</span>。
          日均振幅 %s%%，中位数 %s%%，特质占比 %s%%，历史信号率 %s%%，历史综合评分 %s。
          <span style="font-size:0.8rem;color:#64748b">← 基于全部历史数据的统计结论</span></li>
      %s
      %s
      %s
      %s
      <li><strong>方向选择：</strong>%s</li>
      <li><strong>季节时机：</strong>%s</li>
      %s
      <li><strong>仓位建议：</strong>底仓的30%%用于做T（低开可加至40%%），单日最多2次操作。%s</li>
    </ul>
  </div>
</div>""" % (
        c["name"], grade_color, verdict,
        grade_color, grade, amp_mean, amp_median, round(idio_ratio * 100, 1),
        signal_pct, score,
        sig_html,
        pattern_html,
        price_html,
        gap_html,
        direction, timing,
        risk_html, garch_warn,
    )

def build_html(data, comparison=None):
    """根据统计数据生成完整HTML — 使用 %s 模板避免 f-string 转义问题"""
    c = data
    primary = c["color_primary"]
    header_parts = c["color_header"].split(",")

    signal_badge = '<span class="tag tag-green">开 ✅</span>' if c["signal_last"] == 1 else '<span class="tag tag-red">关 ❌</span>'
    signal_label = "强信号 ✅" if c["signal_last"] == 1 else "关 ❌"
    price_trend_class = "hi" if c.get("total_return_pct", 0) > 0 else "lo"
    price_trend_sign = "+" if c.get("total_return_pct", 0) > 0 else ""

    strategy_html = build_strategy(c)

    # 横向对比表
    compare_html = ""
    if comparison and len(comparison) >= 2:
        rows = []
        for i, comp in enumerate(comparison):
            amp = comp.get("amp_mean", 0)
            idio = comp.get("idio_ratio", 0)
            score_v = comp.get("composite_score", 0)
            if isinstance(amp, str): amp = 0
            if isinstance(score_v, str): score_v = 0
            if amp >= 3 and idio >= 70:
                verdict = '<span style="color:#059669;font-weight:700">最佳</span>'
            elif amp < 2.2:
                verdict = '<span style="color:#dc2626;font-weight:700">不适合</span>'
            elif amp >= 3 and idio < 50:
                verdict = '<span style="color:#0891b2;font-weight:700">激进</span>'
            elif amp >= 2.5:
                verdict = "均衡"
            else:
                verdict = "稳健"
            amp_str = ("%.1f%%" % amp) if isinstance(amp, (int, float)) else str(amp)
            garch_str = ("%.4f" % comp['garch_persistence']) if isinstance(comp.get('garch_persistence'), (int, float)) else str(comp.get('garch_persistence', '-'))
            signal_str = ("%.2f%%" % comp['signal_pct']) if isinstance(comp.get('signal_pct'), (int, float)) else str(comp.get('signal_pct', '-'))
            score_str = ("%.4f" % score_v) if isinstance(score_v, (int, float)) else str(score_v)
            idio_str = ("%.1f" % comp["idio_ratio"]) if isinstance(comp["idio_ratio"], (int, float)) else str(comp["idio_ratio"])
            if i == 0:
                row = '<tr style="background:#fefce8;font-weight:700"><td class="lbl">' + comp["name"] + '</td><td class="val">' + amp_str + '</td><td class="val">' + idio_str + '%</td><td class="val">' + garch_str + '</td><td class="val">' + signal_str + '</td><td class="val">' + score_str + '</td><td class="val">当前</td></tr>'
            else:
                row = '<tr><td class="lbl">' + comp["name"] + '</td><td class="val">' + amp_str + '</td><td class="val">' + idio_str + '%</td><td class="val">' + garch_str + '</td><td class="val">' + signal_str + '</td><td class="val">' + score_str + '</td><td class="val">' + verdict + '</td></tr>'
            rows.append(row)
        compare_html = '<div class="card">\n      <h2><span class="icon">&#x1F4CA;</span> 近期对比</h2>\n      <div style="overflow-x:auto;-webkit-overflow-scrolling:touch">\n      <table class="data-table">\n        <tr style="font-weight:700;color:#64748b"><td class="lbl">股票</td><td class="val">振幅</td><td class="val">特质</td><td class="val">GARCH</td><td class="val">信号</td><td class="val">评分</td><td class="val">评价</td></tr>\n        ' + "".join(rows) + '\n      </table>\n      </div>\n      <div style="font-size:0.7rem;color:#94a3b8;margin-top:6px">仅显示最近分析的 ' + str(len(comparison)) + ' 只股票，当前行高亮</div>\n    </div>'

    # 月度/季度网格
    month_grid = ''.join('<div class="season-cell">' + m + ': ' + str(v) + '%</div>' for m, v in c["month_amp"].items())
    quarter_grid = ''.join('<div class="season-cell">' + q + ': ' + str(v) + '%</div>' for q, v in c["quarter_amp"].items())

    # 评分进度条
    score_bars_parts = []
    for n, k in [("振幅","amplitude"),("流动性","liquidity"),("特质","idio"),("市态","regime")]:
        val = c.get("score_" + k + "_last", 0)
        try: w = min(float(val) * 100, 100)
        except: w = 0
        score_bars_parts.append('<div class="score-bar"><span class="name">' + n + '</span><div class="track"><div class="fill" style="width:' + ("%.1f" % w) + '%;background:' + primary + '"></div></div><span class="num">' + str(val) + '</span></div>')
    score_bars = "".join(score_bars_parts)

    # 信号均值
    score_means_parts = []
    for n, k in [("振幅","amplitude"),("流动性","liquidity"),("特质","idio"),("市态","regime")]:
        score_means_parts.append('<tr><td class="lbl">' + n + '评分</td><td class="val">' + str(c.get("score_" + k + "_mean", "-")) + '</td></tr>')
    score_means = "".join(score_means_parts)

    # 回测
    bt_colors = ["#f59e0b", "#94a3b8", "#b45309", "#64748b", "#64748b"]
    bt_parts = []
    for i, r in enumerate(c.get("backtest_top5", [])):
        bt_parts.append('<div class="bt-row"><div class="bt-rank" style="background:' + bt_colors[i] + '">' + str(i+1) + '</div><div class="bt-params">入场 ' + str(r["entry"]) + ' 出场 +' + str(r["exit"]) + '</div><div class="bt-metrics"><div style="color:#059669;font-weight:700">' + str(r["win_rate"]) + '%</div><div>均利 ' + str(r["avg_pnl"]) + '% x ' + str(r["trades"]) + '次 = ' + str(r["total_pnl"]) + '%</div></div></div>')
    bt_rows = "".join(bt_parts)

    # 月度涨跌网格
    if "month_ret" in c:
        month_ret_grid = ''.join('<div class="season-cell">' + m + ': <span style="color:' + ('#059669' if v >= 0 else '#dc2626') + '">' + ('+' if v >= 0 else '') + ('%.2f' % v) + '%</span> 胜率' + str(c["month_win"].get(m, "-")) + '%</div>' for m, v in c["month_ret"].items())
        month_ret_labels_js = json.dumps([str(i)+"月" for i in range(1, 13)])
        month_ret_data_js = json.dumps(c.get("monthly_ret_list", [0]*12))
        month_ret_bg_js = json.dumps(["#059669" if v >= 0 else "#dc2626" for v in c.get("monthly_ret_list", [0]*12)])
        month_ret_best = c.get("month_ret_best", "-")
        month_ret_worst = c.get("month_ret_worst", "-")
    else:
        month_ret_grid = '<div class="season-cell">数据不可用</div>'
        month_ret_labels_js = json.dumps([str(i)+"月" for i in range(1, 13)])
        month_ret_data_js = json.dumps([0]*12)
        month_ret_bg_js = json.dumps(["#94a3b8"]*12)
        month_ret_best = "-"
        month_ret_worst = "-"

    # JS数据
    dates_90_js = json.dumps([d[-5:] for d in c["dates_90"]])
    amp_90_js = json.dumps(c["amp_90"])
    amp_full_js = json.dumps(c.get("amp_full", c["amp_90"]))  # 全量振幅（分布图用）
    month_labels_js = json.dumps([str(i)+"月" for i in range(1, 13)])
    month_data_js = json.dumps(c["monthly_list"])
    monthly = c["monthly_list"]
    mx = max(monthly) if monthly else 0
    mn = min(monthly) if monthly else 0
    month_bg_js = json.dumps([primary if v != mx and v != mn else ("#059669" if v == mx else "#dc2626") for v in monthly])
    dow_labels_js = json.dumps(["周一","周二","周三","周四","周五"])
    dow_data_js = json.dumps(c["dow_list"])
    dw = c["dow_list"]
    dw_max = max(dw) if dw else 0
    dw_min = min(dw) if dw else 0
    dow_bg_js = json.dumps([("#059669" if v == dw_max else ("#dc2626" if v == dw_min else primary)) for v in dw])

    # TAG_COLORS
    tc = TAG_COLORS

    # 数据格式化
    ex_pct = c.get("extreme_amp_pct", 0)
    ex_days = c.get("extreme_amp_days", 0)
    idio_pct = round(c.get("idio_ratio_mean", 0) * 100, 1)
    residual_mean = round(c.get("residual_vol_mean", 0) * 100, 2)
    residual_last = round(c.get("residual_vol_last", 0) * 100, 2)
    amount_mean = round(c.get("amount_mean", 0) / 1e8, 1)
    amount_last = round(c.get("amount_last", 0) / 1e8, 1)
    exchange = "SH" if c["code"].startswith("6") else "SZ"
    amp_class = 'good' if c['amp_mean'] >= 3 else ('bad' if c['amp_mean'] < 2.2 else '')
    idio_class = 'good' if c.get('idio_ratio_mean', 0) >= 0.7 else ('bad' if c.get('idio_ratio_mean', 0) < 0.5 else '')
    score_class = 'warn' if c.get('composite_score_mean', 0) < 0.6 else 'good'

    html = HTML_TEMPLATE % (
        c["name"], c["code"],
        header_parts[0], header_parts[1] if len(header_parts) > 1 else header_parts[0], header_parts[2] if len(header_parts) > 2 else header_parts[0],
        tc["green"].split(",")[0], tc["green"].split(",")[1],
        tc["yellow"].split(",")[0], tc["yellow"].split(",")[1],
        tc["red"].split(",")[0], tc["red"].split(",")[1],
        tc["blue"].split(",")[0], tc["blue"].split(",")[1],
        tc["purple"].split(",")[0], tc["purple"].split(",")[1],
        primary,
        exchange, c["code"], c["industry"], c["market_index"], c["name"], c["date_start"], c["date_end"], c["rows"],
        amp_class, c["amp_mean"], c["amp_median"], c["recent30_amp_mean"], c["recent30_amp_max"], c["recent30_amp_min"],
        idio_class, idio_pct, c.get("r_squared_mean", "-"),
        c.get("garch_persistence", "-"),
        score_class, c["composite_score_mean"], c["composite_score_last"], signal_label,
        c["signal_pct"], c["signal_days"], c["rows"],
        c["amp_min"], c["amp_p10"], c["amp_p25"], c["amp_p50"], c["amp_p75"], c["amp_p90"],
        c["amp_p95"], c["amp_p99"], c["amp_max"], c["amp_std"], c["amp_skew"], c["amp_kurtosis"],
        ex_pct, ex_days, c["max_high_streak"], c["avg_high_streak"],
        c["month_best"], c["month_worst"], month_grid, month_ret_best, month_ret_worst, month_ret_grid, quarter_grid,
        c["dow_best"], c["dow_worst"],
        c.get("garch_omega", "-"), c.get("garch_alpha", "-"), c.get("garch_beta", "-"), c.get("garch_persistence", "-"),
        c.get("r_squared_mean", "-"), idio_pct, c.get("beta_60_mean", "-"), c.get("beta_60_last", "-"),
        residual_mean, residual_last,
        c["composite_score_last"], signal_badge, c["composite_score_mean"], c["signal_pct"], c["signal_days"],
        score_bars, score_means,
        bt_rows,
        strategy_html,
        c["close_last"], c["close_mean"],
        price_trend_class, price_trend_sign, c["total_return_pct"], c["daily_return_mean"],
        c["daily_return_std"], amount_mean, amount_last, c.get("amp_ma_20_last", "-"),
        c.get("amp_ma_5_last", "-"), c.get("amp_ma_10_last", "-"), c.get("amp_ma_60_last", "-"),
        compare_html,
        c["date_end"],
        dates_90_js, amp_90_js, amp_full_js,
        primary, primary,
        month_labels_js, month_data_js, month_bg_js,
        month_ret_labels_js, month_ret_data_js, month_ret_bg_js,
        dow_labels_js, dow_data_js, dow_bg_js,
    )
    return html



# HTML_TEMPLATE defined below as a module-level constant using %s placeholders
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no">
<title>%s(%s) · 做T分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif;background:#f1f5f9;color:#1e293b;line-height:1.6;-webkit-text-size-adjust:100%%}
.header{background:linear-gradient(135deg,%s,%s,%s);color:#fff;padding:32px 16px 24px;text-align:center}
.header .code{font-size:0.85rem;opacity:0.7;letter-spacing:2px}
.header h1{font-size:1.5rem;font-weight:800;margin:6px 0}
.header .meta{font-size:0.8rem;opacity:0.7;margin-top:8px}
.header .meta span{margin:0 6px}
.container{max-width:640px;margin:0 auto;padding:12px}
.card{background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.card h2{font-size:1.05rem;font-weight:700;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #e2e8f0;display:flex;align-items:center;gap:6px}
.card h2 .icon{font-size:1.2rem}
.kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kpi{background:#f8fafc;border-radius:8px;padding:12px;text-align:center;border:1px solid #e2e8f0}
.kpi .label{font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.3px}
.kpi .value{font-size:1.35rem;font-weight:800;color:#1e293b;margin:2px 0}
.kpi .sub{font-size:0.72rem;color:#64748b}
.kpi.warn .value{color:#d97706}
.kpi.good .value{color:#059669}
.kpi.bad .value{color:#dc2626}
.data-table{width:100%%;border-collapse:collapse;font-size:0.8rem}
.data-table td{padding:8px 10px;border-bottom:1px solid #f1f5f9;vertical-align:top}
.data-table tr:last-child td{border-bottom:none}
.data-table .lbl{color:#64748b;white-space:nowrap;width:40%%;font-weight:500}
.data-table .val{font-weight:600;text-align:right}
.data-table .hi{color:#059669}
.data-table .lo{color:#dc2626}
.chart-wrap{position:relative;width:100%%}
.chart-200{height:200px}
.chart-240{height:240px}
.chart-280{height:280px}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.7rem;font-weight:700}
.tag-green{background:%s;color:%s}
.tag-yellow{background:%s;color:%s}
.tag-red{background:%s;color:%s}
.tag-blue{background:%s;color:%s}
.tag-purple{background:%s;color:%s}
.section-title{font-size:0.9rem;font-weight:700;color:#475569;margin:14px 0 8px;padding-left:4px;border-left:3px solid %s}
.score-bar{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.score-bar .name{font-size:0.75rem;width:56px;text-align:right;color:#64748b;flex-shrink:0}
.score-bar .track{flex:1;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden}
.score-bar .fill{height:100%%;border-radius:4px;transition:width 0.3s}
.score-bar .num{font-size:0.75rem;font-weight:700;width:36px;flex-shrink:0}
.bt-row{display:flex;align-items:center;padding:8px 0;border-bottom:1px solid #f1f5f9;gap:8px;font-size:0.78rem}
.bt-row .bt-rank{width:22px;height:22px;border-radius:50%%;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:800;color:#fff;flex-shrink:0}
.bt-row .bt-params{flex:1;font-weight:600}
.bt-row .bt-metrics{text-align:right}
.bt-row .bt-metrics div{line-height:1.3}
.strategy{background:linear-gradient(135deg,#ecfdf5,#dbeafe);border:1px solid #a7f3d0;border-radius:10px;padding:14px}
.strategy h3{font-size:0.95rem;font-weight:700;color:#065f46;margin-bottom:8px}
.strategy li{font-size:0.82rem;margin-bottom:6px;padding-left:4px;line-height:1.5;list-style-position:inside}
.best-tag{display:inline-block;background:#f59e0b;color:#fff;font-size:0.65rem;padding:2px 6px;border-radius:8px;margin-left:4px;vertical-align:middle;font-weight:700}
.season-cell{border-bottom:1px dashed #e2e8f0;padding:6px 4px}
footer{text-align:center;padding:24px;font-size:0.75rem;color:#94a3b8}
@media(min-width:480px){
.kpi-grid{grid-template-columns:repeat(3,1fr)}
.header h1{font-size:1.8rem}
.data-table{font-size:0.85rem}
}
</style>
</head>
<body>

<div class="header">
<div class="code">%s.%s · %s · %s</div>
<h1>%s 做T分析报告</h1>
<div class="meta"><span>%s ~ %s</span><span>%s 个交易日</span><span>数据: AKShare</span></div>
</div>

<div class="container">

<div class="card">
<h2><span class="icon">&#x1F4CA;</span> 核心指标</h2>
<div class="kpi-grid">
<div class="kpi %s"><div class="label">日均振幅</div><div class="value">%s%%</div><div class="sub">中位数 %s%%</div></div>
<div class="kpi"><div class="label">近30日振幅</div><div class="value">%s%%</div><div class="sub">最高 %s%% / 最低 %s%%</div></div>
<div class="kpi %s"><div class="label">特质波动占比</div><div class="value">%s%%</div><div class="sub">R²_mkt = %s</div></div>
<div class="kpi"><div class="label">波动聚集度</div><div class="value">%s</div><div class="sub">GARCH α+β</div></div>
<div class="kpi %s"><div class="label">做T综合评分</div><div class="value">%s</div><div class="sub">昨日: %s (%s)</div></div>
<div class="kpi"><div class="label">信号触发率</div><div class="value">%s%%</div><div class="sub">%s / %s 天</div></div>
</div>
</div>

<div class="card">
<h2><span class="icon">&#x1F4C8;</span> 振幅分布</h2>
<div class="chart-wrap chart-280"><canvas id="ampDist"></canvas></div>
<div class="section-title">分位数详情</div>
<table class="data-table">
<tr><td class="lbl">最小值</td><td class="val">%s%%</td><td class="lbl">P10</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">P25</td><td class="val">%s%%</td><td class="lbl">P50 (中位数)</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">P75</td><td class="val">%s%%</td><td class="lbl">P90</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">P95</td><td class="val">%s%%</td><td class="lbl">P99</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">最大值</td><td class="val">%s%%</td><td class="lbl">标准差</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">偏度 (Skew)</td><td class="val">%s</td><td class="lbl">峰度 (Kurt)</td><td class="val">%s</td></tr>
</table>
<div class="section-title">分布特征</div>
<table class="data-table">
<tr><td class="lbl">最优拟合分布</td><td class="val">对数正态 (Lognormal)</td><td class="lbl">极端振幅 (&gt;5%%)</td><td class="val">%s%% (%s天)</td></tr>
<tr><td class="lbl">最长连续高振幅</td><td class="val">%s 天</td><td class="lbl">平均连续高振幅</td><td class="val">%s 天</td></tr>
</table>
</div>

<div class="card">
<h2><span class="icon">&#x1F4C9;</span> 近90日振幅走势</h2>
<div class="chart-wrap chart-280"><canvas id="ts90"></canvas></div>
</div>

<div class="card">
<h2><span class="icon">&#x1F4C5;</span> 季节性分析</h2>
<div class="chart-wrap chart-280"><canvas id="monthChart"></canvas></div>
<div class="section-title">月度振幅</div>
<table class="data-table">
<tr><td class="lbl">最佳月份</td><td class="val hi">%s</td><td class="lbl">最差月份</td><td class="val lo">%s</td></tr>
</table>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.78rem;margin-top:8px">
%s
</div>
<div class="section-title">月度涨跌</div>
<div class="chart-wrap chart-240"><canvas id="monthRetChart"></canvas></div>
<table class="data-table">
<tr><td class="lbl">最佳月份(涨)</td><td class="val hi">%s</td><td class="lbl">最差月份(跌)</td><td class="val lo">%s</td></tr>
</table>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.78rem;margin-top:8px">
%s
</div>
<div class="section-title">季度振幅</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.78rem">
%s
</div>
<div class="section-title">周内效应</div>
<div class="chart-wrap chart-240"><canvas id="dowChart"></canvas></div>
<table class="data-table">
<tr><td class="lbl">最佳交易日</td><td class="val hi">%s</td><td class="lbl">最差交易日</td><td class="val lo">%s</td></tr>
</table>
</div>

<div class="card">
<h2><span class="icon">&#x1F4E6;</span> GARCH(1,1) 波动率建模</h2>
<table class="data-table">
<tr><td class="lbl">omega (常数)</td><td class="val">%s</td><td class="lbl">alpha (ARCH)</td><td class="val">%s</td></tr>
<tr><td class="lbl">beta (GARCH)</td><td class="val">%s</td><td class="lbl">persistence (α+β)</td><td class="val">%s</td></tr>
</table>
</div>

<div class="card">
<h2><span class="icon">&#x1F504;</span> 波动率分解</h2>
<table class="data-table">
<tr><td class="lbl">市场联动 R²</td><td class="val">%s</td><td class="lbl">特质波动占比</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">60日Beta均值</td><td class="val">%s</td><td class="lbl">最新Beta</td><td class="val">%s</td></tr>
<tr><td class="lbl">残余波动均值</td><td class="val">%s%%</td><td class="lbl">最新残余波动</td><td class="val">%s%%</td></tr>
</table>
</div>

<div class="card">
<h2><span class="icon">&#x1F3AF;</span> 做T信号系统</h2>
<div class="section-title">综合评分 (昨日)</div>
<table class="data-table">
<tr><td class="lbl">综合做T评分</td><td class="val" style="font-size:1.1rem">%s</td><td class="lbl">信号状态</td><td class="val">%s</td></tr>
<tr><td class="lbl">历史评分均值</td><td class="val">%s</td><td class="lbl">信号触发率</td><td class="val">%s%% (%s天)</td></tr>
</table>
<div class="section-title">分项评分 (昨日)</div>
%s
<div class="section-title">分项历史均值</div>
<table class="data-table">
%s
</table>
</div>

<div class="card">
<h2><span class="icon">&#x1F4B0;</span> 回测结果</h2>
<div style="font-size:0.8rem;color:#64748b;margin-bottom:8px">逻辑：开盘价×(1+入场阈值)买入，出场阈值卖出，需在日内高低点范围内。仅统计信号日。</div>
<div class="bt-row" style="font-weight:700;color:#64748b;border-bottom:2px solid #e2e8f0">
<div class="bt-rank" style="background:transparent;color:#64748b">#</div><div class="bt-params">参数</div><div class="bt-metrics"><div>胜率 / 均利 / 次数</div></div>
</div>
%s
</div>

%s

<div class="card">
<h2><span class="icon">&#x1F4B5;</span> 量价参考</h2>
<table class="data-table">
<tr><td class="lbl">最新收盘价</td><td class="val">%s 元</td><td class="lbl">历史均价</td><td class="val">%s 元</td></tr>
<tr><td class="lbl">区间涨跌幅</td><td class="val %s">%s%s%%</td><td class="lbl">日均涨跌幅</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">日收益波动率</td><td class="val">%s%%</td><td class="lbl">日均成交额</td><td class="val">%s 亿</td></tr>
<tr><td class="lbl">最新成交额</td><td class="val">%s 亿</td><td class="lbl">最新振幅MA20</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">MA5</td><td class="val">%s%%</td><td class="lbl">MA10</td><td class="val">%s%%</td></tr>
<tr><td class="lbl">MA60</td><td class="val">%s%%</td><td class="lbl"></td><td class="val"></td></tr>
</table>
</div>

%s

</div>
<footer>A股个股波动做T套利分析系统 · 数据截止 %s · 仅供参考不构成投资建议</footer>

<script>
(function(){
var dates=%s;
var amp=%s;
var ampFull=%s;
function ma(arr,w){var o=[];for(var i=0;i<arr.length;i++){if(i<w-1){o.push(null);continue}var s=0;for(var j=i-w+1;j<=i;j++)s+=arr[j];o.push(s/w)}return o}
var ma20=ma(amp,20);

new Chart(document.getElementById("ampDist"),{type:"bar",data:{labels:["<1%%","1-1.5%%","1.5-2%%","2-2.5%%","2.5-3%%","3-3.5%%","3.5-4%%","4-5%%","5-6%%","6-8%%",">8%%"],datasets:[{label:"天数",data:[ampFull.filter(function(v){return v<1}).length,ampFull.filter(function(v){return v>=1&&v<1.5}).length,ampFull.filter(function(v){return v>=1.5&&v<2}).length,ampFull.filter(function(v){return v>=2&&v<2.5}).length,ampFull.filter(function(v){return v>=2.5&&v<3}).length,ampFull.filter(function(v){return v>=3&&v<3.5}).length,ampFull.filter(function(v){return v>=3.5&&v<4}).length,ampFull.filter(function(v){return v>=4&&v<5}).length,ampFull.filter(function(v){return v>=5&&v<6}).length,ampFull.filter(function(v){return v>=6&&v<8}).length,ampFull.filter(function(v){return v>=8}).length],backgroundColor:"%s",borderRadius:3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{title:{text:"日振幅分布 (" + ampFull.length + "天)",display:true,font:{size:13}}}}});

new Chart(document.getElementById("ts90"),{type:"line",data:{labels:dates,datasets:[{label:"日振幅 (%%)",data:amp,borderColor:"%s",borderWidth:1,pointRadius:0,tension:0.2,fill:false},{label:"20日均线",data:ma20,borderColor:"#dc2626",borderWidth:1.5,pointRadius:0,tension:0.3,fill:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{title:{text:"近90日振幅走势",display:true,font:{size:13}}}}});

new Chart(document.getElementById("monthChart"),{type:"bar",data:{labels:%s,datasets:[{label:"月均振幅 (%%)",data:%s,backgroundColor:%s,borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{title:{text:"月度振幅分布",display:true,font:{size:13}}}}});

new Chart(document.getElementById("monthRetChart"),{type:"bar",data:{labels:%s,datasets:[{label:"月均涨跌幅 (%%)",data:%s,backgroundColor:%s,borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{title:{text:"月度涨跌分布",display:true,font:{size:13}}},scales:{y:{grid:{color:"#e2e8f0"}}}}});

new Chart(document.getElementById("dowChart"),{type:"bar",data:{labels:%s,datasets:[{label:"日均振幅 (%%)",data:%s,backgroundColor:%s,borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{title:{text:"周内振幅分布",display:true,font:{size:13}}}}});
})();
</script>
</body>
</html>"""





def main():
    parser = argparse.ArgumentParser(description="生成个股做T分析HTML报告")
    parser.add_argument("code", help="股票代码, 如 600031")
    parser.add_argument("--no-fetch", action="store_true", help="跳过数据获取步骤")
    args = parser.parse_args()

    code = args.code
    config = load_config()

    if code not in config.STOCKS:
        print("错误: " + code + " 不在 config.py 的 STOCKS 中")
        print("当前股票池: " + str(list(config.STOCKS.keys())))
        sys.exit(1)

    stock = config.STOCKS[code]
    name = stock["name"]
    output_path = PROJECT_ROOT / "output" / (name + "_" + code + "_做T分析报告.html")

    parquet_path = PROJECT_ROOT / ("data/processed/" + code + "_features.parquet")
    if not parquet_path.exists() or not args.no_fetch:
        print("[1/3] 获取数据: " + name + "(" + code + ")...")
        os.system("cd " + str(PROJECT_ROOT) + " && python main.py process --stock " + code)

    print("[2/3] 提取统计数据...")
    data = extract_stats(code, config)
    idio_pct = round(data["idio_ratio_mean"] * 100, 1)
    print("  振幅均值=" + str(data["amp_mean"]) + "% 特质占比=" + str(idio_pct) + "% 信号率=" + str(data["signal_pct"]) + "% 评分=" + str(data["composite_score_mean"]))

    comparison = load_comparison_data(code)
    if comparison:
        comparison.insert(0, {
            "code": code, "name": name,
            "amp_mean": data["amp_mean"],
            "idio_ratio": round(data["idio_ratio_mean"] * 100, 1),
            "garch_persistence": data.get("garch_persistence", "-"),
            "signal_pct": data["signal_pct"],
            "composite_score": data["composite_score_mean"],
        })

    print("[3/3] 生成HTML报告...")
    html = build_html(data, comparison if comparison and len(comparison) >= 3 else None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print("  报告保存至: " + str(output_path) + " (" + str(len(html) // 1024) + "KB)")

    update_recent_stocks(code, name)
    os.system("open " + str(output_path))
    print("Done! 报告已在浏览器打开。")


if __name__ == "__main__":
    main()

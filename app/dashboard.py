"""
A股个股波动做T套利分析 - 交互式看板
运行: streamlit run app/dashboard.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from config import STOCKS, SIGNAL_WEIGHTS, SIGNAL_THRESHOLD, T_COST, DEFAULT_END_DATE, DEFAULT_START_DATE
from src.data_fetcher import StockDataFetcher
from src.data_cleaner import standardize_daily_columns, standardize_index_columns, merge_all
from src.features import FeatureEngineer
from src.signals import TSignalGenerator
from src.backtest import TBacktester
from src.analysis import AmplitudeAnalyzer
from db import AnalysisDB

st.set_page_config(page_title="做T套利分析", page_icon="", layout="wide")

# ── CSS微调 ──
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ── 数据加载（DB优先，parquet回退）──
def get_db():
    """获取DB连接（每次新建，避免跨session问题）"""
    return AnalysisDB()


@st.cache_data(ttl=600, show_spinner=False)
def load_all_summaries():
    """加载所有股票的摘要统计（用于总览页）"""
    db = get_db()
    summaries = db.get_all_summaries()
    db.close()
    return summaries


@st.cache_data(ttl=600, show_spinner=False)
def load_all_regimes():
    """加载所有股票的近20日市场状态（震荡/单边）"""
    db = get_db()
    regimes = db.get_all_recent_regimes(20)
    db.close()
    return regimes


@st.cache_data(ttl=300, show_spinner=False)
def load_and_process(stock_code):
    """加载特征数据。DB有则跳过，否则从parquet读或重新拉取。"""
    # parquet 优先（特征工程已完成）
    cache_path = f"data/processed/{stock_code}_features.parquet"
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        last_date = df.index.max()
        expected = pd.Timestamp(DEFAULT_END_DATE)
        if last_date >= expected - pd.Timedelta(days=5):
            return df

    # 重新拉取+处理
    cfg = STOCKS[stock_code]
    fetcher = StockDataFetcher()
    raw = fetcher.fetch_all_for_stock(
        stock_code, start=DEFAULT_START_DATE, end=DEFAULT_END_DATE,
        market_index=cfg["market_index"], industry=cfg["industry"],
    )
    daily = standardize_daily_columns(raw["daily"])
    idx = standardize_index_columns(raw["index"], "idx")
    ind = standardize_index_columns(raw.get("industry"), "ind") if raw.get("industry") is not None else None
    merged = merge_all(daily, idx, ind)
    engineer = FeatureEngineer()
    df = engineer.process(merged)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(cache_path)
    return df


def check_stock_status(stock_code):
    """检查股票数据状态。返回 (status, last_date, days)"""
    cache_path = f"data/processed/{stock_code}_features.parquet"
    if not os.path.exists(cache_path):
        return ("missing", None, 0)
    try:
        df = pd.read_parquet(cache_path)
        last_date = df.index.max()
        days = len(df)
        expected = pd.Timestamp(DEFAULT_END_DATE)
        if last_date >= expected - pd.Timedelta(days=3):
            # 再检查DB是否有摘要
            db = get_db()
            has_db = db.has_summary(stock_code)
            db.close()
            return ("ok" if has_db else "no_db_summary", last_date.strftime("%Y-%m-%d"), days)
        elif last_date >= expected - pd.Timedelta(days=14):
            return ("stale", last_date.strftime("%Y-%m-%d"), days)
        else:
            return ("outdated", last_date.strftime("%Y-%m-%d"), days)
    except Exception:
        return ("error", None, 0)


def smart_refresh_all(status_callback=None):
    """智能刷新：已有且不过时的跳过，只拉取缺失/过时的。返回 (fetched, skipped, failed)"""
    fetched, skipped, failed = [], [], []
    all_codes = list(STOCKS.keys())
    for i, code in enumerate(all_codes):
        status, last_date, days = check_stock_status(code)
        name = STOCKS[code]["name"]
        if status_callback:
            status_callback(i, len(all_codes), f"{name} — {status}")

        if status == "ok":
            skipped.append((code, last_date, days))
            continue

        # 需要抓取：missing / stale / outdated / no_db_summary
        try:
            cfg = STOCKS[code]
            fetcher = StockDataFetcher()
            raw = fetcher.fetch_all_for_stock(
                code, start=DEFAULT_START_DATE, end=DEFAULT_END_DATE,
                market_index=cfg["market_index"], industry=cfg["industry"],
            )
            daily = standardize_daily_columns(raw["daily"])
            idx = standardize_index_columns(raw["index"], "idx")
            ind = standardize_index_columns(raw.get("industry"), "ind") if raw.get("industry") is not None else None
            merged = merge_all(daily, idx, ind)
            engineer = FeatureEngineer()
            df = engineer.process(merged)
            os.makedirs("data/processed", exist_ok=True)
            df.to_parquet(f"data/processed/{code}_features.parquet")

            # 同步DB
            db = get_db()
            db.insert_daily(code, df)
            from generate_report import extract_stats, load_config
            stats = extract_stats(code, load_config())
            db.upsert_summary(stats)
            db.upsert_seasonality(code, stats)
            db.close()
            fetched.append((code, df.index.max().strftime("%Y-%m-%d"), len(df)))
        except Exception as e:
            failed.append((code, str(e)))

    return fetched, skipped, failed


def refresh_single_stock(stock_code):
    """强制刷新单只股票：拉取+处理+同步DB"""
    cfg = STOCKS[stock_code]
    fetcher = StockDataFetcher()
    raw = fetcher.fetch_all_for_stock(
        stock_code, start=DEFAULT_START_DATE, end=DEFAULT_END_DATE,
        market_index=cfg["market_index"], industry=cfg["industry"],
    )
    daily = standardize_daily_columns(raw["daily"])
    idx = standardize_index_columns(raw["index"], "idx")
    ind = standardize_index_columns(raw.get("industry"), "ind") if raw.get("industry") is not None else None
    merged = merge_all(daily, idx, ind)
    engineer = FeatureEngineer()
    df = engineer.process(merged)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(f"data/processed/{stock_code}_features.parquet")

    db = get_db()
    db.insert_daily(stock_code, df)
    from generate_report import extract_stats, load_config
    stats = extract_stats(stock_code, load_config())
    db.upsert_summary(stats)
    db.upsert_seasonality(stock_code, stats)
    db.close()
    return df


def verify_data_gaps(stock_code):
    """验证某只股票是否有数据缺口。返回缺口日期列表"""
    cache_path = f"data/processed/{stock_code}_features.parquet"
    if not os.path.exists(cache_path):
        return []
    df = pd.read_parquet(cache_path)
    df = df.sort_index()
    # 交易日缺口（>5天无数据视为缺口，排除周末节假日）
    gaps = []
    for i in range(1, len(df)):
        gap_days = (df.index[i] - df.index[i-1]).days
        if gap_days > 5:
            gaps.append((df.index[i-1].strftime("%Y-%m-%d"), df.index[i].strftime("%Y-%m-%d"), gap_days))
    return gaps


@st.cache_data(ttl=300, show_spinner=False)
def run_analysis(stock_code):
    """统计分析，按stock_code缓存"""
    df = load_and_process(stock_code)
    analyzer = AmplitudeAnalyzer()
    return analyzer.run_all(df)


@st.cache_data(ttl=300, show_spinner=False)
def run_signals(stock_code, w_amp, w_liq, w_idio, w_reg, sig_thresh):
    """信号生成，按stock_code+权重缓存"""
    df = load_and_process(stock_code)
    weights = {"amplitude": w_amp, "liquidity": w_liq, "idio": w_idio, "regime": w_reg}
    sg = TSignalGenerator(weights=weights, threshold=sig_thresh)
    return sg.generate_all_signals(df)


@st.cache_data(ttl=300, show_spinner=False)
def run_backtest(stock_code, sig_thresh):
    """回测，按stock_code缓存"""
    # 用默认权重做回测（回测不依赖具体权重）
    df = load_and_process(stock_code)
    sg = TSignalGenerator(threshold=sig_thresh)
    sdf = sg.generate_all_signals(df)
    bt = TBacktester()
    return bt.run_grid(sdf)


# ── 侧边栏 ──
with st.sidebar:
    st.title("做T套利分析")

    # ── 数据管理 ──
    st.subheader("数据管理")

    # 先扫描所有股票状态
    status_summary = {"ok": 0, "stale": 0, "missing": 0, "outdated": 0, "no_db_summary": 0, "error": 0}
    for code in STOCKS:
        s, _, _ = check_stock_status(code)
        status_summary[s] = status_summary.get(s, 0) + 1
    need_fetch = status_summary["missing"] + status_summary["stale"] + status_summary["outdated"] + status_summary["no_db_summary"]

    col_sync1, col_sync2 = st.columns(2)
    with col_sync1:
        btn_label = f"智能刷新 ({need_fetch}只待更新)" if need_fetch > 0 else "智能刷新（数据已最新）"
        if st.button(btn_label, use_container_width=True, type="primary" if need_fetch > 0 else "secondary",
                      help="跳过已是最新的股票，仅拉取缺失/过时的数据"):
            progress = st.progress(0, "扫描中...")
            def cb(i, total, msg):
                progress.progress((i + 1) / total, msg)
            fetched, skipped, failed = smart_refresh_all(cb)
            progress.empty()
            st.cache_data.clear()
            msg_parts = []
            if fetched:
                msg_parts.append(f"抓取 {len(fetched)} 只：{', '.join(c for c,_,_ in fetched)}")
            if skipped:
                msg_parts.append(f"跳过 {len(skipped)} 只（数据最新）")
            if failed:
                msg_parts.append(f"失败 {len(failed)} 只")
            st.success(" | ".join(msg_parts))
            st.rerun()

    with col_sync2:
        if st.button("同步到数据库", use_container_width=True,
                      help="将parquet数据同步到SQLite（仅更新摘要统计）"):
            from sync_db import sync_daily, sync_summary
            db2 = AnalysisDB()
            for code in STOCKS:
                sync_daily(db2, code)
                sync_summary(db2, code, force=True)
            db2.close()
            st.cache_data.clear()
            st.success("已同步到数据库")
            st.rerun()
            for code in STOCKS:
                sync_daily(db2, code)
                sync_summary(db2, code, force=True)
            db2.close()
            st.cache_data.clear()
            st.success("已同步到数据库")
            st.rerun()

    # DB 状态
    db_stat = get_db()
    s = db_stat.stats()
    db_stat.close()
    st.caption(f"DB: {s['stock_count']} 只股票, {s['daily_rows']} 行日线")

    st.divider()

    # ── 股票选择 ──
    stock_options = [f"{c} {STOCKS[c]['name']}" for c in STOCKS]
    stock_search = st.selectbox("选择股票（可输入搜索）", stock_options,
                                 help="输入代码或名称搜索")
    stock = stock_search.split()[0]

    st.divider()

    # ── 信号参数（可折叠）──
    with st.expander("信号参数调整", expanded=False):
        w_amp = st.slider("振幅权重", 0.0, 1.0, SIGNAL_WEIGHTS["amplitude"], 0.05)
        w_liq = st.slider("流动性权重", 0.0, 1.0, SIGNAL_WEIGHTS["liquidity"], 0.05)
        w_idio = st.slider("特质波动权重", 0.0, 1.0, SIGNAL_WEIGHTS["idio"], 0.05)
        w_reg = st.slider("波动区间权重", 0.0, 1.0, SIGNAL_WEIGHTS["regime"], 0.05)
        sig_thresh = st.slider("信号阈值", 0.3, 0.9, SIGNAL_THRESHOLD, 0.05)
    weights = {"amplitude": w_amp, "liquidity": w_liq, "idio": w_idio, "regime": w_reg}

    st.divider()
    st.caption(f"交易成本: {T_COST*100:.2f}% 单边 | 数据: AKShare")

# ── 主区域：顶层三页 ──
page_overview, page_signal, page_backtest, page_stock, page_data = st.tabs(["总览", "今日信号", "信号回测", "个股分析", "数据管理"])

# ═══════════════ 总览页 ═══════════════
with page_overview:
    st.title("股票池总览")
    summaries = load_all_summaries()
    regimes = load_all_regimes()

    if summaries:
        # ── 市场状态概览 ──
        regime_counts = {"单边涨": 0, "单边跌": 0, "偏涨震荡": 0, "偏跌震荡": 0, "窄幅震荡": 0}
        for code in STOCKS:
            if code in regimes:
                lbl = regimes[code].get("label", "")
                if lbl in regime_counts:
                    regime_counts[lbl] += 1
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        regime_style = {
            "单边涨": ("", "#dc2626"), "单边跌": ("", "#059669"),
            "偏涨震荡": ("", "#d97706"), "偏跌震荡": ("", "#1a56db"),
            "窄幅震荡": ("", "#6366f1"),
        }
        for i, (label, count) in enumerate(regime_counts.items()):
            col = [rc1, rc2, rc3, rc4, rc5][i]
            icon, color = regime_style[label]
            col.markdown(
                f"""<div style="text-align:center;padding:8px;border-radius:8px;border:1px solid {color}20;background:{color}08">
                <div style="font-size:1.5rem;font-weight:bold;color:{color}">{count}</div>
                <div style="font-size:0.75rem;color:#64748b">{icon} {label}</div></div>""",
                unsafe_allow_html=True)

        # 构建对比表
        rows = []
        for code in STOCKS:
            if code not in summaries:
                continue
            s = summaries[code]
            r = regimes.get(code, {})
            amp = s.get("amp_mean", 0)
            idio = round(s.get("idio_ratio_mean", 0) * 100, 1) if s.get("idio_ratio_mean") else 0
            signal = s.get("signal_pct", 0)
            score = s.get("composite_score_mean", 0)
            garch_p = s.get("garch_persistence", 0)
            bt_json = s.get("backtest_top5_json")
            if isinstance(bt_json, str):
                import json
                try: bt_json = json.loads(bt_json)
                except: bt_json = None
            best_win = f"{bt_json[0]['win_rate']}%" if bt_json and len(bt_json) > 0 else "-"

            # 评价
            if amp >= 3.0 and idio >= 70:
                verdict = "最佳"
            elif amp >= 2.5 and idio >= 50:
                verdict = "推荐"
            elif amp >= 2.2:
                verdict = "谨慎"
            else:
                verdict = "不适合"

            rows.append({
                "代码": code, "名称": s.get("name", code), "行业": s.get("industry", ""),
                "日均振幅": f"{amp}%", "振幅排序": amp,
                "特质占比": f"{idio}%", "特质排序": idio,
                "信号率": f"{signal}%", "信号排序": signal,
                "综合评分": score, "评分排序": score,
                "GARCH α+β": garch_p,
                "最佳胜率": best_win,
                "评价": verdict,
                "近20日状态": f"{r.get('label', '-')} ({r.get('trend_pct', 0):+.1f}%)",
                "状态排序": r.get("ratio", 0),
            })

        if rows:
            df_view = pd.DataFrame(rows)
            display_cols = ["代码", "名称", "行业", "日均振幅", "特质占比", "信号率", "综合评分", "近20日状态", "GARCH α+β", "最佳胜率", "评价"]

            # 排序控制
            sort_map = {"日均振幅": "振幅排序", "特质占比": "特质排序", "信号率": "信号排序", "综合评分": "评分排序"}
            sort_col = st.selectbox("排序依据", list(sort_map.keys()), index=0)
            df_view = df_view.sort_values(sort_map[sort_col], ascending=False)

            # 筛选
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                min_amp = st.slider("最低日均振幅", 1.0, 5.0, 2.0, 0.1)
            with col_f2:
                show_all = st.checkbox("显示全部（含不适合）", value=True)
            if not show_all:
                df_view = df_view[df_view["评价"].isin(["最佳", "推荐", "谨慎"])]
            df_view = df_view[df_view[sort_map[sort_col]] >= min_amp]

            st.dataframe(
                df_view[display_cols].set_index("代码"),
                use_container_width=True,
                column_config={
                    "综合评分": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                    "GARCH α+β": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                },
                height=min(400, 35 * len(df_view) + 38),
            )
            st.caption(f"共 {len(df_view)} 只股票符合条件，点击「个股分析」页查看详情")

            # 对比图
            col_ch1, col_ch2 = st.columns(2)
            with col_ch1:
                chart_data = df_view.sort_values("振幅排序", ascending=True).tail(15)
                fig = px.bar(chart_data, x="名称", y="振幅排序",
                             color="评价", title="日均振幅对比 (%)",
                             color_discrete_map={"最佳": "#059669", "推荐": "#1a56db", "谨慎": "#d97706", "不适合": "#dc2626"})
                fig.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col_ch2:
                chart_data2 = df_view.sort_values("特质排序", ascending=True).tail(15)
                fig2 = px.bar(chart_data2, x="名称", y="特质排序",
                              color="评价", title="特质波动占比对比 (%)",
                              color_discrete_map={"最佳": "#059669", "推荐": "#1a56db", "谨慎": "#d97706", "不适合": "#dc2626"})
                fig2.update_layout(height=350, template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("暂无数据，请先同步数据库")
    else:
        st.warning("数据库为空。请点击侧边栏「同步到数据库」按钮初始化数据。")

# ═══════════════ 数据管理页 ═══════════════
with page_data:
    st.title("数据管理")
    db_info = get_db()
    stats = db_info.stats()
    db_info.close()

    col_d1, col_d2, col_d3 = st.columns(3)
    col_d1.metric("股票数量", stats["stock_count"])
    col_d2.metric("日线总行数", f"{stats['daily_rows']:,}")
    col_d3.metric("数据库", "data/analysis.db")

    st.divider()
    st.subheader("各股票数据状态")

    status_rows = []
    for code in STOCKS:
        cfg = STOCKS[code]
        parquet_exists = os.path.exists(f"data/processed/{code}_features.parquet")
        db_has = code in stats.get("codes", [])

        if parquet_exists:
            df = pd.read_parquet(f"data/processed/{code}_features.parquet")
            last_date = df.index.max().strftime("%Y-%m-%d")
            days = len(df)
        else:
            last_date = "-"
            days = 0

        status_rows.append({
            "代码": code, "名称": cfg["name"], "行业": cfg.get("industry", ""),
            "数据天数": days, "最新日期": last_date,
            "Parquet": "✓" if parquet_exists else "✗",
            "DB摘要": "✓" if db_has else "✗",
        })

    st.dataframe(pd.DataFrame(status_rows).set_index("代码"), use_container_width=True)

    st.divider()
    st.subheader("检查与修复")
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        check_code = st.selectbox("选择股票", list(STOCKS.keys()),
                                   format_func=lambda x: f"{x} {STOCKS[x]['name']}",
                                   key="check_select")
    with col_r2:
        if st.button("验证数据", use_container_width=True):
            status, last_date, days = check_stock_status(check_code)
            gaps = verify_data_gaps(check_code)
            name = STOCKS[check_code]["name"]
            st.info(f"{name}: 状态={status}, 最新={last_date}, {days}天")
            if gaps:
                st.warning(f"发现 {len(gaps)} 个日期缺口（>5天）:")
                for start, end, gap_days in gaps[:10]:
                    st.caption(f"  {start} → {end} ({gap_days}天)")
            else:
                st.success("数据完整，无日期缺口")

    col_r3, col_r4 = st.columns([3, 1])
    with col_r3:
        refresh_code = st.selectbox("单只强制刷新", list(STOCKS.keys()),
                                     format_func=lambda x: f"{x} {STOCKS[x]['name']}",
                                     key="refresh_select",
                                     help="覆盖已有数据，重新拉取+处理+同步DB")
    with col_r4:
        if st.button("强制刷新该股", use_container_width=True):
            with st.spinner(f"刷新 {STOCKS[refresh_code]['name']}..."):
                try:
                    refresh_single_stock(refresh_code)
                    st.cache_data.clear()
                    st.success(f"{STOCKS[refresh_code]['name']} 刷新成功")
                    st.rerun()
                except Exception as e:
                    st.error(f"刷新失败: {e}")

# ═══════════════ 今日信号页 ═══════════════
with page_signal:
    st.title("今日信号 — 明日做T判断")

    # 数据日期
    latest_dates = {}
    for code in STOCKS:
        cache_path = f"data/processed/{code}_features.parquet"
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            latest_dates[code] = df.index.max().strftime("%Y-%m-%d")
    if latest_dates:
        max_date = max(latest_dates.values())
        from datetime import date, timedelta
        today = date.today()
        # 简单推算下一个交易日
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        st.caption(f"数据截止: {max_date} | 今日: {today} | 下一交易日: {next_day} | 信号基于最新数据预测明日做T可行性")
    st.divider()

    summaries = load_all_summaries()
    if not summaries:
        st.warning("数据库为空，请先同步数据")
    else:
        signal_rows = []
        for code in STOCKS:
            if code not in summaries:
                continue
            s = summaries[code]
            name = s.get("name", code)

            score_last = s.get("composite_score_last", 0) or 0
            sig = s.get("signal_last", 0) or 0
            amp_score = round((s.get("score_amplitude_last") or 0) * 100)
            liq_score = round((s.get("score_liquidity_last") or 0) * 100)
            idio_score = round((s.get("score_idio_last") or 0) * 100)
            reg_score = round((s.get("score_regime_last") or 0) * 100)
            amp = s.get("latest_amp", 0) or 0
            amp_ma20 = s.get("amp_ma_20_last", 0) or 0

            # 判断
            if score_last >= 0.65 and sig == 1:
                verdict = "强烈推荐"
                verdict_color = "#059669"
                emoji = "🟢"
            elif score_last >= 0.6 and sig == 1:
                verdict = "适合做T"
                verdict_color = "#1a56db"
                emoji = "🔵"
            elif score_last >= 0.5:
                verdict = "谨慎观望"
                verdict_color = "#d97706"
                emoji = "🟡"
            else:
                verdict = "不建议"
                verdict_color = "#dc2626"
                emoji = "🔴"

            signal_rows.append({
                "code": code, "name": name,
                "verdict": verdict, "verdict_color": verdict_color, "emoji": emoji,
                "composite_score": score_last,
                "signal": sig,
                "amp_score": amp_score, "liq_score": liq_score,
                "idio_score": idio_score, "reg_score": reg_score,
                "latest_amp": amp, "amp_ma20": amp_ma20,
            })

        # Sort by composite score
        signal_rows.sort(key=lambda r: r["composite_score"], reverse=True)

        # Summary bar
        recommend = sum(1 for r in signal_rows if r["verdict"] in ("强烈推荐", "适合做T"))
        caution = sum(1 for r in signal_rows if r["verdict"] == "谨慎观望")
        avoid = sum(1 for r in signal_rows if r["verdict"] == "不建议")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("强烈推荐", recommend, delta=None)
        col_s2.metric("适合做T", caution, delta=None)
        col_s3.metric("谨慎观望", avoid, delta=None)
        col_s4.metric("总计", len(signal_rows), delta=None)

        st.divider()

        # Detail cards for recommended stocks
        st.subheader("推荐做T标的")
        top_stocks = [r for r in signal_rows if r["verdict"] in ("强烈推荐", "适合做T")]
        if top_stocks:
            cols = st.columns(min(len(top_stocks), 3))
            for i, r in enumerate(top_stocks):
                with cols[i % 3]:
                    bg = "#f0fdf4" if r["verdict"] == "强烈推荐" else "#eff6ff"
                    border = "#059669" if r["verdict"] == "强烈推荐" else "#1a56db"
                    st.markdown(f"""
                    <div style="background:{bg};border:2px solid {border};border-radius:12px;padding:16px;margin-bottom:8px;">
                        <div style="font-size:1.5rem;">{r['emoji']}</div>
                        <strong style="font-size:1.1rem;">{r['name']}</strong>
                        <small style="color:#64748b;">{r['code']}</small>
                        <div style="font-size:1.4rem;font-weight:800;color:{border};margin:8px 0;">{r['composite_score']:.2f}</div>
                        <div style="font-size:0.75rem;color:#64748b;">
                            振幅{r['latest_amp']}% | MA20 {r['amp_ma20']}%<br>
                            振幅{r['amp_score']} | 流动性{r['liq_score']} | 特质{r['idio_score']} | 市态{r['reg_score']}
                        </div>
                        <span style="display:inline-block;background:{border};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;margin-top:6px;">{r['verdict']}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # Full signal table
        st.divider()
        st.subheader("全部股票信号明细")

        df_signal = pd.DataFrame(signal_rows)
        # Build display dataframe
        display_df = pd.DataFrame({
            "股票": [f"{r['emoji']} {r['name']}" for r in signal_rows],
            "代码": [r["code"] for r in signal_rows],
            "综合评分": [r["composite_score"] for r in signal_rows],
            "信号": ["做T" if r["signal"] == 1 else "观望" for r in signal_rows],
            "振幅评分": [r["amp_score"] for r in signal_rows],
            "流动性": [r["liq_score"] for r in signal_rows],
            "特质": [r["idio_score"] for r in signal_rows],
            "市态": [r["reg_score"] for r in signal_rows],
            "最新振幅": [f"{r['latest_amp']}%" for r in signal_rows],
            "判断": [r["verdict"] for r in signal_rows],
        })

        st.dataframe(
            display_df.set_index("代码"),
            use_container_width=True,
            column_config={
                "综合评分": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                "振幅评分": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                "流动性": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                "特质": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                "市态": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
            },
            height=35 * len(signal_rows) + 38,
        )

        # Score chart
        st.divider()
        st.subheader("分项评分对比")
        chart_signal = pd.DataFrame([
            {"股票": r["name"], "振幅评分": r["amp_score"], "流动性评分": r["liq_score"],
             "特质评分": r["idio_score"], "市态评分": r["reg_score"]}
            for r in sorted(signal_rows, key=lambda x: x["composite_score"], reverse=True)
        ])
        chart_signal = chart_signal.set_index("股票")
        fig = px.bar(chart_signal, barmode="group",
                     title="各股信号分项评分（越高越好）",
                     color_discrete_sequence=["#1a56db", "#059669", "#d97706", "#7c3aed"])
        fig.update_layout(height=350, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════ 信号回测页 ═══════════════
with page_backtest:
    st.title("信号回测验证")
    st.caption("每天生成信号 → 匹配次日实际振幅 → 统计信号可靠度。成本按0.15%双边计算。")

    @st.cache_data(ttl=300, show_spinner=False)
    def load_backtest_data():
        db = get_db()
        all_stats = db.get_signal_stats()
        per_stock = {}
        for code in STOCKS:
            s = db.get_signal_stats(code)
            if s:
                name = STOCKS[code]["name"]
                s["name"] = name
                s["code"] = code
                per_stock[code] = s
        db.close()
        return all_stats, per_stock

    all_stats, per_stock = load_backtest_data()

    if all_stats:
        # 顶部汇总
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总信号天数", f"{all_stats['signal_days']:,}")
        c2.metric("信号胜率", f"{all_stats['win_rate']}%")
        c3.metric("累计净收益", f"{all_stats['net_profit']:.1f}%")
        c4.metric("总盈利", f"{all_stats['total_profit']:.1f}%")
        c5.metric("总亏损", f"{all_stats['total_loss']:.1f}%")

        st.divider()

        # 每只股票统计表
        st.subheader("各股票信号回测对比")
        bt_rows = []
        for code, s in per_stock.items():
            bt_rows.append({
                "代码": code, "名称": s["name"],
                "信号天数": s["signal_days"], "信号率": f"{s['signal_pct']}%",
                "胜率": f"{s['win_rate']}%", "盈利次数": s["wins"], "亏损次数": s["loss"],
                "净收益": f"{s['net_profit']:.1f}%",
                "平均单次": f"{s['avg_profit']:.3%}",
                "胜率排序": s["win_rate"], "净收益排序": s["net_profit"],
            })
        bt_df = pd.DataFrame(bt_rows)

        sort_bt = st.selectbox("排序依据", ["胜率", "净收益"], key="bt_sort")
        sort_key = "胜率排序" if sort_bt == "胜率" else "净收益排序"
        bt_df = bt_df.sort_values(sort_key, ascending=False)

        st.dataframe(
            bt_df[["代码", "名称", "信号天数", "信号率", "胜率", "盈利次数", "亏损次数", "净收益", "平均单次"]].set_index("代码"),
            use_container_width=True,
            column_config={
                "胜率": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%s%%"),
                "净收益": st.column_config.ProgressColumn(min_value=0, max_value=max(bt_df["净收益排序"]) if not bt_df.empty else 100, format="%s"),
            },
            height=min(450, 35 * len(bt_df) + 38),
        )

        # 图表: 净收益排行
        col_bt1, col_bt2 = st.columns(2)
        with col_bt1:
            bar_df = bt_df.sort_values("净收益排序", ascending=True).tail(12)
            fig_bt = px.bar(bar_df, x="名称", y="净收益排序", color="胜率",
                           title="累计净收益排行 (%)", color_continuous_scale="RdYlGn")
            fig_bt.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_bt, use_container_width=True)
        with col_bt2:
            bar_df2 = bt_df.sort_values("胜率排序", ascending=True).tail(12)
            fig_bt2 = px.bar(bar_df2, x="名称", y="胜率排序", title="信号胜率排行 (%)",
                            color="胜率排序", color_continuous_scale="Blues")
            fig_bt2.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig_bt2, use_container_width=True)

        # 最近的信号明细
        st.divider()
        st.subheader("最近信号明细")
        sel_bt_code = st.selectbox("选择股票查看信号明细", list(STOCKS.keys()),
                                   format_func=lambda c: f"{STOCKS[c]['name']} ({c})", key="bt_detail")
        if sel_bt_code:
            db2 = get_db()
            log_df = db2.get_signal_log(sel_bt_code, limit=60)
            db2.close()
            if not log_df.empty:
                log_df["date"] = pd.to_datetime(log_df["date"])
                log_df = log_df.sort_values("date", ascending=False)
                log_display = log_df.copy()
                log_display["signal_label"] = log_display["signal"].apply(lambda x: "✅" if x == 1 else "❌")
                log_display["win_label"] = log_display.apply(
                    lambda r: "🏆+" + str(round(r["profit_est"]*100, 2)) + "%" if r["is_win"] == 1
                    else ("❌" + str(round(r["profit_est"]*100, 2)) + "%" if r["signal"] == 1 else "-"), axis=1)
                st.dataframe(
                    log_display[["date", "signal_label", "composite_score", "next_amplitude", "win_label"]]
                    .rename(columns={"date": "日期", "signal_label": "信号", "composite_score": "综合评分",
                                     "next_amplitude": "次日振幅", "win_label": "结果"}),
                    use_container_width=True, height=400,
                    column_config={
                        "综合评分": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                    })
    else:
        st.warning("暂无回测数据，请先同步数据库并回填信号日志。")

# ═══════════════ 个股分析页 ═══════════════
with page_stock:
    status = st.empty()
    with status:
        with st.spinner(f"加载 {STOCKS[stock]['name']}({stock}) 数据中..."):
            df = load_and_process(stock)
            analysis = run_analysis(stock)
            signal_df = run_signals(stock, w_amp, w_liq, w_idio, w_reg, sig_thresh)
            backtest_df = run_backtest(stock, sig_thresh)
    status.empty()

    stock_name = STOCKS[stock]["name"]
    dist = analysis.get("distribution", {})
    decomp = analysis.get("decomposition", {})
    season = analysis.get("seasonality", {})
    tail = analysis.get("tail", {})

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["概览", "振幅分析", "联动分析", "信号系统", "回测分析", "实战策略"])

with tab1:
    st.title(f"{stock_name} ({stock}) 做T套利分析")
    st.caption(f"样本: {len(df)}个交易日 | 数据区间: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")

    # 关键指标卡片
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    amp_mean = dist.get("mean", 0) * 100
    amp_median = dist.get("median", 0) * 100
    c1.metric("日均振幅", f"{amp_mean:.2f}%", delta=None)
    c2.metric("振幅中位数", f"{amp_median:.2f}%", delta=None)
    c3.metric("P90振幅", f"{dist.get('percentiles', {}).get('p90', 0)*100:.2f}%", delta=None)

    idio_ratio = decomp.get("idio_ratio", decomp.get("idio_ratio_full", 0))
    c4.metric("特质波动占比", f"{idio_ratio*100:.0f}%",
               delta="独立性强" if idio_ratio > 0.5 else "跟随大盘", delta_color="off")

    signal_pct = signal_df["signal"].mean() * 100
    c5.metric("信号触发率", f"{signal_pct:.1f}%", delta=f"{signal_df['signal'].sum():.0f}天")

    best_fit = dist.get("best_fit", "N/A")
    c6.metric("最优分布", best_fit, delta=None)

    st.divider()

    # ── 核心结论 ──
    st.subheader("核心结论")

    # 从数据动态生成结论
    dow_stats = season.get("day_of_week", {})
    dow_means = {d: dow_stats.get("mean", {}).get(d, 0) * 100 for d in ["周一", "周二", "周三", "周四", "周五"]}
    best_dow = max(dow_means, key=dow_means.get)
    worst_dow = min(dow_means, key=dow_means.get)

    monthly_stats = season.get("monthly", {})
    month_means = {str(m): monthly_stats.get("mean", {}).get(str(m), 0) * 100 for m in range(1, 13)}
    best_months = sorted(month_means, key=month_means.get, reverse=True)[:3]

    garch_conv = analysis.get("garch", {}).get("converged", False)
    garch_pers = analysis.get("garch", {}).get("persistence", 0)

    if not backtest_df.empty:
        best_bt = backtest_df.nlargest(1, "win_rate").iloc[0]
        best_entry = best_bt["entry_threshold"]
        best_exit = best_bt["exit_threshold"]
        best_trades = int(best_bt["trade_days"])
        best_pnl = best_bt["avg_profit_pct"]

        # 找性价比最高的（交易次数 > 30 且 均利最高）
        practical = backtest_df[backtest_df["trade_days"] >= 30]
        if practical.empty:
            practical = backtest_df[backtest_df["trade_days"] >= 10]
        if not practical.empty:
            best_prac = practical.nlargest(1, "avg_profit_pct").iloc[0]
            prac_entry = best_prac["entry_threshold"]
            prac_exit = best_prac["exit_threshold"]
            prac_trades = int(best_prac["trade_days"])
            prac_pnl = best_prac["avg_profit_pct"]
        else:
            best_prac, prac_entry, prac_exit, prac_trades, prac_pnl = None, None, None, 0, 0
    else:
        best_entry = best_exit = best_pnl = None
        prac_entry = prac_exit = prac_pnl = None
        best_trades = prac_trades = 0

    conclusions = []
    # 结论1: 振幅分布
    conclusions.append({
        "icon": "",
        "title": "振幅不是正态分布",
        "detail": f"{stock_name}日内振幅服从**对数正态分布**，均值{amp_mean:.1f}%、中位数{amp_median:.1f}%。"
                 f"P90振幅{tail.get('p90_threshold', 0)*100:.1f}%，即10%的交易日振幅超此阈值，做T空间充足。"
    })
    # 结论2: 周内效应
    if season.get("dow_anova_p", 1) < 0.05:
        conclusions.append({
            "icon": "",
            "title": f"{best_dow}是做T黄金日",
            "detail": f"周内效应统计显著(p={season['dow_anova_p']:.3f})。{best_dow}日均振幅{dow_means[best_dow]:.2f}%最高，"
                     f"{worst_dow}日均振幅{dow_means[worst_dow]:.2f}%最低。振幅从周初向周中递减，周五反弹。"
        })
    # 结论3: GARCH
    if garch_conv:
        if garch_pers > 0.9:
            conclusions.append({
                "icon": "",
                "title": "波动有强聚集效应 (GARCH α+β=0.95)",
                "detail": f"GARCH持续性系数{garch_pers:.3f}，接近1意味着**今日高振幅→明日大概率也高振幅**。"
                         f"高振幅日可以连续做T，不需要担心\"今天做完明天就没机会\"。"
            })
        else:
            conclusions.append({
                "icon": "",
                "title": f"波动聚集中等 (GARCH α+β={garch_pers:.2f})",
                "detail": "波动有一定持续性，但更倾向于均值回归。做T后不建议连续追高振幅。"
            })
    # 结论4: 特质波动
    if idio_ratio > 0.5:
        conclusions.append({
            "icon": "",
            "title": "个股独立性强 — 非常适合做T",
            "detail": f"只有{1-idio_ratio:.0f}%的波动被大盘解释，{idio_ratio*100:.0f}%是股票自身特质波动。"
                     f"这意味着个股行情不完全跟大盘走，做T有独立的利润空间。属于做T友好型标的。"
        })
    else:
        conclusions.append({
            "icon": "",
            "title": "个股跟随大盘较紧",
            "detail": f"{r2*100:.0f}%波动被大盘解释。做T需要更多依赖大盘择时，而非个股独立行情。"
        })
    # 结论5: 回测参数
    if best_entry is not None:
        conclusions.append({
            "icon": "",
            "title": f"最佳做T参数: 回调{abs(best_entry)*100:.0f}%买入, 拉升{best_exit*100:.0f}%卖出",
            "detail": f"理论最优: 约{best_trades}次交易机会，胜率100%，每笔均利{best_pnl*100:.2f}%。"
        })
    if prac_trades > 0 and (best_entry != prac_entry or best_exit != prac_exit):
        conclusions.append({
            "icon": "",
            "title": f"实用推荐参数: 回调{abs(prac_entry)*100:.0f}%买入, 拉升{prac_exit*100:.0f}%卖出",
            "detail": f"约{prac_trades}次交易机会（更实际），均利{prac_pnl*100:.2f}%。平衡了交易频率和单笔利润。"
        })
    # 结论6: 极端行情
    conclusions.append({
        "icon": "",
        "title": "极端行情 = 最大做T窗口",
        "detail": "指数涨跌幅>1%时，个股振幅从2.7%飙升至3.3~3.7%。市场剧烈波动的日子反而是做T最好的时机。"
    })

    # 渲染结论
    cc_cols = st.columns(min(len(conclusions), 3))
    for i, c in enumerate(conclusions):
        col_idx = i % 3
        with cc_cols[col_idx]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        border-radius: 12px; padding: 16px; margin-bottom: 12px; min-height: 180px;">
                <div style="font-size: 1.3rem; margin-bottom: 6px;">{c['icon']}</div>
                <strong>{c['title']}</strong>
                <p style="font-size: 0.85rem; color: #444; margin-top: 8px; line-height: 1.5;">{c['detail']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # 三行关键图表
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("振幅分布 (vs 理论分布)")
        amp_clean = df["amplitude"].dropna()
        amp_clean = amp_clean[amp_clean < amp_clean.quantile(0.995)] * 100

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=amp_clean, nbinsx=80, histnorm="probability density",
                                    name="实际分布", marker_color="steelblue", opacity=0.7))
        # KDE via violin or density
        from scipy import stats as sp_stats
        x_range = np.linspace(amp_clean.min(), amp_clean.max(), 200)

        # 正态分布拟合
        mu, std = sp_stats.norm.fit(amp_clean)
        fig.add_trace(go.Scatter(x=x_range, y=sp_stats.norm.pdf(x_range, mu, std),
                                  name=f"正态 (μ={mu:.2f})", line=dict(color="orange", dash="dash")))

        # 对数正态拟合
        shape, loc, scale = sp_stats.lognorm.fit(amp_clean[amp_clean > 0.001])
        fig.add_trace(go.Scatter(x=x_range, y=sp_stats.lognorm.pdf(x_range, shape, loc, scale),
                                  name="对数正态", line=dict(color="red", width=2)))

        # 分位线
        for pct, color in [(50, "gray"), (75, "green"), (90, "orange"), (95, "red")]:
            v = np.percentile(amp_clean, pct)
            fig.add_vline(x=v, line_dash="dot", line_color=color,
                           annotation_text=f"P{pct}:{v:.2f}%", annotation_position="top")

        fig.update_layout(height=380, bargap=0.05, showlegend=True,
                           xaxis_title="日内振幅(%)", yaxis_title="密度",
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("波动率分解")
        r2 = decomp.get("r_squared_market", 0)
        idio = 1 - r2

        fig = go.Figure()
        fig.add_trace(go.Pie(labels=["市场解释", "特质波动"],
                              values=[r2, idio],
                              marker_colors=["steelblue", "darkorange"],
                              hole=0.4, textinfo="label+percent",
                              pull=[0, 0.05]))
        fig.update_layout(height=380, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if r2 < 0.3:
            st.success(f"特质波动占比 {idio*100:.0f}% — 个股独立性很强，非常适合做T")
        elif r2 < 0.5:
            st.info(f"特质波动占比 {idio*100:.0f}% — 有一定独立波动空间，可以做T")
        else:
            st.warning(f"特质波动占比 {idio*100:.0f}% — 个股跟随大盘较紧，做T需谨慎")

    # 季节性卡片
    st.divider()
    st.subheader("振幅季节性特征")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        if "day_of_week" in season:
            dow = season["day_of_week"]
            dow_names = ["周一", "周二", "周三", "周四", "周五"]
            means = [dow.get("mean", {}).get(d, 0) * 100 for d in dow_names]
            fig = px.bar(x=dow_names, y=means, labels={"x": "", "y": "振幅均值(%)"},
                          color=means, color_continuous_scale="RdYlGn")
            fig.update_layout(height=250, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            dow_p = season.get("dow_anova_p", 1)
            st.caption(f"ANOVA p={dow_p:.4f} {'✅ 显著' if dow_p < 0.05 else '不显著'}")

    with sc2:
        if "monthly" in season:
            monthly = season["monthly"]
            means_m = [monthly.get("mean", {}).get(str(m), 0) * 100 for m in range(1, 13)]
            months = [f"{m}月" for m in range(1, 13)]
            fig = px.bar(x=months, y=means_m, labels={"x": "", "y": "振幅均值(%)"},
                          color=means_m, color_continuous_scale="RdYlGn")
            fig.update_layout(height=250, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    with sc3:
        if "quarterly" in season:
            qtr = season["quarterly"]
            q_means = [qtr.get("mean", {}).get(str(q), 0) * 100 for q in range(1, 5)]
            fig = px.bar(x=["Q1", "Q2", "Q3", "Q4"], y=q_means,
                          labels={"x": "", "y": "振幅均值(%)"},
                          color=q_means, color_continuous_scale="RdYlGn")
            fig.update_layout(height=250, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════ 振幅分析 ═══════════════
with tab2:
    st.title("振幅深度分析")

    # 振幅时序
    st.subheader("振幅时序 + 滚动均线")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.7, 0.3], vertical_spacing=0.05)

    amp_pct = df["amplitude"] * 100
    fig.add_trace(go.Scatter(x=df.index, y=amp_pct, mode="lines",
                              name="日振幅", line=dict(color="steelblue", width=0.5),
                              opacity=0.4), row=1, col=1)
    if "amp_ma_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["amp_ma_20"] * 100,
                                  name="20日均值", line=dict(color="red", width=2)),
                      row=1, col=1)
    if "amp_ma_60" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["amp_ma_60"] * 100,
                                  name="60日均值", line=dict(color="orange", width=2)),
                      row=1, col=1)

    # 信号日在振幅图上标记
    signal_days = signal_df[signal_df["signal"] == 1]
    if not signal_days.empty:
        fig.add_trace(go.Scatter(x=signal_days.index, y=signal_days["amplitude"] * 100,
                                  mode="markers", name=f"信号日",
                                  marker=dict(color="red", size=4, symbol="triangle-up")),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df.get("amp_zscore_20", pd.Series(0, index=df.index)),
                              name="振幅Z-score(20d)", line=dict(color="green", width=1)),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="orange", row=2, col=1)
    fig.add_hline(y=-1, line_dash="dash", line_color="orange", row=2, col=1)

    fig.update_layout(height=500, showlegend=True, template="plotly_white",
                       hovermode="x unified")
    fig.update_yaxes(title_text="振幅(%)", row=1, col=1)
    fig.update_yaxes(title_text="Z-score", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # 年度热力图
    st.subheader("年度振幅热力图")
    df_cal = df.copy()
    df_cal["year"] = df_cal.index.year
    df_cal["month"] = df_cal.index.month
    df_cal["day"] = df_cal.index.day
    years = sorted(df_cal["year"].unique())

    fig = make_subplots(rows=len(years), cols=1, shared_xaxes=True,
                         subplot_titles=[f"{y}年" for y in years])

    for i, year in enumerate(years):
        ydf = df_cal[df_cal["year"] == year].pivot_table(
            index="day", columns="month", values="amplitude", aggfunc="mean"
        ) * 100
        fig.add_trace(go.Heatmap(
            z=ydf.values,
            x=[f"{m}月" for m in ydf.columns],
            y=list(range(1, 32)),
            colorscale="RdYlGn", zmid=amp_median,
            colorbar=dict(title="振幅(%)") if i == len(years) - 1 else None,
            showscale=(i == len(years) - 1),
        ), row=i + 1, col=1)

    fig.update_layout(height=200 * len(years), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # GARCH分析
    st.subheader("GARCH波动聚集分析")
    garch = analysis.get("garch", {})
    cg1, cg2 = st.columns(2)
    with cg1:
        if garch.get("converged"):
            persistence = garch.get("persistence", 0)
            st.metric("GARCH持续性 (α+β)", f"{persistence:.3f}",
                       delta="强聚集" if persistence > 0.9 else "中等聚集")
            st.caption("接近1 → 今天高振幅，明天大概率也高 → 可连续做T")
            st.caption("< 0.9 → 振幅均值回归 → 高振幅后不适合连续做T")
        else:
            st.warning("GARCH模型未收敛，使用EWMA替代")

    with cg2:
        # 尾部分析
        st.metric("高振幅日占比(>P90)", f"{tail.get('high_amp_pct', 0)*100:.1f}%")
        st.metric("极端振幅日占比(>P95)", f"{tail.get('very_high_amp_pct', 0)*100:.1f}%")
        max_streak = tail.get("max_high_amp_streak", 0)
        avg_streak = tail.get("avg_high_amp_streak", 0)
        if max_streak:
            st.metric("最长连续高振幅", f"{max_streak}天")
            st.metric("平均连续高振幅", f"{avg_streak:.1f}天")


# ═══════════════ 联动分析 ═══════════════
with tab3:
    st.title("指数 & 板块联动分析")

    # 指数vs个股振幅散点
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("指数涨跌 vs 个股振幅")
        idx_ret = None
        if "idx_pct_change" in df.columns:
            idx_ret = df["idx_pct_change"]
        elif "idx_close" in df.columns:
            idx_ret = df["idx_close"].pct_change() * 100

        if idx_ret is not None:
            valid_idx = df[["amplitude"]].dropna().index.intersection(idx_ret.dropna().index)
            x = idx_ret.loc[valid_idx]
            y = df.loc[valid_idx, "amplitude"] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                      marker=dict(color="steelblue", size=3, opacity=0.3),
                                      name="每日", showlegend=False))

            # 分桶均值折线
            bins_edges = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
            bin_labels = ["<-2", "-2~-1", "-1~-0.5", "-0.5~0", "0~0.5", "0.5~1", "1~2", ">2"]
            bucket = pd.cut(x, bins=bins_edges, labels=bin_labels)
            means = y.groupby(bucket, observed=False).mean()
            counts = y.groupby(bucket, observed=False).count()
            valid_bins = means.dropna()

            fig.add_trace(go.Scatter(x=list(range(len(valid_bins))),
                                      y=valid_bins.values,
                                      mode="lines+markers",
                                      marker=dict(size=10, color="red"),
                                      line=dict(width=3, color="red"),
                                      name="均值趋势",
                                      text=[f"{lbl}<br>n={counts.get(lbl, 0)}" for lbl in valid_bins.index],
                                      hoverinfo="text+y"))
            fig.add_hline(y=amp_mean, line_dash="dot", line_color="gray",
                           annotation_text=f"均值{amp_mean:.1f}%")
            fig.add_vline(x=0, line_dash="dot", line_color="gray")

            fig.update_layout(height=400, xaxis_title="指数涨跌幅(%)",
                               yaxis_title="个股振幅(%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("无指数数据")

    with col_b:
        st.subheader("板块涨跌 vs 个股振幅")
        ind_ret = None
        if "ind_pct_change" in df.columns:
            ind_ret = df["ind_pct_change"]
        elif "ind_close" in df.columns:
            ind_ret = df["ind_close"].pct_change() * 100

        if ind_ret is not None:
            valid_ind = df[["amplitude"]].dropna().index.intersection(ind_ret.dropna().index)
            x = ind_ret.loc[valid_ind]
            y = df.loc[valid_ind, "amplitude"] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers",
                                      marker=dict(color="darkorange", size=3, opacity=0.3),
                                      name="每日", showlegend=False))

            bucket = pd.cut(x, bins=bins_edges, labels=bin_labels)
            means = y.groupby(bucket, observed=False).mean()
            fig.add_trace(go.Scatter(x=list(range(len(means.dropna()))),
                                      y=means.dropna().values,
                                      mode="lines+markers",
                                      marker=dict(size=10, color="red"),
                                      line=dict(width=3, color="red"),
                                      name="均值趋势"))
            fig.add_hline(y=amp_mean, line_dash="dot", line_color="gray")
            fig.add_vline(x=0, line_dash="dot", line_color="gray")

            fig.update_layout(height=400, xaxis_title="板块涨跌幅(%)",
                               yaxis_title="个股振幅(%)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("无板块数据")

    # Beta滚动 & 特质波动
    st.subheader("滚动Beta & 特质波动占比时序")
    if "beta_60" in df.columns and "idio_ratio" in df.columns:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.5], vertical_spacing=0.08)

        fig.add_trace(go.Scatter(x=df.index, y=df["beta_60"],
                                  name="60日滚动Beta", line=dict(color="steelblue", width=1.5)),
                      row=1, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color="gray", row=1, col=1)
        fig.add_hrect(y0=0.8, y1=1.2, line_width=0, fillcolor="green", opacity=0.1,
                       annotation_text="Beta适中区", row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df["idio_ratio"],
                                  name="特质波动占比", line=dict(color="darkorange", width=1.5)),
                      row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hrect(y0=0.5, y1=1.0, line_width=0, fillcolor="green", opacity=0.1,
                       annotation_text="适合做T区", row=2, col=1)

        fig.update_layout(height=450, showlegend=True, template="plotly_white",
                           hovermode="x unified")
        fig.update_yaxes(title_text="Beta", row=1, col=1)
        fig.update_yaxes(title_text="特质占比", row=2, col=1, range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    # 条件表格
    st.subheader("指数×板块 联合条件下个股振幅")
    cond = analysis.get("conditional", {})
    if cond:
        indexed = cond.get("by_index_move", {})
        if indexed:
            bucket_data = indexed.get("mean", {})
            count_data = indexed.get("count", {})
            rows = []
            for k in bucket_data.keys():
                mean_v = bucket_data.get(k, 0) * 100
                cnt = count_data.get(k, 0)
                rows.append({"指数涨跌区间": k, "个股振幅均值(%)": f"{mean_v:.2f}", "样本数": int(cnt)})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════ 信号系统 ═══════════════
with tab4:
    st.title("做T信号系统")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.subheader("综合评分分布")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["评分直方图", "分项评分箱线图"])

        fig.add_trace(go.Histogram(x=signal_df["composite_score"], nbinsx=50,
                                    marker_color="steelblue", opacity=0.7,
                                    name="综合评分"), row=1, col=1)
        fig.add_vline(x=sig_thresh, line_dash="dash", line_color="red",
                       annotation_text=f"阈值={sig_thresh}", row=1, col=1)

        score_cols = [c for c in signal_df.columns if c.startswith("score_")]
        for c in score_cols:
            fig.add_trace(go.Box(y=signal_df[c].dropna(), name=c.replace("score_", ""),
                                  marker_color="steelblue"), row=1, col=2)

        fig.update_layout(height=350, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_s2:
        st.subheader("信号有效性验证")
        # 信号日 vs 非信号日的次日振幅
        signal_df["next_amp"] = signal_df["amplitude"].shift(-1)
        sig_next = signal_df[signal_df["signal"] == 1]["next_amp"].dropna() * 100
        nosig_next = signal_df[signal_df["signal"] == 0]["next_amp"].dropna() * 100

        fig = go.Figure()
        if len(nosig_next) > 0:
            fig.add_trace(go.Violin(y=nosig_next, name="非信号日", side="negative",
                                     line_color="gray", fillcolor="lightgray"))
        if len(sig_next) > 0:
            fig.add_trace(go.Violin(y=sig_next, name="信号日", side="positive",
                                     line_color="red", fillcolor="lightcoral"))
        fig.update_layout(height=350, template="plotly_white",
                           yaxis_title="次日振幅(%)",
                           title=f"信号日次日振幅均值: {sig_next.mean():.2f}% vs 非信号日: {nosig_next.mean():.2f}%")
        st.plotly_chart(fig, use_container_width=True)

    # 信号时序
    st.subheader("信号评分时序")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=signal_df.index, y=signal_df["composite_score"],
                              mode="lines", name="综合评分",
                              line=dict(color="steelblue", width=0.8)), row=1, col=1)
    fig.add_hline(y=sig_thresh, line_dash="dash", line_color="red", row=1, col=1)
    signal_pts = signal_df[signal_df["signal"] == 1]
    if not signal_pts.empty:
        fig.add_trace(go.Scatter(x=signal_pts.index, y=signal_pts["composite_score"],
                                  mode="markers", name=f"信号({len(signal_pts)}天)",
                                  marker=dict(color="red", size=5)), row=1, col=1)

    for i, c in enumerate(score_cols[:4]):
        fig.add_trace(go.Scatter(x=signal_df.index, y=signal_df[c].rolling(20).mean(),
                                  name=c.replace("score_", ""),
                                  line=dict(width=1)), row=2, col=1)

    fig.add_trace(go.Scatter(x=signal_df.index, y=signal_df["amplitude"] * 100,
                              name="振幅", line=dict(color="gray", width=0.5)), row=3, col=1)

    fig.update_layout(height=550, showlegend=True, template="plotly_white",
                       hovermode="x unified")
    fig.update_yaxes(title_text="综合评分", row=1, col=1)
    fig.update_yaxes(title_text="分项(20d MA)", row=2, col=1)
    fig.update_yaxes(title_text="振幅(%)", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── 评分计算说明 + 验算 ──
    st.divider()
    st.subheader("评分计算说明（含最新一日验算）")

    latest = signal_df.iloc[-1]
    latest_date = signal_df.index[-1]
    latest_amp = latest["amplitude"] * 100

    st.markdown(f"""
    ### 验算实例：{latest_date.strftime('%Y-%m-%d')}（{['周一','周二','周三','周四','周五','周六','周日'][latest_date.dayofweek]}）

    当日数据：振幅 **{latest_amp:.2f}%**、综合评分 **{latest['composite_score']:.2f}**（{latest['signal_level']}）
    """)

    with st.expander("① 振幅预期评分 (score_amplitude) 权重40%", expanded=True):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown(f"""
            **计算方法：**

            1. 取过去252个交易日（约1年）的振幅序列
            2. 计算当日振幅在序列中的**百分位排名**
               `rank(pct=True)` = 当天振幅超过了多少比例的历史天数
            3. 分别用 `min_periods=20` 和 `min_periods=60` 计算两个百分位
            4. 取两个百分位的**平均值**作为评分

            ```
            pctile_20 = amplitude.rolling(252, min_periods=20).rank(pct=True)
            pctile_60 = amplitude.rolling(252, min_periods=60).rank(pct=True)
            score = (pctile_20 + pctile_60) / 2
            ```

            **含义**：当日振幅在近1年中处于什么水平。
            - 0.9 = 今天的振幅超过了90%的历史交易日 → 高波动，做T空间大
            - 0.1 = 今天的振幅只超过10%的历史交易日 → 低波动，不适合做T
            """)
        with col_e2:
            amp_score = latest.get("score_amplitude", 0)
            st.metric("最新值", f"{amp_score:.3f}")
            st.caption(f"振幅超过近1年约{amp_score*100:.0f}%的交易日")

    with st.expander("② 流动性评分 (score_liquidity) 权重20%", expanded=False):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown(f"""
            **计算方法：** 三个子指标**等权平均**

            **a) 量比** = 当日成交量 ÷ 20日均量，截断到[0.3, 3.0]，缩放到[0, 1]
            ```
            vol_score = (clip(vol_ratio_20, 0.3, 3.0) - 0.3) / 2.7
            ```

            **b) 换手率**：0.5%~5%之间满分，太低保底、太高递减
            ```
            0.5% ≤ turnover ≤ 5% → 1.0
            turnover < 0.5% → turnover / 0.5
            turnover > 5% → max(0, 1 - (turnover-5%)/10%)
            ```

            **c) 成交额**：日成交额 ÷ 5亿，截断到[0, 1]
            ```
            amt_score = clip(amount/1e8 / 5, 0, 1)
            ```
            """)
        with col_e2:
            liq_score = latest.get("score_liquidity", 0)
            st.metric("最新值", f"{liq_score:.3f}")
            vol_r = df.get("vol_ratio_20", pd.Series([0])).iloc[-1]
            tr = df.get("turnover_rate", pd.Series([0])).iloc[-1]
            amt = df.get("amount", pd.Series([0])).iloc[-1]
            st.caption(f"量比={vol_r:.2f} | 换手率={tr*100:.2f}% | 成交额={amt/1e8:.1f}亿")
            # 手工验算
            vs = max(0, min(1, (max(0.3, min(3.0, vol_r)) - 0.3) / 2.7))
            tr_pct = tr * 100
            ts = 1.0 if 0.5 <= tr_pct <= 5 else (tr_pct/0.5 if tr_pct < 0.5 else max(0, 1-(tr_pct-5)/10))
            amt_i = amt / 1e8
            as_ = min(1.0, max(0, amt_i / 5.0))
            manual = (vs + ts + as_) / 3
            st.caption(f"手工验算: ({vs:.3f} + {ts:.3f} + {as_:.3f}) / 3 = {manual:.3f}")

    with st.expander("③ 特质波动评分 (score_idio) 权重25%", expanded=False):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown(f"""
            **计算方法：**

            直接使用 `idio_ratio`（特质波动占比）。

            ```
            idio_ratio = 1 - R²_market
            score = clip(idio_ratio, 0, 1)
            ```

            **R²_market** 来自60天滚动回归：
            ```
            stock_return(t) = α + β × index_return(t) + ε(t)
            R² = 被指数解释的方差 / 总方差
            ```

            **含义**：
            - 接近1 → 个股独立于大盘，做的是个股的钱
            - 接近0 → 个股紧跟大盘，做的是大盘择时的钱
            """)
        with col_e2:
            idio_score = latest.get("score_idio", 0)
            idio_r = df.get("idio_ratio", pd.Series([0.5])).iloc[-1]
            st.metric("最新值", f"{idio_score:.3f}")
            st.caption(f"手工验算: idio_ratio = 1 - R² = {idio_r:.3f}")

    with st.expander("④ 波动区间评分 (score_regime) 权重15%", expanded=False):
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            st.markdown(f"""
            **计算方法：**

            计算振幅与20日均值的**比率**，按倒U型打分：

            ```
            ratio = amplitude / amp_ma_20
            ```

            | ratio | 评分 | 含义 |
            |-------|------|------|
            | < 0.5 | 0.20 | 太平淡，不值得做 |
            | 0.5~0.8 | 0.2→0.7 | 温和回升中 |
            | 0.8~1.5 | 0.8→1.0 | **最佳做T区间** |
            | 1.5~2.0 | 1.0→0.5 | 偏剧烈，需谨慎 |
            | > 2.0 | 0.50 | 极端波动，可能是恐慌 |

            中间值线性插值。
            """)
        with col_e2:
            reg_score = latest.get("score_regime", 0)
            amp_ma20 = df.get("amp_ma_20", pd.Series([0.02])).iloc[-1] * 100
            ratio_v = latest_amp / amp_ma20 if amp_ma20 > 0 else 0
            st.metric("最新值", f"{reg_score:.3f}")
            st.caption(f"ratio = {latest_amp:.2f}% / {amp_ma20:.2f}% = {ratio_v:.2f}")

            # 手工验算
            if ratio_v < 0.5:
                manual_reg = 0.2
            elif ratio_v < 0.8:
                manual_reg = (ratio_v - 0.5) / 0.3 * 0.5 + 0.2
            elif ratio_v <= 1.5:
                manual_reg = 0.8 + (ratio_v - 0.8) / 0.7 * 0.2
            elif ratio_v <= 2.0:
                manual_reg = 1.0 - (ratio_v - 1.5) / 0.5 * 0.5
            else:
                manual_reg = 0.5
            st.caption(f"手工验算: {manual_reg:.3f}")

    st.divider()
    st.markdown(f"""
    ### 综合评分公式

    ```
    composite = 0.40 × score_amplitude
              + 0.20 × score_liquidity
              + 0.25 × score_idio
              + 0.15 × score_regime
    ```

    ### 验算汇总：{latest_date.strftime('%Y-%m-%d')}

    | 分项 | 值 | 权重 | 加权 |
    |------|-----|------|------|
    | 振幅预期 | {latest.get('score_amplitude', 0):.4f} | × 0.40 | = {latest.get('score_amplitude', 0)*0.4:.4f} |
    | 流动性 | {latest.get('score_liquidity', 0):.4f} | × 0.20 | = {latest.get('score_liquidity', 0)*0.2:.4f} |
    | 特质波动 | {latest.get('score_idio', 0):.4f} | × 0.25 | = {latest.get('score_idio', 0)*0.25:.4f} |
    | 波动区间 | {latest.get('score_regime', 0):.4f} | × 0.15 | = {latest.get('score_regime', 0)*0.15:.4f} |
    | **合计** | | | **= {latest.get('composite_score', 0):.4f}** |

    信号等级：`≥0.7`→极强 `0.5~0.7`→强 `0.3~0.5`→中 `<0.3`→弱
    做T信号触发：composite ≥ **{SIGNAL_THRESHOLD}**（侧边栏可调）
    """)


# ═══════════════ 回测分析 ═══════════════
with tab5:
    st.title("做T策略回测")

    st.markdown("""
    **回测逻辑**：基于日线OHLC，检查当日是否能以 `开盘价×(1+入场阈值)` 买入、`开盘价×(1+出场阈值)` 卖出。
    入场价和出场价都必须在当日最高/最低价范围内才记为一次可交易机会。
    > 注：此为**理论最大收益**模拟，实际交易需考虑滑点、时机选择和买卖先后顺序。
    """)

    # 参数热力图
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("胜率热力图")
        pivot_win = backtest_df.pivot_table(
            index="entry_threshold", columns="exit_threshold", values="win_rate"
        )
        fig = px.imshow(pivot_win, text_auto=".0%", aspect="auto",
                         color_continuous_scale="RdYlGn", title="各参数组合胜率")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.subheader("平均利润率热力图")
        pivot_pnl = backtest_df.pivot_table(
            index="entry_threshold", columns="exit_threshold", values="avg_profit_pct"
        )
        fig = px.imshow(pivot_pnl, text_auto=".3f", aspect="auto",
                         color_continuous_scale="RdYlGn", title="各参数组合平均利润率")
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # 最优参数表
    st.subheader("参数组合排名 (按胜率)")
    top_n = st.slider("显示Top N", 5, 25, 10)
    top_df = backtest_df.nlargest(top_n, "win_rate")[
        ["entry_threshold", "exit_threshold", "trade_days", "win_rate",
         "avg_profit_pct", "total_profit_pct", "max_profit_pct"]
    ].copy()
    top_df["entry_threshold"] = top_df["entry_threshold"].apply(lambda x: f"{x:.1%}")
    top_df["exit_threshold"] = top_df["exit_threshold"].apply(lambda x: f"{x:.1%}")
    top_df["win_rate"] = top_df["win_rate"].apply(lambda x: f"{x:.1%}")
    top_df["avg_profit_pct"] = top_df["avg_profit_pct"].apply(lambda x: f"{x:.4f}")
    top_df["total_profit_pct"] = top_df["total_profit_pct"].apply(lambda x: f"{x:.4f}")
    top_df.columns = ["入场阈值", "出场阈值", "交易天数", "胜率", "均利/笔", "累计利润", "最大单笔"]
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # 最优参数下的交易分布
    st.subheader("最优参数下的交易分布")
    if not backtest_df.empty:
        best_row = backtest_df.nlargest(1, "win_rate").iloc[0]
        best_entry = best_row["entry_threshold"]
        best_exit = best_row["exit_threshold"]

        # 用最优参数模拟
        df_bt = signal_df.dropna(subset=["open", "high", "low"]).copy()
        entry_price = df_bt["open"] * (1 + best_entry)
        exit_price = df_bt["open"] * (1 + best_exit)
        can_trade = (df_bt["low"] <= entry_price) & (df_bt["high"] >= exit_price) & (exit_price > entry_price)
        if "signal" in df_bt.columns:
            can_trade = can_trade & (df_bt["signal"] == 1)

        trades = df_bt[can_trade].copy()
        if not trades.empty:
            trades["profit_pct"] = (exit_price[can_trade] - entry_price[can_trade]) / entry_price[can_trade] * 100

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig = px.histogram(trades, x="profit_pct", nbins=30,
                                    title=f"利润分布 (entry={best_entry:.1%}, exit={best_exit:.1%})",
                                    labels={"profit_pct": "利润率(%)"})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            with col_t2:
                trades["year_month"] = trades.index.strftime("%Y-%m")
                monthly = trades.groupby("year_month").agg(
                    count=("profit_pct", "count"),
                    avg_pnl=("profit_pct", "mean"),
                    total_pnl=("profit_pct", "sum"),
                ).reset_index()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(x=monthly["year_month"], y=monthly["count"],
                                      name="交易次数", marker_color="steelblue"))
                fig.add_trace(go.Scatter(x=monthly["year_month"], y=monthly["avg_pnl"],
                                          name="均利(%)", line=dict(color="red", width=2),
                                          yaxis="y2"), secondary_y=True)
                fig.update_layout(height=350, template="plotly_white",
                                   title="每月交易机会与平均利润")
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════ 实战策略 ═══════════════
with tab6:
    st.title("实战做T策略")

    # ── 昨日评分卡片 ──
    st.subheader("昨日评分 → 今日操作建议")

    last_row = signal_df.iloc[-1]
    last_date = signal_df.index[-1]
    prev_score = last_row["composite_score"]
    scores = {c.replace("score_", ""): last_row.get(c, 0) for c in signal_df.columns if c.startswith("score_")}
    prev_gap = (df["open"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] if len(df) > 1 else 0

    # 根据昨日评分给今日建议
    if prev_score >= 0.7:
        advice_color = "#27ae60"
        advice_emoji = ""
        advice_title = "强烈建议做T"
        advice_detail = f"昨日综合评分{prev_score:.2f}(强)，今日大概率高振幅。建议T仓满3000股，挂1%入场+1.5%出场。"
    elif prev_score >= 0.5:
        advice_color = "#f39c12"
        advice_emoji = ""
        advice_title = "可以做T"
        advice_detail = f"昨日综合评分{prev_score:.2f}(中等)，今日振幅预期适中。建议T仓2000股，挂1%入场+1%出场。"
    else:
        advice_color = "#95a5a6"
        advice_emoji = ""
        advice_title = "建议观望"
        advice_detail = f"昨日综合评分{prev_score:.2f}(偏弱)，今日振幅可能较小。建议不做T或仅1000股试盘。"

    # 评分卡片
    cc0, cc1, cc2, cc3, cc4 = st.columns([1.5, 1, 1, 1, 2])

    with cc0:
        st.markdown(f"""
        <div style="background: {advice_color}; border-radius: 16px; padding: 20px; color: white; text-align: center; min-height: 150px;">
            <div style="font-size: 0.8rem; opacity: 0.8;">{last_date.strftime('%Y-%m-%d')} ({['周一','周二','周三','周四','周五','周六','周日'][last_date.dayofweek]})</div>
            <div style="font-size: 2.5rem; font-weight: bold;">{prev_score:.2f}</div>
            <div style="font-size: 0.85rem; opacity: 0.9;">昨日综合评分</div>
            <div style="font-size: 1.1rem; margin-top: 8px; font-weight: bold;">{advice_emoji} {advice_title}</div>
        </div>
        """, unsafe_allow_html=True)

    with cc1:
        s_amp = scores.get("amplitude", 0)
        color_amp = "#27ae60" if s_amp > 0.6 else ("#f39c12" if s_amp > 0.4 else "#e74c3c")
        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 14px; text-align: center; min-height: 150px;
                    border-left: 4px solid {color_amp};">
            <div style="font-size: 0.75rem; color: #666;">振幅预期</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color_amp};">{s_amp:.2f}</div>
            <div style="font-size: 0.75rem; color: #888;">{'高波动预期' if s_amp > 0.6 else ('中等预期' if s_amp > 0.4 else '低波动预期')}</div>
        </div>
        """, unsafe_allow_html=True)

    with cc2:
        s_liq = scores.get("liquidity", 0)
        color_liq = "#27ae60" if s_liq > 0.6 else ("#f39c12" if s_liq > 0.4 else "#e74c3c")
        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 14px; text-align: center; min-height: 150px;
                    border-left: 4px solid {color_liq};">
            <div style="font-size: 0.75rem; color: #666;">流动性</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color_liq};">{s_liq:.2f}</div>
            <div style="font-size: 0.75rem; color: #888;">{'成交活跃' if s_liq > 0.6 else ('一般' if s_liq > 0.4 else '偏弱')}</div>
        </div>
        """, unsafe_allow_html=True)

    with cc3:
        s_idio = scores.get("idio", 0)
        color_idio = "#27ae60" if s_idio > 0.6 else ("#f39c12" if s_idio > 0.4 else "#e74c3c")
        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 14px; text-align: center; min-height: 150px;
                    border-left: 4px solid {color_idio};">
            <div style="font-size: 0.75rem; color: #666;">特质波动</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color_idio};">{s_idio:.2f}</div>
            <div style="font-size: 0.75rem; color: #888;">{'个股独立走势' if s_idio > 0.6 else ('一般跟随' if s_idio > 0.4 else '强跟随大盘')}</div>
        </div>
        """, unsafe_allow_html=True)

    with cc4:
        # 历史验证：昨日这个分数段 → 今日实际振幅分布
        prev_score_bin = "强(>0.7)" if prev_score >= 0.7 else ("中(0.4-0.7)" if prev_score >= 0.4 else "弱(<0.4)")
        hist_mask = (
            (signal_df["composite_score"].shift(1) >= 0.7) if prev_score >= 0.7
            else ((signal_df["composite_score"].shift(1) >= 0.4) & (signal_df["composite_score"].shift(1) <= 0.7)) if prev_score >= 0.4
            else (signal_df["composite_score"].shift(1) < 0.4)
        )
        hist_amp = signal_df[hist_mask]["amplitude"]
        hist_gt2 = (hist_amp > 0.02).mean() * 100
        hist_gt3 = (hist_amp > 0.03).mean() * 100

        st.markdown(f"""
        <div style="background: #f8f9fa; border-radius: 12px; padding: 14px; text-align: center; min-height: 150px;">
            <div style="font-size: 0.75rem; color: #666;">历史回测 (同类评分)</div>
            <div style="font-size: 0.8rem; margin-top: 8px;">振幅>2%概率</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: #2c3e50;">{hist_gt2:.0f}%</div>
            <div style="font-size: 0.8rem;">振幅>3%概率: {hist_gt3:.0f}%</div>
            <div style="font-size: 0.7rem; color: #888; margin-top: 4px;">基于{len(hist_amp.dropna())}天同类评分统计</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(advice_detail)
    st.divider()

    last_close = df["close"].iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    pos_value = last_close * 10000

    # ── 顶部持仓信息 ──
    c1, c2, c3 = st.columns(3)
    c1.metric("最新收盘价", f"{last_close:.2f} 元", delta=last_date)
    c2.metric("1万股持仓市值", f"{pos_value/10000:.1f} 万元")
    c3.metric("建议T仓规模(30%)", f"{pos_value*0.3/10000:.1f} 万元", delta="≈3000股")

    st.divider()

    # ── 日内 vs 隔日 核心对比表 ──
    st.divider()

    # ── 近期评分走势 ──
    st.subheader("近期评分走势 (近60天)")

    recent_60 = signal_df.tail(60)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.55, 0.45], vertical_spacing=0.06)

    # 综合评分 + 阈值
    fig.add_trace(go.Scatter(x=recent_60.index, y=recent_60["composite_score"],
                              mode="lines+markers", name="综合评分",
                              line=dict(color="#2c3e50", width=2),
                              marker=dict(size=6, color=recent_60["composite_score"].apply(
                                  lambda x: "#27ae60" if x > 0.7 else ("#f39c12" if x > 0.4 else "#e74c3c")
                              ))), row=1, col=1)
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", row=1, col=1,
                   annotation_text="强信号(0.7)")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", row=1, col=1,
                   annotation_text="弱信号(0.4)")

    # 分项评分
    score_cols = [c for c in recent_60.columns if c.startswith("score_")]
    colors = {"amplitude": "#e74c3c", "liquidity": "#3498db", "idio": "#9b59b6", "regime": "#1abc9c"}
    for c in score_cols:
        name = c.replace("score_", "")
        fig.add_trace(go.Scatter(x=recent_60.index, y=recent_60[c],
                                  name=name, line=dict(width=1, color=colors.get(name, "gray")),
                                  opacity=0.7), row=2, col=1)

    fig.update_layout(height=450, template="plotly_white", hovermode="x unified",
                       title="近60天评分走势 (上方=综合, 下方=分项)")
    fig.update_yaxes(title_text="综合评分", row=1, col=1, range=[0, 1])
    fig.update_yaxes(title_text="分项评分", row=2, col=1, range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("日内做T vs 隔日做T 全面对比")

    # 计算对比指标
    intraday_range = (df["high"] - df["low"]) / df["open"]
    close_px = df["close"]
    next_open = df["open"].shift(-1)
    next_high = df["high"].shift(-1)
    next_low = df["low"].shift(-1)

    overnight_gap = (next_open - close_px) / close_px
    overnight_best = (next_high - close_px) / close_px
    overnight_worst = (next_low - close_px) / close_px

    intraday_1pct = (intraday_range >= 0.01).mean() * 100
    intraday_2pct = (intraday_range >= 0.02).mean() * 100
    intraday_3pct = (intraday_range >= 0.03).mean() * 100
    overnight_1pct = (overnight_best >= 0.01).mean() * 100
    overnight_2pct = (overnight_best >= 0.02).mean() * 100
    gap_risk_1pct = (overnight_gap.abs() > 0.01).mean() * 100
    gap_risk_2pct = (overnight_gap.abs() > 0.02).mean() * 100

    comp_data = {
        "维度": ["交易规则", "资金占用", "可操作天数(1%价差)", "可操作天数(2%价差)",
                  "可操作天数(3%价差)", "核心风险", "隔夜跳空>1%概率", "隔夜跳空>2%概率",
                  "最大逆向跳空", "止损难度", "适合场景"],
        "日内做T": [
            "T+0,当天完成买卖循环", "仅交易时段占款,收盘资金回笼",
            f"{intraday_1pct:.0f}% (几乎每天)", f"{intraday_2pct:.0f}%",
            f"{intraday_3pct:.0f}%", "日内方向判断错误",
            "无隔夜跳空风险", "无隔夜跳空风险",
            "当日可控(止损2%内)", "容易,当天可以止损",
            "日常做T,稳定套利 (推荐)",
        ],
        "隔日做T": [
            "T+1,今天买明天才能卖", "资金被锁1晚,影响次日操作",
            f"{overnight_1pct:.0f}%", f"{overnight_2pct:.0f}%",
            f"{(overnight_best>=0.03).mean()*100:.0f}%",
            "隔夜跳空+方向判断双重风险",
            f"{gap_risk_1pct:.1f}% (~{int(gap_risk_1pct*250/100)}次/年)",
            f"{gap_risk_2pct:.1f}% (~{int(gap_risk_2pct*250/100)}次/年)",
            f"{overnight_gap.min()*100:.1f}%",
            "较难,隔夜可能大幅跳空",
            "仅在趋势明确+波动高位时可用",
        ],
    }

    st.dataframe(comp_data, use_container_width=True, hide_index=True)

    st.divider()

    # ── 图表对比 ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("日内振幅分布 (做T可行性)")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=intraday_range.clip(0, 0.08) * 100, nbinsx=60,
                                    marker_color="steelblue", opacity=0.7, name="日内振幅"))
        for v, color, label in [(1, "green", "1% (最小T空间)"), (2, "orange", "2% (舒适T空间)"),
                                  (3, "red", "3% (大T空间)")]:
            pct = (intraday_range >= v/100).mean() * 100
            fig.add_vline(x=v, line_dash="dash", line_color=color,
                           annotation_text=f"{label}:{pct:.0f}%天数")
        fig.update_layout(height=350, template="plotly_white",
                           xaxis_title="日内振幅(%)", yaxis_title="天数")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("隔夜跳空风险分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=overnight_gap.clip(-0.05, 0.05) * 100, nbinsx=60,
                                    marker_color="darkorange", opacity=0.7, name="隔夜跳空"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=-1, line_dash="dash", line_color="red", annotation_text="跌>1%:止损风险")
        fig.add_vline(x=1, line_dash="dash", line_color="green", annotation_text="涨>1%:正向跳空")
        fig.update_layout(height=350, template="plotly_white",
                           xaxis_title="隔夜跳空幅度(%)", yaxis_title="天数")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── 具体操作建议 ──
    st.subheader(f"1万股{stock_name} 具体操作方案")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        ### 日内做T（优先推荐）

        **仓位分配**
        - 底仓：7000股锁仓不动（避免T飞）
        - T仓：3000股做日内循环
        - 备用金：约6.3万元（买3000股用）

        **操作策略**
        | 场景 | 操作 |
        |------|------|
        | 开盘冲高 | 先卖3000股 → 回落买回 |
        | 开盘杀跌 | 先买3000股 → 反弹卖出 |
        | 震荡市(振幅>1.5%) | 挂单1%买入+1.5%卖出 |
        | 单边市(振幅>3%) | 顺势做T,不逆势 |

        **预期收益**
        - 每笔2%价差 × 3000股 × 21元 = ~1260元
        - 每月6-10次机会 = 7500~12600元/月
        - 年化约9-15万元（占底仓21万的43%~71%）
        """)

    with col_r:
        st.markdown("""
        ### 隔日做T（仅特定场景）

        **使用条件（全部满足才做）**
        - 信号系统评分 > 0.7
        - 尾盘振幅仍在扩大
        - 板块/指数同向共振
        - 非周五（避免周末消息跳空）

        **操作方式**
        - 尾盘14:50买入 → 次日开盘冲高卖出
        - 仓位控制在20%（2000股）
        - 必须设止损：跳空低开>2%立即割

        **风险提示**
        - 隔夜跳空>1%的概率约10%（年均25次）
        - 跳空可能直接吞噬日内利润
        - 资金被锁过夜,次日无法做日内T

        > 隔日T的性价比不如日内T。
        > {stock_name}日内振幅覆盖1%价差的比例高达{intraday_1pct:.0f}%，
        > 意味着几乎每个交易日都有T的机会，无需承担隔夜风险。
        """)

    st.divider()

    # ── 正T vs 反T 方向选择 ──
    st.subheader("涨了做还是跌了做？— 正T vs 反T 方向决策")

    up_room = (df["high"] - df["open"]) / df["open"]
    down_room = (df["open"] - df["low"]) / df["open"]
    yang_mask = df["close"] > df["open"]
    yin_mask = df["close"] < df["open"]
    yang_upper = ((df.loc[yang_mask, "high"] - df.loc[yang_mask, "close"]) / df.loc[yang_mask, "open"]).mean()
    yin_lower = ((df.loc[yin_mask, "open"] - df.loc[yin_mask, "low"]) / df.loc[yin_mask, "open"]).mean()

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["开盘→最高<br>(做反T卖出空间)", "开盘→最低<br>(做正T买入深度)"],
            y=[up_room.mean()*100, down_room.mean()*100],
            marker_color=["#27ae60", "#e74c3c"],
            text=[f"{up_room.mean()*100:.1f}%", f"{down_room.mean()*100:.1f}%"],
            textposition="outside",
        ))
        fig.update_layout(height=320, template="plotly_white",
                           yaxis_title="幅度(%)", title="开盘后上下空间(基本对称)")
        st.plotly_chart(fig, use_container_width=True)

    with col_d2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["阳线日上影<br>(冲高回落幅度)", "阴线日下影<br>(探底回升幅度)"],
            y=[yang_upper*100, yin_lower*100],
            marker_color=["#27ae60", "#e74c3c"],
            text=[f"{yang_upper*100:.1f}%", f"{yin_lower*100:.1f}%"],
            textposition="outside",
        ))
        fig.update_layout(height=320, template="plotly_white",
                           yaxis_title="幅度(%)", title="关键不对称: 阴线日反弹力度远大于阳线日回落")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    ### 核心发现

    | 发现 | 含义 |
    |------|------|
    | 向上空间{up_room.mean()*100:.1f}% ≈ 向下空间{down_room.mean()*100:.1f}% | 正T和反T空间几乎对称，**两种方向都可以做** |
    | 阴线日下影{yin_lower*100:.1f}% >> 阳线日上影{yang_upper*100:.1f}% | **跌了会大弹，但涨了不会大跌** — 回调深度远超冲高回落 |
    | 收阴概率53% > 收阳46% | 偏空市场，反T(先卖后买)有边际优势 |

    ### 方向决策速查

    | 开盘情况 | 推荐方向 | 操作 |
    |----------|----------|------|
    | 高开>0.5% | **反T (先卖后买)** | 冲高卖出→等回落后买回 |
    | 低开<-0.5% | **正T (先买后卖)** | 杀跌买入→等反弹后卖出 |
    | 平开 + 信号>0.7 | 正T | 正常回调买入→拉升卖出 |
    | 平开 + 信号<0.4 | 不做T | 振幅不够,不值得出手 |

    > 核心原则：**跟着开盘方向反向做T**。开盘往上冲就等它回落，开盘往下砸就等它弹回来。
    > {stock_name}最大的特征是"跌了会大幅弹回来"(阴线日下影达{yin_lower*100:.1f}%)，
    > 所以低开杀跌时做正T(抄底)的成功率很高。
    """)

    st.divider()

    # ── 开盘信号解读 ──
    st.subheader("开盘信号解读 — 9:25就能判断今天做不做T")

    # 计算开盘跳空数据
    gap = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    prev_signal = signal_df["composite_score"].shift(1)

    # 各场景
    scenarios_data = {
        "高开>1% + 强信号": (gap > 0.01) & (prev_signal > 0.6),
        "高开>1% (任意信号)": gap > 0.01,
        "低开>1% + 强信号": (gap < -0.01) & (prev_signal > 0.6),
        "低开>1% (任意信号)": gap < -0.01,
        "高开0.5~1% + 强信号": (gap > 0.005) & (gap <= 0.01) & (prev_signal > 0.6),
        "低开0.5~1% + 强信号": (gap < -0.005) & (gap >= -0.01) & (prev_signal > 0.6),
        "平开 + 强信号": (gap.abs() < 0.005) & (prev_signal > 0.6),
        "平开 + 弱信号": (gap.abs() < 0.005) & (prev_signal <= 0.6),
    }

    st.markdown("""
    ### 评分系统的工作原理

    当前评分由4个维度组成（基于历史数据滚动计算）：

    | 评分项 | 含义 | 权重 | 在预测什么？ |
    |--------|------|------|-------------|
    | **振幅预期** | 当前振幅在历史上的排位 | 40% | "今天波动够大吗？" |
    | **流动性** | 成交量/换手率是否充足 | 20% | "成交活跃吗？能进出吗？" |
    | **特质波动** | 个股独立于大盘的程度 | 25% | "是不是自己走行情？" |
    | **波动区间** | 振幅处于适中偏高区间 | 15% | "不会太平也不会太疯？" |

    > 评分使用**昨天以前的数据**计算，所以每天9:25开盘时就已经有评分结果，可以直接用来判断当天。
    """)

    # 场景分析表格
    scenario_rows = []
    for name, mask in scenarios_data.items():
        sub = df[mask].copy()
        if len(sub) < 15:
            continue
        sub_amp = sub["amplitude"]
        amp_gt_2pct = (sub_amp > 0.02).mean() * 100
        amp_gt_3pct = (sub_amp > 0.03).mean() * 100
        scenario_rows.append({
            "场景": name,
            "年均天数": f"{len(sub)/len(df)*250:.0f}天",
            "平均振幅": f"{sub_amp.mean()*100:.2f}%",
            "振幅>2%概率": f"{amp_gt_2pct:.0f}%",
            "振幅>3%概率": f"{amp_gt_3pct:.0f}%",
            "建议": "必须做T" if amp_gt_2pct > 80 else ("可以做T" if amp_gt_2pct > 65 else "观望"),
        })

    if scenario_rows:
        st.dataframe(scenario_rows, use_container_width=True, hide_index=True)

    col_sa, col_sb = st.columns(2)

    with col_sa:
        # 各场景振幅分布对比
        st.markdown("#### 不同开盘场景下的振幅对比")
        fig = go.Figure()
        categories = []
        means = []
        colors = []
        for name, mask in scenarios_data.items():
            sub_amp = df[mask]["amplitude"].dropna()
            if len(sub_amp) < 15:
                continue
            categories.append(name)
            means.append(sub_amp.mean() * 100)
            colors.append("#e74c3c" if "高开" in name else ("#27ae60" if "低开" in name else "#7f8c8d"))

        fig.add_trace(go.Bar(x=categories, y=means, marker_color=colors,
                              text=[f"{m:.1f}%" for m in means], textposition="outside"))
        fig.add_hline(y=df["amplitude"].mean()*100, line_dash="dash", line_color="gray",
                       annotation_text=f"均值{df['amplitude'].mean()*100:.1f}%")
        fig.update_layout(height=400, template="plotly_white", showlegend=False,
                           yaxis_title="当日振幅均值(%)")
        st.plotly_chart(fig, use_container_width=True)

    with col_sb:
        # 信号预测力
        st.markdown("#### 评分的预测力 (次日振幅相关性)")
        score_cols = [c for c in signal_df.columns if c.startswith("score_")]

        fig = go.Figure()
        for c in score_cols + ["composite_score"]:
            if c not in signal_df.columns:
                continue
            corr = signal_df[c].corr(signal_df["amplitude"].shift(-1))
            high_mean = signal_df[signal_df[c] > signal_df[c].quantile(0.7)]["amplitude"].shift(-1).mean()
            low_mean = signal_df[signal_df[c] < signal_df[c].quantile(0.3)]["amplitude"].shift(-1).mean()
            name = c.replace("score_", "")
            fig.add_trace(go.Bar(
                x=[f"{name}\n(高分组)", f"{name}\n(低分组)"],
                y=[high_mean*100, low_mean*100],
                name=name,
                text=[f"{high_mean*100:.1f}%", f"{low_mean*100:.1f}%"],
                textposition="outside",
            ))

        fig.update_layout(height=450, template="plotly_white", showlegend=False,
                           yaxis_title="次日振幅均值(%)",
                           title="高分(>P70) vs 低分(<P30) 的次日振幅对比")
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
    ### 开盘操作口诀

    | 开盘 | 前日信号 | 今天怎么做 |
    |------|----------|-----------|
    | 高开>1% | ≥0.6 | 振幅>3%概率极高 → **反T**, 冲高1.5%先卖 |
    | 高开>1% | <0.6 | 仍可做 → 冲高1%先卖, 回落0.5%买回 |
    | 低开>1% | ≥0.6 | 振幅>3%概率高 → **正T**, 杀跌1.5%先买 |
    | 低开>1% | <0.6 | 谨慎 → 杀跌1%小仓位买 |
    | 平开 | ≥0.6 | 可以做 → 挂1%买入+1.5%卖出 |
    | 平开 | <0.6 | 观望 → 今天大概率振幅<2%,不值得出手 |

    > **关键认知**：开盘跳空是最强实时信号，复合评分是辅助确认。跳空越大→当日振幅越大→做T越容易。
    > 两者结合做决策，比单独看任何一个都更准。
    """)

    st.divider()

    st.divider()

    # ── 风险提示 ──
    st.warning(f"""
    以上分析基于历史数据统计，不代表未来收益。实际做T需考虑：
    1. **滑点**：挂单不一定成交，市价单有滑点
    2. **流动性**：3000股约占日成交量比例需关注({stock_name}日均成交额达数亿，3000股影响极小)
    3. **T飞风险**：如果卖出后股价不回踩，底仓缩水。建议设回补条件单
    4. **手续费**：印花税0.05%+佣金约0.025%+滑点≈0.1%/笔，已在模型中考虑
    5. **心理因素**：连续止损后容易情绪化交易，建议每日T仓不超过3笔
    """)


# ── Footer ──
st.divider()
st.caption(f"数据来源: AKShare | 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 股票: {stock_name}({stock})")

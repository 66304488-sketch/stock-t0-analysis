"""
可视化：matplotlib静态图 + plotly交互图
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from src.utils import logger

FIG_DIR = "output/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def _save_or_show(fig, save_path=None):
    if save_path:
        path = os.path.join(FIG_DIR, save_path) if not save_path.startswith(FIG_DIR) else save_path
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"  图表保存: {path}")
    plt.close(fig)


# ============ 1. 振幅分布 ============

def plot_amplitude_distribution(df, dist_results=None, save_path=None):
    """振幅直方图 + KDE + 拟合分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    amp = df["amplitude"].dropna()
    amp = amp[amp < amp.quantile(0.995)]

    # 左：直方图+KDE
    ax = axes[0]
    ax.hist(amp * 100, bins=60, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    sns.kdeplot(amp * 100, ax=ax, color="darkred", linewidth=2, label="KDE")
    for pct_label, pct_val in [("P50", 50), ("P75", 75), ("P90", 90), ("P95", 95)]:
        v = np.percentile(amp * 100, pct_val)
        ax.axvline(v, linestyle="--", alpha=0.5, color="gray")
        ax.text(v, ax.get_ylim()[1] * 0.95, f"{pct_label}\n{v:.2f}%", ha="center", fontsize=8)
    ax.set_xlabel("日内振幅 (%)")
    ax.set_ylabel("密度")
    ax.set_title("日内振幅分布")
    ax.legend()

    # 右：Q-Q plot vs normal
    ax = axes[1]
    from scipy import stats
    stats.probplot(amp * 100, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (vs 正态分布)")
    ax.get_lines()[1].set_color("steelblue")

    fig.suptitle(f"振幅分布分析 (n={len(amp)})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path or "amplitude_distribution.png")
    return fig


def plot_amplitude_calendar_heatmap(df, save_path=None):
    """年度热力图：每格=一天的振幅"""
    df = df.copy()
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day

    years = sorted(df["year"].unique())
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(16, 3 * n_years))
    if n_years == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        ydf = df[df["year"] == year].pivot_table(
            index="day", columns="month", values="amplitude", aggfunc="mean"
        )
        sns.heatmap(ydf * 100, ax=ax, cmap="RdYlGn", center=2.5,
                     annot=False, cbar_kws={"label": "振幅(%)"})
        ax.set_title(f"{year}年 日均振幅热力图")
        ax.set_xlabel("月份")
        ax.set_ylabel("日")

    plt.tight_layout()
    _save_or_show(fig, save_path or "amplitude_calendar.png")
    return fig


# ============ 2. 时序图 ============

def plot_amplitude_timeseries(df, save_path=None):
    """振幅时序 + 滚动均值带"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1]})

    amp = df["amplitude"] * 100
    ax = axes[0]
    ax.plot(df.index, amp, alpha=0.3, color="steelblue", linewidth=0.5, label="日振幅")
    if "amp_ma_20" in df.columns:
        ax.plot(df.index, df["amp_ma_20"] * 100, color="darkred", linewidth=1.5, label="20日均值")
    if "amp_ma_60" in df.columns:
        ax.plot(df.index, df["amp_ma_60"] * 100, color="darkorange", linewidth=1.5, label="60日均值")
    ax.axhline(amp.median(), linestyle="--", color="gray", alpha=0.5, label=f"中位数({amp.median():.2f}%)")
    ax.set_ylabel("振幅 (%)")
    ax.set_title("日内振幅时序图")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 下：成交额
    ax = axes[1]
    if "amount" in df.columns:
        ax.fill_between(df.index, df["amount"] / 1e8, alpha=0.5, color="steelblue")
        ax.set_ylabel("成交额(亿)")
    ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or "amplitude_timeseries.png")
    return fig


# ============ 3. 条件振幅 ============

def plot_conditional_amplitude(df, save_path=None):
    """指数涨跌 vs 个股振幅 散点图 + 拟合线"""
    idx_ret = None
    if "idx_pct_change" in df.columns:
        idx_ret = df["idx_pct_change"]
    elif "idx_close" in df.columns:
        idx_ret = df["idx_close"].pct_change() * 100

    if idx_ret is None:
        logger.warning("无指数数据，跳过条件振幅图")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：散点
    ax = axes[0]
    valid = df[["amplitude"]].dropna().index.intersection(idx_ret.dropna().index)
    x = idx_ret.loc[valid]
    y = df.loc[valid, "amplitude"] * 100
    ax.scatter(x, y, alpha=0.3, s=3, color="steelblue")
    # lowess拟合
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y, x, frac=0.2)
        ax.plot(smooth[:, 0], smooth[:, 1], color="darkred", linewidth=2, label="LOWESS")
    except Exception:
        pass
    ax.set_xlabel("指数涨跌幅 (%)")
    ax.set_ylabel("个股振幅 (%)")
    ax.set_title("指数涨跌 vs 个股振幅")
    ax.axhline(y.mean(), linestyle="--", color="gray", alpha=0.5)
    ax.axvline(0, linestyle="--", color="gray", alpha=0.5)
    ax.legend()

    # 右：分桶箱线图
    ax = axes[1]
    bins = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    labels = ["<-2%", "-2~-1%", "-1~0%", "0~1%", "1~2%", ">2%"]
    bucket = pd.cut(x, bins=bins, labels=labels)
    box_data = [y[bucket == lbl].dropna().values for lbl in labels]
    ax.boxplot(box_data, labels=labels)
    ax.set_xlabel("指数涨跌区间")
    ax.set_ylabel("个股振幅 (%)")
    ax.set_title("不同指数涨跌区间下的个股振幅分布")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or "conditional_amplitude.png")
    return fig


# ============ 4. 波动分解 ============

def plot_vol_decomposition(decomp_results, save_path=None):
    """波动成分饼图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if "r_squared_market" in decomp_results:
        r2 = decomp_results["r_squared_market"]
        labels = [f"市场解释 ({r2:.1%})", f"特质波动 ({1-r2:.1%})"]
        sizes = [r2, 1 - r2]
        colors = ["steelblue", "darkorange"]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, explode=(0, 0.05))
        ax.set_title(f"波动率分解: 个股总方差 = 市场 + 特质\n(R²_mkt = {r2:.3f})")

    plt.tight_layout()
    _save_or_show(fig, save_path or "vol_decomposition.png")
    return fig


# ============ 5. 信号分析 ============

def plot_signal_distribution(df, save_path=None):
    """信号评分分布"""
    if "composite_score" not in df.columns:
        logger.warning("无信号评分列")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：综合评分直方图
    ax = axes[0]
    ax.hist(df["composite_score"].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.6, linestyle="--", color="darkred", linewidth=2, label="信号阈值(0.6)")
    ax.set_xlabel("综合做T评分")
    ax.set_ylabel("天数")
    ax.set_title("做T综合评分分布")
    ax.legend()

    # 右：各维度评分箱线图
    ax = axes[1]
    score_cols = [c for c in df.columns if c.startswith("score_")]
    if score_cols:
        scores = df[score_cols].dropna()
        ax.boxplot([scores[c].values for c in score_cols],
                    labels=[c.replace("score_", "") for c in score_cols])
        ax.set_ylabel("评分")
        ax.set_title("各维度评分分布")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or "signal_distribution.png")
    return fig


def plot_signal_timeseries(df, save_path=None):
    """信号时序图"""
    if "composite_score" not in df.columns:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                              gridspec_kw={"height_ratios": [2, 1, 1]},
                              sharex=True)

    # 上：综合评分时序
    ax = axes[0]
    ax.plot(df.index, df["composite_score"], alpha=0.6, color="steelblue", linewidth=0.5)
    ax.axhline(0.6, linestyle="--", color="darkred", alpha=0.5, label="信号阈值")
    signal_days = df[df["signal"] == 1] if "signal" in df.columns else pd.DataFrame()
    if not signal_days.empty:
        ax.scatter(signal_days.index, signal_days["composite_score"],
                    color="darkred", s=10, alpha=0.5, label=f"信号日({len(signal_days)}天)")
    ax.set_ylabel("综合评分")
    ax.set_title("做T综合评分时序图")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 中：分项评分
    ax = axes[1]
    score_cols = [c for c in df.columns if c.startswith("score_")]
    for c in score_cols:
        ax.plot(df.index, df[c].rolling(20).mean(), linewidth=1, alpha=0.7, label=c.replace("score_", ""))
    ax.set_ylabel("20日均值")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 下：振幅
    ax = axes[2]
    ax.fill_between(df.index, df["amplitude"] * 100, alpha=0.3, color="steelblue")
    ax.set_ylabel("振幅(%)")
    ax.set_xlabel("日期")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or "signal_timeseries.png")
    return fig


# ============ 6. 回测可视化 ============

def plot_backtest_results(grid_results, save_path=None):
    """回测参数热力图"""
    if grid_results is None or grid_results.empty:
        return None

    pivot_win = grid_results.pivot_table(
        index="entry_threshold", columns="exit_threshold", values="win_rate"
    )
    pivot_pnl = grid_results.pivot_table(
        index="entry_threshold", columns="exit_threshold", values="avg_profit_pct"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sns.heatmap(pivot_win, ax=ax, annot=True, fmt=".1%", cmap="RdYlGn",
                cbar_kws={"label": "胜率"})
    ax.set_title("参数组合胜率热力图")
    ax.set_xlabel("出场阈值")
    ax.set_ylabel("入场阈值")

    ax = axes[1]
    sns.heatmap(pivot_pnl, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                cbar_kws={"label": "均收益率"})
    ax.set_title("参数组合平均收益率热力图")
    ax.set_xlabel("出场阈值")
    ax.set_ylabel("入场阈值")

    plt.tight_layout()
    _save_or_show(fig, save_path or "backtest_heatmap.png")
    return fig


# ============ 7. 综合报告 ============

def generate_report(df, analysis_results, signal_df, backtest_df, stock_name="三一重工"):
    """生成完整的分析报告图表集"""
    logger.info("生成综合报告图表...")

    plot_amplitude_distribution(df, analysis_results.get("distribution"), "report_01_distribution.png")
    plot_amplitude_calendar_heatmap(df, "report_02_calendar.png")
    plot_amplitude_timeseries(df, "report_03_timeseries.png")
    plot_conditional_amplitude(df, "report_04_conditional.png")

    decomp = analysis_results.get("decomposition", {})
    if decomp:
        plot_vol_decomposition(decomp, "report_05_decomposition.png")

    if signal_df is not None and "composite_score" in signal_df.columns:
        plot_signal_distribution(signal_df, "report_06_signal_dist.png")
        plot_signal_timeseries(signal_df, "report_07_signal_ts.png")

    if backtest_df is not None and not backtest_df.empty:
        plot_backtest_results(backtest_df, "report_08_backtest.png")

    logger.info(f"综合报告完成, 图表保存至 {FIG_DIR}/")

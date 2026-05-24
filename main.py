"""
A股个股波动做T套利分析 - CLI入口
用法:
  python main.py fetch --stock 600031
  python main.py process --stock 600031
  python main.py analyze --stock 600031
  python main.py backtest --stock 600031
  python main.py report --stock 600031
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import STOCKS, DEFAULT_START_DATE, DEFAULT_END_DATE
from src.utils import logger


def _load_or_fetch(stock_code, stock_cfg):
    """优先从缓存加载，缓存不存在则获取数据"""
    from src.data_fetcher import StockDataFetcher
    fetcher = StockDataFetcher()
    return fetcher.fetch_all_for_stock(
        stock_code,
        start=DEFAULT_START_DATE,
        end=DEFAULT_END_DATE,
        market_index=stock_cfg["market_index"],
        industry=stock_cfg["industry"],
    )


def cmd_fetch(args):
    from src.data_fetcher import StockDataFetcher
    fetcher = StockDataFetcher()
    stock_cfg = STOCKS[args.stock]
    result = fetcher.fetch_all_for_stock(
        args.stock,
        start=args.start or DEFAULT_START_DATE,
        end=args.end or DEFAULT_END_DATE,
        market_index=stock_cfg["market_index"],
        industry=stock_cfg["industry"],
    )
    logger.info(f"数据获取完成: {args.stock} {stock_cfg['name']}")
    for k, v in result.items():
        if v is not None:
            logger.info(f"  {k}: {len(v)} 行")


def cmd_process(args):
    from src.data_cleaner import merge_all, standardize_daily_columns, standardize_index_columns
    from src.features import FeatureEngineer

    stock_cfg = STOCKS[args.stock]
    raw = _load_or_fetch(args.stock, stock_cfg)

    daily = standardize_daily_columns(raw["daily"])
    idx = standardize_index_columns(raw["index"], "idx")
    ind = standardize_index_columns(raw.get("industry"), "ind") if raw.get("industry") is not None else None

    merged = merge_all(daily, idx, ind)

    engineer = FeatureEngineer()
    featured = engineer.process(merged)

    out = f"data/processed/{args.stock}_features.parquet"
    os.makedirs("data/processed", exist_ok=True)
    featured.to_parquet(out)
    logger.info(f"特征工程完成, 保存至 {out}, shape={featured.shape}, 列数={len(featured.columns)}")


def cmd_analyze(args):
    import pandas as pd
    from src.analysis import AmplitudeAnalyzer

    path = f"data/processed/{args.stock}_features.parquet"
    if not os.path.exists(path):
        logger.error(f"特征文件不存在: {path}, 请先运行 process")
        return
    df = pd.read_parquet(path)
    analyzer = AmplitudeAnalyzer()
    results = analyzer.run_all(df)

    # 保存分析结果
    import json
    out = f"data/processed/{args.stock}_analysis.json"
    # 只保存标量结果
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {kk: vv for kk, vv in v.items()
                               if isinstance(vv, (int, float, str, bool, list, dict)) and not isinstance(vv, list)}
    with open(out, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"分析结果保存至 {out}")


def cmd_backtest(args):
    import pandas as pd
    from src.signals import TSignalGenerator
    from src.backtest import TBacktester

    path = f"data/processed/{args.stock}_features.parquet"
    if not os.path.exists(path):
        logger.error(f"特征文件不存在: {path}, 请先运行 process")
        return
    df = pd.read_parquet(path)

    # 先做信号
    sg = TSignalGenerator()
    signal_df = sg.generate_all_signals(df)

    # 回测
    bt = TBacktester()
    results = bt.run_grid(signal_df)
    logger.info(f"回测完成, 共{len(results)}组参数组合")

    if not results.empty:
        # 按胜率排
        results_sorted = results.sort_values("win_rate", ascending=False)
        top = results_sorted.head(10)
        logger.info("Top 10 参数组合 (按胜率):")
        for _, row in top.iterrows():
            logger.info(f"  entry={row['entry_threshold']:.3f} exit={row['exit_threshold']:.3f} "
                         f"win_rate={row['win_rate']:.2%} avg_profit_pct={row['avg_profit_pct']:.4f} "
                         f"trades={int(row['trade_days'])}")


def cmd_report(args):
    import pandas as pd
    from src.signals import TSignalGenerator
    from src.backtest import TBacktester
    from src.analysis import AmplitudeAnalyzer

    path = f"data/processed/{args.stock}_features.parquet"
    if not os.path.exists(path):
        logger.error(f"特征文件不存在: {path}, 请先运行 process")
        return

    df = pd.read_parquet(path)
    stock_name = STOCKS.get(args.stock, {}).get("name", args.stock)

    # 分析
    analyzer = AmplitudeAnalyzer()
    analysis_results = analyzer.run_all(df)

    # 信号
    sg = TSignalGenerator()
    signal_df = sg.generate_all_signals(df)

    # 回测
    bt = TBacktester()
    backtest_df = bt.run_grid(signal_df)

    # 可视化
    from src.visualize import generate_report
    generate_report(df, analysis_results, signal_df, backtest_df, stock_name=stock_name)

    logger.info(f"===== {stock_name}({args.stock}) 做T分析报告 =====")
    dist = analysis_results.get("distribution", {})
    if dist:
        logger.info(f"样本数: {dist.get('n', 'N/A')}")
        logger.info(f"振幅均值: {dist.get('mean', 0)*100:.2f}% / 中位数: {dist.get('median', 0)*100:.2f}%")
        logger.info(f"振幅P75: {dist['percentiles']['p75']*100:.2f}% P90: {dist['percentiles']['p90']*100:.2f}% P95: {dist['percentiles']['p95']*100:.2f}%")
        best_fit = dist.get('best_fit', 'N/A')
        if best_fit in dist.get('fits', {}):
            logger.info(f"最优分布: {best_fit} (AIC={dist['fits'][best_fit]['aic']:.1f})")

    decomp = analysis_results.get("decomposition", {})
    if decomp:
        logger.info(f"特质波动占比: {decomp.get('idio_ratio', 0)*100:.1f}% (R²_mkt={decomp.get('r_squared_market', 0):.3f})")

    season = analysis_results.get("seasonality", {})
    if "dow_anova_p" in season:
        logger.info(f"周内效应: ANOVA p={season['dow_anova_p']:.4f} {'(显著)' if season['dow_anova_p'] < 0.05 else '(不显著)'}")

    logger.info(f"信号触发率: {signal_df['signal'].mean()*100:.1f}% ({signal_df['signal'].sum():.0f}/{len(signal_df)}天)")

    if not backtest_df.empty:
        best = backtest_df.sort_values("win_rate", ascending=False).iloc[0]
        logger.info(f"最优回测参数: entry={best['entry_threshold']:.3f} exit={best['exit_threshold']:.3f}")
        logger.info(f"  交易天数: {int(best['trade_days'])} / {int(best.get('trade_pct', 0)*len(signal_df)):.0f}")
        logger.info(f"  胜率: {best['win_rate']:.2%}")
        logger.info(f"  平均利润率: {best['avg_profit_pct']:.4f}")
        logger.info(f"  总利润率: {best['total_profit_pct']:.4f}")

    logger.info(f"图表已保存至 output/figures/")


def main():
    parser = argparse.ArgumentParser(description="A股个股波动做T套利分析")
    sub = parser.add_subparsers(dest="command")

    p_fetch = sub.add_parser("fetch", help="获取数据")
    p_fetch.add_argument("--stock", default="600031")
    p_fetch.add_argument("--start")
    p_fetch.add_argument("--end")

    p_process = sub.add_parser("process", help="特征工程")
    p_process.add_argument("--stock", default="600031")

    p_analyze = sub.add_parser("analyze", help="统计分析")
    p_analyze.add_argument("--stock", default="600031")

    p_backtest = sub.add_parser("backtest", help="回测")
    p_backtest.add_argument("--stock", default="600031")

    p_report = sub.add_parser("report", help="生成完整报告")
    p_report.add_argument("--stock", default="600031")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    cmds = {"fetch": cmd_fetch, "process": cmd_process, "analyze": cmd_analyze,
            "backtest": cmd_backtest, "report": cmd_report}
    cmds[args.command](args)


if __name__ == "__main__":
    main()

"""
做T回测引擎：基于日线OHLC模拟日内T交易
"""
import itertools
import numpy as np
import pandas as pd
from src.utils import logger


class TBacktester:
    """做T策略回测器"""

    def __init__(self, cost=0.0015):
        """
        cost: 单边交易成本（含印花税0.05%、佣金、滑点）
        """
        self.cost = cost

    def run_grid(self, df, entry_thresholds=None, exit_thresholds=None):
        """
        网格搜索最优入场/出场阈值

        逻辑：对于每个交易日，检查是否能以 open*(1+entry) 买入、open*(1+exit) 卖出
        前提是 entry_pct 和 exit_pct 都在当天的高低点范围内
        """
        if entry_thresholds is None:
            entry_thresholds = [-0.03, -0.025, -0.02, -0.015, -0.01]
        if exit_thresholds is None:
            exit_thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]

        df = df.dropna(subset=["open", "high", "low"])
        results = []

        for entry_pct, exit_pct in itertools.product(entry_thresholds, exit_thresholds):
            metrics = self._simulate(df, entry_pct, exit_pct)
            metrics["entry_threshold"] = entry_pct
            metrics["exit_threshold"] = exit_pct
            results.append(metrics)

        result_df = pd.DataFrame(results)
        return result_df

    def _simulate(self, df, entry_pct, exit_pct):
        """
        做T模拟（先买后卖路径）

        逻辑：
        - 当天以 open_price 为基准
        - entry_price = open * (1 + entry_pct)  （entry_pct为负表示等待回调买入）
        - exit_price = open * (1 + exit_pct)    （exit_pct为正表示等待拉升卖出）
        - 如果 low <= entry_price 且 high >= exit_price 且 entry_price < exit_price
          则日内T交易可行
        - 每股收益 = exit_price - entry_price - 2*cost (双边成本)
        - 收益率 = 收益 / entry_price
        """
        # 计算理论买入价和卖出价
        entry_price = df["open"] * (1 + entry_pct)
        exit_price = df["open"] * (1 + exit_pct)

        # 条件：entry和exit必须都在当天价格范围内
        can_buy = df["low"] <= entry_price
        can_sell = df["high"] >= exit_price
        valid_spread = exit_price > entry_price  # 先买后卖
        can_trade = can_buy & can_sell & valid_spread

        # 计算实际交易利润
        profit_per_share = exit_price - entry_price
        cost_per_share = (entry_price + exit_price) * self.cost / 2
        net_profit = profit_per_share - cost_per_share * 2

        # 按信号过滤
        if "signal" in df.columns:
            can_trade = can_trade & (df["signal"] == 1)

        # 指标
        trade_days = can_trade.sum()
        total_days = len(df)

        return {
            "trade_days": int(trade_days),
            "trade_pct": float(trade_days / total_days) if total_days > 0 else 0,
            "win_days": int((net_profit[can_trade] > 0).sum()),
            "win_rate": float((net_profit[can_trade] > 0).mean()) if trade_days > 0 else 0,
            "avg_profit_per_share": float(net_profit[can_trade].mean()) if trade_days > 0 else 0,
            "avg_profit_pct": float((net_profit[can_trade] / entry_price[can_trade]).mean()) if trade_days > 0 else 0,
            "total_profit_pct": float((net_profit[can_trade] / entry_price[can_trade]).sum()) if trade_days > 0 else 0,
            "max_profit_pct": float((net_profit[can_trade] / entry_price[can_trade]).max()) if trade_days > 0 else 0,
            "min_profit_pct": float((net_profit[can_trade] / entry_price[can_trade]).min()) if trade_days > 0 else 0,
        }

    def run_signal_filtered(self, df):
        """仅对信号日做回测，用最佳参数"""
        if "signal" not in df.columns:
            logger.error("无信号列，请先运行 TSignalGenerator")
            return None

        # 先跑网格找最优参数
        grid = self.run_grid(df[df["signal"] == 0])  # 用非信号日找最优（避免过拟合）
        if grid.empty:
            return None

        best = grid.nlargest(1, "win_rate").iloc[0] if len(grid) > 0 else None
        if best is None:
            return None

        entry = best["entry_threshold"]
        exit_ = best["exit_threshold"]

        # 在信号日上测试
        signal_df = df[df["signal"] == 1]
        if signal_df.empty:
            return None

        metrics = self._simulate(signal_df, entry, exit_)
        metrics["best_entry"] = entry
        metrics["best_exit"] = exit_
        return metrics

    def daily_trade_log(self, df, entry_pct, exit_pct):
        """生成每日交易日志"""
        entry_price = df["open"] * (1 + entry_pct)
        exit_price = df["open"] * (1 + exit_pct)
        can_trade = (df["low"] <= entry_price) & (df["high"] >= exit_price) & (exit_price > entry_price)

        log = df[can_trade].copy()
        log["entry_price"] = entry_price[can_trade]
        log["exit_price"] = exit_price[can_trade]
        log["profit_pct"] = (exit_price[can_trade] - entry_price[can_trade]) / entry_price[can_trade]
        log["net_profit_pct"] = log["profit_pct"] - self.cost
        log["trade_result"] = log["net_profit_pct"].apply(
            lambda x: "盈利" if x > 0 else ("持平" if abs(x) < 0.001 else "亏损")
        )

        return log[["open", "high", "low", "close", "entry_price", "exit_price",
                      "profit_pct", "net_profit_pct", "trade_result", "amplitude"]]

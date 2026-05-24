"""
特征工程：从OHLCV数据构建做T相关的特征矩阵
"""
import numpy as np
import pandas as pd
from statsmodels.api import add_constant, OLS
from src.utils import logger


class FeatureEngineer:
    """做T特征工程器"""

    def __init__(self, rolling_windows=(5, 10, 20, 60)):
        self.windows = rolling_windows

    def process(self, merged_df):
        """主入口：依次计算所有特征类别"""
        df = merged_df.copy()
        logger.info(f"开始特征工程, 输入shape={df.shape}")

        df = self._amplitude_features(df)
        df = self._volume_liquidity_features(df)
        df = self._relative_volatility_features(df)
        df = self._calendar_features(df)
        df = self._target_features(df)

        # 移除前60行（rolling window需要预热）
        df = df.iloc[60:].copy()
        logger.info(f"特征工程完成, 输出shape={df.shape}, 特征数={len(df.columns)}")
        return df

    # ---- 1. 振幅特征 ----
    def _amplitude_features(self, df):
        """日内振幅的各类统计特征"""
        # 确保振幅列存在（百分比形式）
        if "amplitude" in df.columns:
            amp = df["amplitude"] / 100.0  # 转为小数
        else:
            prev_close = df["close"].shift(1)
            amp = (df["high"] - df["low"]) / prev_close.abs()

        df["amplitude"] = amp

        for w in self.windows:
            df[f"amp_ma_{w}"] = amp.rolling(w).mean()
            df[f"amp_std_{w}"] = amp.rolling(w).std()
            df[f"amp_max_{w}"] = amp.rolling(w).max()

        # 振幅zscore和百分位
        df["amp_zscore_20"] = (amp - df["amp_ma_20"]) / df["amp_std_20"].replace(0, np.nan)
        df["amp_ratio_vs_20d"] = amp / df["amp_ma_20"].replace(0, np.nan)

        # 振幅百分位（当前振幅在过去N天中的排位）
        for w in [20, 60]:
            df[f"amp_pctile_{w}"] = amp.rolling(w).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x), raw=False
            )

        logger.info(f"  振幅特征: {len([c for c in df.columns if c.startswith('amp_')])}个")
        return df

    # ---- 2. 成交量和流动性特征 ----
    def _volume_liquidity_features(self, df):
        if "volume" not in df.columns:
            return df

        vol = df["volume"]
        for w in self.windows:
            vol_ma = vol.rolling(w).mean()
            df[f"vol_ratio_{w}"] = vol / vol_ma.replace(0, np.nan)

        df["vol_zscore_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std().replace(0, np.nan)

        # 量-振幅互动：高量+高振幅 = 有意义的波动
        if "amplitude" in df.columns:
            df["vol_amp_score"] = df["amp_zscore_20"].fillna(0) * df["vol_zscore_20"].fillna(0)
            df["vol_amp_score"] = df["vol_amp_score"].clip(-3, 3)

        # 换手率
        if "turnover_rate" in df.columns:
            tr = df["turnover_rate"]
            for w in self.windows:
                df[f"turnover_ma_{w}"] = tr.rolling(w).mean()
            df["turnover_zscore"] = (tr - tr.rolling(20).mean()) / tr.rolling(20).std().replace(0, np.nan)

        # 成交额（金额）
        if "amount" in df.columns:
            amt = df["amount"]
            for w in [5, 20]:
                df[f"amount_ma_{w}"] = amt.rolling(w).mean()

        logger.info(f"  量流动性特征: {len([c for c in df.columns if c.startswith(('vol_','turnover_','amount_'))])}个")
        return df

    # ---- 3. 相对波动特征 ----
    def _relative_volatility_features(self, df):
        """计算个股相对于指数、板块的超额波动和beta"""
        # 指数相对振幅
        idx_cols = [c for c in df.columns if c.startswith("idx_")]
        ind_cols = [c for c in df.columns if c.startswith("ind_")]

        # 计算指数日内振幅
        if "idx_high" in df.columns and "idx_low" in df.columns and "idx_close" in df.columns:
            prev_idx_close = df["idx_close"].shift(1)
            df["idx_amplitude"] = (df["idx_high"] - df["idx_low"]) / prev_idx_close.abs()

        # 计算板块日内振幅
        if "ind_high" in df.columns and "ind_low" in df.columns and "ind_close" in df.columns:
            prev_ind_close = df["ind_close"].shift(1)
            df["ind_amplitude"] = (df["ind_high"] - df["ind_low"]) / prev_ind_close.abs()

        # 超额振幅
        if "amplitude" in df.columns and "idx_amplitude" in df.columns:
            df["excess_amplitude"] = df["amplitude"] - df["idx_amplitude"]

        # Rolling CAPM: stock_return = alpha + beta * market_return + epsilon
        if "pct_change" in df.columns and "idx_close" in df.columns:
            stock_ret = df["pct_change"] / 100.0  # % → 小数
        else:
            stock_ret = df["close"].pct_change()

        if "idx_close" in df.columns:
            idx_ret = df["idx_close"].pct_change()
        elif "idx_pct_change" in df.columns:
            idx_ret = df["idx_pct_change"] / 100.0
        else:
            idx_ret = None

        if idx_ret is not None:
            df["beta_60"] = np.nan
            df["r_squared"] = np.nan
            df["residual_vol"] = np.nan

            for i in range(60, len(df)):
                y = stock_ret.iloc[i-60:i].values
                x = idx_ret.iloc[i-60:i].values
                mask = ~(np.isnan(y) | np.isnan(x))
                y, x = y[mask], x[mask]
                if len(y) < 30:
                    continue
                try:
                    x_c = add_constant(x)
                    model = OLS(y, x_c).fit()
                    df.iloc[i, df.columns.get_loc("beta_60")] = model.params[1] if len(model.params) > 1 else np.nan
                    df.iloc[i, df.columns.get_loc("r_squared")] = model.rsquared
                    df.iloc[i, df.columns.get_loc("residual_vol")] = np.std(model.resid)
                except Exception:
                    pass

            # 特质波动占比 = 1 - R²（越高越适合做T）
            df["idio_ratio"] = 1.0 - df["r_squared"].fillna(0.5)
            df["idio_ratio"] = df["idio_ratio"].clip(0, 1)

            # 指数联动程度分类
            df["market_sensitivity"] = pd.cut(
                df["beta_60"].fillna(1.0),
                bins=[-np.inf, 0.5, 0.8, 1.2, 1.5, np.inf],
                labels=["弱相关", "低相关", "中等", "高相关", "强相关"],
            )

            logger.info(f"  相对波动特征: beta/r_squared/residual_vol/ido_ratio/excess_amplitude")
        return df

    # ---- 4. 日历特征 ----
    def _calendar_features(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        df["dow"] = df.index.dayofweek  # 0=周一, 4=周五
        df["is_monday"] = (df["dow"] == 0).astype(int)
        df["is_friday"] = (df["dow"] == 4).astype(int)
        df["is_month_start"] = (df.index.day <= 3).astype(int)
        df["is_month_end"] = (df.index.day >= df.index.days_in_month - 3).astype(int)
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

        return df

    # ---- 5. 目标特征（用于回测和分析） ----
    def _target_features(self, df):
        """构造做T的潜在收益和方向特征"""
        if "amplitude" not in df.columns:
            return df

        amp = df["amplitude"]

        # 次日振幅（检验GARCH预测力时用）
        df["next_amplitude"] = amp.shift(-1)

        # 高振幅日标记（振幅 > 滚动中位数）
        for w in [20, 60]:
            median = amp.rolling(w).median()
            df[f"is_high_amp_{w}"] = (amp > median * 1.5).astype(int)

        # 极端振幅标记（振幅 > P90）
        for w in [60, 120]:
            q90 = amp.rolling(w).quantile(0.9)
            df[f"is_extreme_amp_{w}"] = (amp > q90).astype(int)

        # 做T理论最大收益（如果在最低点买入、最高点卖出）
        # max_profit = amplitude - 2*cost（简化）
        df["t_max_profit"] = amp - 0.003  # 扣除双边成本约0.3%

        return df

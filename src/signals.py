"""
信号系统：生成做T可行性综合评分
"""
import numpy as np
import pandas as pd
from src.utils import logger


class TSignalGenerator:
    """做T信号生成器"""

    def __init__(self, weights=None, threshold=0.6):
        self.weights = weights or {
            "amplitude": 0.40,
            "liquidity": 0.20,
            "idio": 0.25,
            "regime": 0.15,
        }
        self.threshold = threshold

    def generate_all_signals(self, df):
        """主入口：计算各维度评分 + 综合评分"""
        df = df.copy()

        df["score_amplitude"] = self._amplitude_score(df)
        df["score_liquidity"] = self._liquidity_score(df)
        df["score_idio"] = self._idio_score(df)
        df["score_regime"] = self._regime_score(df)

        # 综合评分
        w = self.weights
        df["composite_score"] = (
            w["amplitude"] * df["score_amplitude"].fillna(0.5) +
            w["liquidity"] * df["score_liquidity"].fillna(0.5) +
            w["idio"] * df["score_idio"].fillna(0.5) +
            w["regime"] * df["score_regime"].fillna(0.5)
        )

        # 信号分级
        df["signal"] = (df["composite_score"] >= self.threshold).astype(int)
        df["signal_level"] = pd.cut(
            df["composite_score"],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["弱", "中", "强", "极强"],
        )

        signal_pct = df["signal"].mean()
        logger.info(f"信号生成完成: 信号日占比={signal_pct:.2%} (阈值={self.threshold})")
        return df

    # ---- 振幅预期评分 ----
    def _amplitude_score(self, df):
        """基于振幅的历史分位数评分"""
        if "amplitude" not in df.columns:
            return pd.Series(0.5, index=df.index)

        amp = df["amplitude"]

        # 用20天和60天的百分位取平均
        pctile_20 = amp.rolling(252, min_periods=20).rank(pct=True)  # 1年滚动窗口
        pctile_60 = amp.rolling(252, min_periods=60).rank(pct=True)

        # 组合百分位评分
        score = (pctile_20.fillna(0.5) + pctile_60.fillna(0.5)) / 2

        # 如果GARCH条件波动率可用（analysis阶段计算），则结合
        return score.clip(0, 1)

    # ---- 流动性评分 ----
    def _liquidity_score(self, df):
        """成交量和换手率的综合评分"""
        scores = []

        if "vol_ratio_20" in df.columns:
            vol_score = df["vol_ratio_20"].clip(0.3, 3.0)
            vol_score = (vol_score - 0.3) / (3.0 - 0.3)
            vol_score = vol_score.clip(0, 1)
            scores.append(vol_score)

        if "turnover_rate" in df.columns:
            # 换手率适中最好（0.5%~5%为佳）
            tr = df["turnover_rate"] * 100  # 转百分比
            tr_score = np.where(
                (tr >= 0.5) & (tr <= 5.0),
                1.0,
                np.where(tr < 0.5, tr / 0.5, np.maximum(0, 1 - (tr - 5.0) / 10.0)),
            )
            scores.append(pd.Series(tr_score, index=df.index))

        if "amount" in df.columns:
            # 成交额 > 1亿为佳
            amt = df["amount"] / 1e8  # 转亿
            amt_score = np.clip(amt / 5.0, 0, 1)  # 5亿为满分
            scores.append(pd.Series(amt_score, index=df.index))

        if not scores:
            return pd.Series(0.5, index=df.index)

        result = sum(scores) / len(scores)
        return result.fillna(0.5).clip(0, 1)

    # ---- 特质波动评分 ----
    def _idio_score(self, df):
        """特质波动占比越高越好"""
        if "idio_ratio" in df.columns:
            return df["idio_ratio"].fillna(0.5).clip(0, 1)

        if "r_squared" in df.columns:
            return (1.0 - df["r_squared"].fillna(0.5)).clip(0, 1)

        return pd.Series(0.5, index=df.index)

    # ---- 波动率区间评分 ----
    def _regime_score(self, df):
        """偏好中等偏高的波动区间（太低没空间，太高可能恐慌）"""
        if "amp_ma_20" not in df.columns or "amplitude" not in df.columns:
            return pd.Series(0.5, index=df.index)

        # 当前振幅与历史20日均值的比率
        ratio = df["amplitude"] / df["amp_ma_20"].replace(0, np.nan)

        # 倒U型：0.8~1.5倍均值最好
        # <0.5: 太平淡; >2.0: 太剧烈
        score = np.where(
            ratio.isna(), 0.5,
            np.where(
                ratio < 0.5, 0.2,
                np.where(
                    ratio < 0.8, (ratio - 0.5) / 0.3 * 0.5 + 0.2,
                    np.where(
                        ratio <= 1.5, 0.8 + (ratio - 0.8) / 0.7 * 0.2,
                        np.where(
                            ratio <= 2.0, 1.0 - (ratio - 1.5) / 0.5 * 0.5,
                            0.5,
                        ),
                    ),
                ),
            ),
        )

        return pd.Series(score, index=df.index).fillna(0.5).clip(0, 1)

    # ---- 跨股票排名（多股票扩展时使用） ----
    @staticmethod
    def rank_signals(signals_dict):
        """输入 {stock_code: signal_df}，返回按日期的截面排名"""
        # 取每天的 composite_score，按日排名
        all_scores = {}
        for code, sdf in signals_dict.items():
            if "composite_score" in sdf.columns:
                all_scores[code] = sdf["composite_score"]

        if not all_scores:
            return None

        score_df = pd.DataFrame(all_scores)
        rank_df = score_df.rank(axis=1, ascending=False, method="min")
        return rank_df

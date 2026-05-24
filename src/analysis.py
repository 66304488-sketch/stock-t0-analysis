"""
统计分析：振幅分布、GARCH波动聚集、条件振幅、波动分解
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from src.utils import logger


class AmplitudeAnalyzer:
    """振幅统计分析器"""

    def __init__(self):
        self.results = {}

    def run_all(self, df):
        """运行全部分析"""
        logger.info("=" * 50)
        logger.info("开始统计分析...")
        amp = df["amplitude"].dropna()

        self.results["distribution"] = self.fit_distribution(amp)
        self.results["seasonality"] = self.analyze_seasonality(df)
        self.results["conditional"] = self.conditional_amplitude(df)
        self.results["decomposition"] = self.decompose_volatility(df)
        self.results["garch"] = self.analyze_garch(df)
        self.results["tail"] = self.tail_analysis(df)

        self._print_summary()
        return self.results

    # ---- 1. 分布拟合 ----
    def fit_distribution(self, amplitude_series):
        """对日内振幅做多分布拟合并AIC选优"""
        amp = amplitude_series.dropna()
        # 移除极端异常值（> P99.9）
        amp = amp[amp < amp.quantile(0.999)]

        result = {
            "n": len(amp),
            "mean": amp.mean(),
            "std": amp.std(),
            "median": amp.median(),
            "min": amp.min(),
            "max": amp.max(),
            "skewness": amp.skew(),
            "kurtosis": amp.kurtosis(),
            "percentiles": {
                "p50": amp.quantile(0.50),
                "p75": amp.quantile(0.75),
                "p90": amp.quantile(0.90),
                "p95": amp.quantile(0.95),
                "p99": amp.quantile(0.99),
            },
            "fits": {},
        }

        # 尝试拟合多个分布
        dists_to_try = {
            "lognorm": sp_stats.lognorm,
            "gamma": sp_stats.gamma,
            "norm": sp_stats.norm,
            "t": sp_stats.t,
        }

        for name, dist in dists_to_try.items():
            try:
                if name == "t":
                    # t分布额外参数：df（自由度）
                    params = sp_stats.t.fit(amp)
                    log_lik = np.sum(sp_stats.t.logpdf(amp, *params))
                    k = len(params)
                else:
                    params = dist.fit(amp)
                    log_lik = np.sum(dist.logpdf(amp, *params))
                    k = len(params)

                aic = 2 * k - 2 * log_lik
                bic = k * np.log(len(amp)) - 2 * log_lik
                result["fits"][name] = {
                    "params": [float(p) for p in params],
                    "aic": float(aic),
                    "bic": float(bic),
                    "log_lik": float(log_lik),
                }
            except Exception as e:
                logger.warning(f"  分布拟合失败: {name} - {e}")

        # AIC最优
        if result["fits"]:
            best = min(result["fits"].items(), key=lambda x: x[1]["aic"])
            result["best_fit"] = best[0]
            logger.info(f"  最优分布: {best[0]} (AIC={best[1]['aic']:.1f})")

        return result

    # ---- 2. 季节性分析 ----
    def analyze_seasonality(self, df):
        """振幅的周内/月内/季内效应"""
        result = {}
        amp = df["amplitude"].dropna()

        if "dow" in df.columns:
            dow_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
            dow_stats = amp.groupby(df["dow"]).agg(["mean", "std", "count"])
            dow_stats.index = dow_stats.index.map(dow_names)
            result["day_of_week"] = dow_stats.to_dict()

            # ANOVA检验
            groups = [amp[df["dow"] == i].dropna() for i in range(5)]
            if all(len(g) > 5 for g in groups):
                f_stat, p_value = sp_stats.f_oneway(*groups)
                result["dow_anova_p"] = float(p_value)
                logger.info(f"  周内效应ANOVA p={p_value:.4f} {'(显著)' if p_value < 0.05 else '(不显著)'}")

        if "month" in df.columns:
            month_stats = amp.groupby(df["month"]).agg(["mean", "std", "count"])
            result["monthly"] = month_stats.to_dict()

        if "quarter" in df.columns:
            qtr_stats = amp.groupby(df["quarter"]).agg(["mean", "std", "count"])
            result["quarterly"] = qtr_stats.to_dict()

        return result

    # ---- 3. 条件振幅分析 ----
    def conditional_amplitude(self, df):
        """给定指数和板块涨跌幅，分析个股振幅的条件分布"""
        result = {}

        # 指数的条件分析
        idx_ret_col = None
        for c in ["idx_pct_change", "idx_close"]:
            if c in df.columns:
                idx_ret_col = c
                break

        if idx_ret_col and "amplitude" in df.columns:
            if idx_ret_col == "idx_close":
                idx_ret = df["idx_close"].pct_change()
            else:
                idx_ret = df[idx_ret_col] / 100.0

            # 按指数涨跌幅分桶
            bins = [-np.inf, -0.03, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, np.inf]
            labels = ["<-3%", "-3%~-2%", "-2%~-1%", "-1%~-0.5%", "-0.5%~0", "0~0.5%", "0.5%~1%", "1%~2%", "2%~3%", ">3%"]
            bucket = pd.cut(idx_ret, bins=bins, labels=labels)

            cond_stats = df["amplitude"].groupby(bucket, observed=False).agg(["mean", "std", "count", "median"])
            result["by_index_move"] = cond_stats.to_dict()
            logger.info(f"  指数条件分析: {len(cond_stats.dropna())}个分桶")

        # 板块的条件分析
        ind_ret_col = None
        for c in ["ind_pct_change", "ind_close"]:
            if c in df.columns:
                ind_ret_col = c
                break

        if ind_ret_col and "amplitude" in df.columns:
            if ind_ret_col == "ind_close":
                ind_ret = df["ind_close"].pct_change()
            else:
                ind_ret = df[ind_ret_col] / 100.0

            bucket = pd.cut(ind_ret, bins=bins, labels=labels)
            cond_stats = df["amplitude"].groupby(bucket, observed=False).agg(["mean", "std", "count", "median"])
            result["by_industry_move"] = cond_stats.to_dict()

        # 2D分析：指数×板块 联合条件
        if (idx_ret_col or idx_ret_col is not None) and (ind_ret_col or ind_ret_col is not None):
            # 简化：指数涨/跌 × 板块涨/跌 四种组合
            df_tmp = df.copy()
            if idx_ret_col == "idx_close":
                df_tmp["_idx_dir"] = np.where(df["idx_close"].pct_change() > 0, "指数涨", "指数跌")
            else:
                df_tmp["_idx_dir"] = np.where(df[idx_ret_col] > 0, "指数涨", "指数跌")

            if ind_ret_col == "ind_close":
                df_tmp["_ind_dir"] = np.where(df["ind_close"].pct_change() > 0, "板块涨", "板块跌")
            else:
                df_tmp["_ind_dir"] = np.where(df[ind_ret_col] > 0, "板块涨", "板块跌")

            combo = df_tmp.groupby(["_idx_dir", "_ind_dir"], observed=False)["amplitude"].mean()
            result["by_index_industry_combo"] = combo.to_dict()

        return result

    # ---- 4. 波动分解 ----
    def decompose_volatility(self, df):
        """CAPM风格：total_var = market_var + industry_var + idiosyncratic_var"""
        result = {}

        # 计算日收益率
        stock_ret = df.get("pct_change", df["close"].pct_change() * 100) / 100.0

        has_idx = any(c.startswith("idx_close") or c.startswith("idx_pct_change") for c in df.columns)
        has_ind = any(c.startswith("ind_close") or c.startswith("ind_pct_change") for c in df.columns)

        if not has_idx:
            logger.info("  无指数数据，跳过波动分解")
            return result

        # 准备回归变量
        if "idx_pct_change" in df.columns:
            market_ret = df["idx_pct_change"] / 100.0
        else:
            market_ret = df["idx_close"].pct_change()

        # Market model: stock_ret = alpha + beta_mkt * market_ret + epsilon
        data = pd.DataFrame({"stock": stock_ret, "market": market_ret}).dropna()

        if has_ind:
            if "ind_pct_change" in df.columns:
                ind_ret = df["ind_pct_change"] / 100.0
            elif "ind_close" in df.columns:
                ind_ret = df["ind_close"].pct_change()
            else:
                ind_ret = None

            if ind_ret is not None:
                data["industry"] = ind_ret
                # 先对行业回归market，取残差（正交化）
                ind_on_mkt = ind_ret - market_ret  # 简化：板块超额 = 板块 - 市场

        data = data.dropna()

        if len(data) < 30:
            return result

        # 总方差
        total_var = data["stock"].var()
        market_var = data["market"].var()

        # Market model regression
        import statsmodels.api as sm
        X = sm.add_constant(data["market"])
        model = sm.OLS(data["stock"], X).fit()
        r2_mkt = model.rsquared
        explained_by_market = r2_mkt * total_var

        result["total_variance"] = float(total_var)
        result["market_variance"] = float(market_var)
        result["explained_by_market"] = float(explained_by_market)
        result["r_squared_market"] = float(r2_mkt)
        result["idio_ratio"] = float(1.0 - r2_mkt)

        if "industry" in data.columns:
            X2 = sm.add_constant(data[["market", "industry"]])
            model2 = sm.OLS(data["stock"], X2).fit()
            r2_full = model2.rsquared
            result["r_squared_full"] = float(r2_full)
            result["explained_by_market_industry"] = float(r2_full * total_var)
            result["idio_ratio_full"] = float(1.0 - r2_full)

        logger.info(f"  波动分解: 总方差={total_var:.6f}, R²_mkt={r2_mkt:.3f}, 特质占比={1-r2_mkt:.3f}")
        return result

    # ---- 5. GARCH分析 ----
    def analyze_garch(self, df):
        """GARCH(1,1)波动聚集分析"""
        result = {}
        amp = df["amplitude"].dropna()

        try:
            from arch import arch_model
            # 放大振幅以避免数值过小
            scaled = amp * 100
            scaled = scaled.dropna()
            model = arch_model(scaled, vol="GARCH", p=1, q=1, dist="normal")
            fit = model.fit(disp="off")
            result["converged"] = True
            result["params"] = {k: float(v) for k, v in fit.params.items()}
            result["aic"] = float(fit.aic)
            result["bic"] = float(fit.bic)
            # GARCH持续性：alpha+beta接近1 = 强聚集
            alpha1 = fit.params.get("alpha[1]", 0)
            beta1 = fit.params.get("beta[1]", 0)
            result["persistence"] = float(alpha1 + beta1)
            result["conditional_vol"] = fit.conditional_volatility.values.tolist()
            logger.info(f"  GARCH(1,1) converge, persistence={alpha1+beta1:.3f} (接近1=强聚集)")
        except Exception as e:
            result["converged"] = False
            result["error"] = str(e)
            logger.warning(f"  GARCH不收敛, 将使用EWMA: {e}")

        return result

    # ---- 6. 尾部分析 ----
    def tail_analysis(self, df):
        """高振幅日的统计特征"""
        amp = df["amplitude"].dropna()
        p90 = amp.quantile(0.90)
        p95 = amp.quantile(0.95)

        # 高振幅日的前一天特征
        df_tmp = df.copy()
        df_tmp["high_amp"] = (df_tmp["amplitude"] > p90).astype(int)
        df_tmp["very_high_amp"] = (df_tmp["amplitude"] > p95).astype(int)

        result = {
            "p90_threshold": float(p90),
            "p95_threshold": float(p95),
            "high_amp_pct": float(df_tmp["high_amp"].mean()),
            "very_high_amp_pct": float(df_tmp["very_high_amp"].mean()),
        }

        # 高振幅前1天的平均振幅（检验"暴风雨前的宁静"）
        high_days = df_tmp[df_tmp["high_amp"] == 1].index
        if len(high_days) > 0:
            prev_amps = []
            for d in high_days:
                prev_idx = d - pd.Timedelta(days=1)
                if prev_idx in df_tmp.index:
                    prev_amps.append(df_tmp.loc[prev_idx, "amplitude"])
            if prev_amps:
                result["avg_amplitude_day_before_high"] = float(np.mean(prev_amps))
                result["overall_avg_amplitude"] = float(amp.mean())

        # 高振幅日的持续性
        streaks = []
        current = 0
        for v in df_tmp["high_amp"]:
            if v == 1:
                current += 1
            else:
                if current > 0:
                    streaks.append(current)
                current = 0
        if current > 0:
            streaks.append(current)

        if streaks:
            result["max_high_amp_streak"] = max(streaks)
            result["avg_high_amp_streak"] = float(np.mean(streaks))

        logger.info(f"  尾部分析: P90={p90:.4f}, 高振幅占比={df_tmp['high_amp'].mean():.2%}")
        return result

    # ---- 结果汇总 ----
    def _print_summary(self):
        dist = self.results.get("distribution", {})
        decomp = self.results.get("decomposition", {})
        tail = self.results.get("tail", {})

        logger.info("=" * 50)
        logger.info("分析摘要:")
        logger.info(f"  样本数: {dist.get('n', 'N/A')}")
        logger.info(f"  振幅均值: {dist.get('mean', 0):.4f} ({dist.get('mean', 0)*100:.2f}%)")
        logger.info(f"  振幅中位数: {dist.get('median', 0):.4f} ({dist.get('median', 0)*100:.2f}%)")
        logger.info(f"  最优分布: {dist.get('best_fit', 'N/A')}")
        logger.info(f"  特质波动占比(R²_mkt): {1-decomp.get('r_squared_market', 0):.3f}")
        logger.info(f"  高振幅日(P>P90)占比: {tail.get('high_amp_pct', 0):.2%}")
        logger.info("=" * 50)

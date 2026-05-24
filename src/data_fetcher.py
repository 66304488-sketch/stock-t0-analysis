"""
数据获取层 - AKShare封装 + 本地缓存
"""
import json
import os
import time
from datetime import datetime

import pandas as pd

from config import CACHE_DIR, CACHE_REGISTRY, MAX_RETRIES, RETRY_DELAY, INTRADAY_BATCH_DAYS, INTRADAY_PERIOD
from src.utils import logger


class DataCache:
    """本地缓存管理，key = (func_name, symbol, start, end)"""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.registry_path = os.path.join(os.path.dirname(CACHE_DIR), os.path.basename(CACHE_REGISTRY))
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def _key(self, func_name, symbol, start, end):
        return f"{func_name}__{symbol}__{start}__{end}"

    def get(self, func_name, symbol, start, end, max_age_days=1):
        key = self._key(func_name, symbol, start, end)
        if key in self.registry:
            entry = self.registry[key]
            filepath = entry["filepath"]
            fetch_time = datetime.fromisoformat(entry["fetch_time"])
            age = (datetime.now() - fetch_time).days
            if age <= max_age_days and os.path.exists(filepath):
                logger.info(f"缓存命中: {key}")
                return pd.read_parquet(filepath)
        return None

    def put(self, func_name, symbol, start, end, df):
        key = self._key(func_name, symbol, start, end)
        filename = f"{func_name}_{symbol}_{start}_{end}.parquet"
        filepath = os.path.join(CACHE_DIR, filename)
        df.to_parquet(filepath, index=False)
        self.registry[key] = {
            "filepath": filepath,
            "fetch_time": datetime.now().isoformat(),
            "rows": len(df),
        }
        self._save()
        logger.info(f"缓存已保存: {key} ({len(df)}行)")


class StockDataFetcher:
    """股票数据获取器"""

    def __init__(self):
        self.cache = DataCache()

    def _to_sina_code(self, stock_code):
        """转为新浪格式: 600031 -> sh600031"""
        if stock_code.startswith("sh") or stock_code.startswith("sz"):
            return stock_code
        if stock_code.startswith(("6", "5")):
            return f"sh{stock_code}"
        return f"sz{stock_code}"

    def _supplement_fields(self, df):
        """补充计算缺失字段: amplitude, pct_change, volume"""
        df = df.copy()
        if "close" in df.columns and len(df) > 1:
            prev_close = df["close"].shift(1)
            if "pct_change" not in df.columns:
                df["pct_change"] = ((df["close"] - prev_close) / prev_close * 100).round(4)
            if "amplitude" not in df.columns and "high" in df.columns and "low" in df.columns:
                df["振幅"] = ((df["high"] - df["low"]) / prev_close * 100).round(4)
        if "volume" in df.columns and "amount" not in df.columns:
            # 从成交额推算volume（近似，如果没有的话）
            pass
        return df

    def fetch_daily(self, stock_code, start, end, adjust="qfq"):
        """获取个股日线OHLCV（前复权），多数据源自动切换"""
        cached = self.cache.get("daily", stock_code, start, end, max_age_days=1)
        if cached is not None:
            return cached

        import akshare as ak

        # 数据源1: 东方财富 (最完整)
        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
            if df is not None and not df.empty:
                df = self._supplement_fields(df)
                self.cache.put("daily", stock_code, start, end, df)
                return df
        except Exception as e:
            logger.warning(f"东方财富数据源失败: {e}, 切换到腾讯数据源")

        # 数据源2: 新浪 (有volume和turnover)
        try:
            sina_code = self._to_sina_code(stock_code)
            df = ak.stock_zh_a_daily(
                symbol=sina_code, start_date=start, end_date=end, adjust=adjust,
            )
            if df is not None and not df.empty:
                df = self._supplement_fields(df)
                self.cache.put("daily", stock_code, start, end, df)
                logger.info(f"新浪数据源成功: {len(df)}行")
                return df
        except Exception as e:
            logger.warning(f"新浪数据源失败: {e}, 切换到腾讯数据源")

        # 数据源3: 腾讯 (保底，字段最少)
        tx_code = self._to_sina_code(stock_code)
        df = ak.stock_zh_a_hist_tx(symbol=tx_code, start_date=start, end_date=end, adjust=adjust)
        if df is not None and not df.empty:
            df = self._supplement_fields(df)
            self.cache.put("daily", stock_code, start, end, df)
            logger.info(f"腾讯数据源成功: {len(df)}行")
            return df

        raise RuntimeError(f"所有数据源均无法获取 {stock_code} 的数据")

    def fetch_minute(self, stock_code, period=None):
        """获取个股分钟线数据（15分钟），可能因API不稳定而失败"""
        period = period or INTRADAY_PERIOD
        if period not in ("1", "5", "15", "30", "60"):
            period = "15"

        try:
            import akshare as ak
            # 分钟线数据量大，不设固定日期范围，获取最近约1年数据
            df = ak.stock_zh_a_hist_min_em(symbol=stock_code, period=period, adjust="qfq")
            if df is not None and not df.empty:
                logger.info(f"分钟线获取成功: {stock_code}, {len(df)}行, period={period}")
                return df
            return None
        except Exception as e:
            logger.warning(f"分钟线获取失败({stock_code}), 将仅使用日线分析: {e}")
            return None

    def fetch_index_daily(self, index_code, start, end):
        """获取指数日线（如上证指数 sh000001）"""
        cached = self.cache.get("index", index_code, start, end, max_age_days=1)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES + 1):
            try:
                import akshare as ak
                df = ak.stock_zh_index_daily(symbol=index_code)
                if df is None or df.empty:
                    raise ValueError("返回空数据")
                # 标准化列名并过滤日期范围
                col_map = {"date": "date", "open": "open", "close": "close",
                           "high": "high", "low": "low", "volume": "volume"}
                df = df.rename(columns=col_map) if any(c in df.columns for c in col_map) else df
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[(df["date"] >= start) & (df["date"] <= end)]
                self.cache.put("index", index_code, start, end, df)
                return df
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"fetch_index {index_code} 失败(尝试{attempt+1}): {e}, 重试中...")
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    raise

    def fetch_industry_index(self, industry_name, start, end):
        """获取板块指数日线（同花顺行业指数）"""
        cached = self.cache.get("industry", industry_name, start, end, max_age_days=1)
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES + 1):
            try:
                import akshare as ak
                df = ak.stock_board_industry_index_ths(symbol=industry_name)
                if df is None or df.empty:
                    raise ValueError(f"未找到板块: {industry_name}")
                if "date" not in df.columns and "日期" in df.columns:
                    df = df.rename(columns={"日期": "date"})
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[(df["date"] >= start) & (df["date"] <= end)]
                self.cache.put("industry", industry_name, start, end, df)
                return df
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"fetch_industry {industry_name} 失败(尝试{attempt+1}): {e}, 重试中...")
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    logger.warning(f"板块数据获取失败({industry_name}), 将跳过板块分析: {e}")
                    return None

    def fetch_all_for_stock(self, stock_code, start, end, market_index=None, industry=None):
        """一次性获取个股+指数+板块+分钟线"""
        logger.info(f"开始获取 {stock_code} 数据 ({start} ~ {end})")
        result = {}

        result["daily"] = self.fetch_daily(stock_code, start, end)

        if market_index:
            result["index"] = self.fetch_index_daily(market_index, start, end)

        if industry:
            result["industry"] = self.fetch_industry_index(industry, start, end)

        result["minute"] = self.fetch_minute(stock_code)

        return result

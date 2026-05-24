"""
数据清洗：列名标准化、多源对齐、数据验证
"""
import pandas as pd
import numpy as np
from src.utils import logger


# 东方财富数据源的列名映射
_CN_COL_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    "振幅": "amplitude", "涨跌幅": "pct_change", "涨跌额": "change",
    "换手率": "turnover_rate",
}

# 板块指数列名映射
_IND_COL_MAP = {
    "日期": "date", "开盘价": "open", "收盘价": "close",
    "最高价": "high", "最低价": "low", "成交量": "volume", "成交额": "amount",
}


def standardize_daily_columns(df):
    """标准化个股日线列名，设置日期索引"""
    if df is None or df.empty:
        return None
    df = df.copy()

    # 映射中文列名（如果有的话）
    df = df.rename(columns={k: v for k, v in _CN_COL_MAP.items() if k in df.columns})

    # 统一振幅列名（新浪数据源为 振幅，东方财富映射后为amplitude）
    if "振幅" in df.columns and "amplitude" not in df.columns:
        df = df.rename(columns={"振幅": "amplitude"})

    # 统一换手率列名（新浪为turnover）
    if "turnover" in df.columns and "turnover_rate" not in df.columns:
        df = df.rename(columns={"turnover": "turnover_rate"})

    # date列 → DatetimeIndex
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"
        except Exception:
            pass

    df = df.sort_index()

    # 确保必要列存在
    required = ["open", "close", "high", "low"]
    for col in required:
        if col not in df.columns:
            logger.warning(f"缺少列: {col}")

    # 移除OHLC全空的行
    available = [c for c in required if c in df.columns]
    before = len(df)
    df = df.dropna(subset=available)
    if len(df) < before:
        logger.info(f"移除了 {before - len(df)} 行含NaN的数据")

    return df


def standardize_index_columns(df, prefix="idx"):
    """标准化指数/板块日线列名，加前缀避免混淆"""
    if df is None or df.empty:
        return None
    df = df.copy()

    # 映射板块指数的中文列名
    df = df.rename(columns={k: v for k, v in _IND_COL_MAP.items() if k in df.columns})
    # 也尝试通用中文映射
    cn_extra = {"涨跌幅": "pct_change", "涨跌额": "change"}
    df = df.rename(columns={k: v for k, v in cn_extra.items() if k in df.columns})

    # date列
    date_cols = [c for c in df.columns if c.lower() in ("date", "日期", "trade_date")]
    if date_cols:
        df["date"] = pd.to_datetime(df[date_cols[0]])
        if date_cols[0] != "date":
            df = df.drop(columns=[date_cols[0]])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            logger.warning("无法将索引转为DatetimeIndex")

    df = df.sort_index()

    # 给除date外的所有列加前缀
    rename = {}
    for c in df.columns:
        if not c.startswith(f"{prefix}_"):
            rename[c] = f"{prefix}_{c}"
    df = df.rename(columns=rename)

    return df


def merge_all(daily_df, index_df=None, industry_df=None):
    """左连接：以个股日线为主，合并指数和板块数据"""
    if daily_df is None or daily_df.empty:
        raise ValueError("个股日线数据为空")

    merged = daily_df.copy()

    for df, label in [(index_df, "指数"), (industry_df, "板块")]:
        if df is not None and not df.empty:
            before = len(merged)
            merged = merged.join(df, how="left")
            matched = df.index.intersection(merged.index)
            logger.info(f"{label}数据合并: 个股{before}行, {label}{len(df)}行, 匹配{len(matched)}天")

    return merged


def validate_and_fix(merged_df):
    """数据质量检查"""
    df = merged_df.copy()
    issues = []

    # 检查振幅列
    amp_col = "amplitude" if "amplitude" in df.columns else None
    if amp_col and "high" in df.columns and "low" in df.columns and "close" in df.columns:
        prev_close = df["close"].shift(1)
        calc_amplitude = (df["high"] - df["low"]) / prev_close.abs()
        # 振幅在数据中是百分比形式（如5.0 = 5%），也可能是小数形式
        amp_values = df[amp_col]
        if amp_values.abs().max() < 1:  # 小数形式
            calc_amplitude_pct = calc_amplitude * 100
        else:
            calc_amplitude_pct = calc_amplitude

    # 检查高低点顺序
    if "high" in df.columns and "low" in df.columns:
        bad_hl = df[df["high"] < df["low"]]
        if len(bad_hl) > 0:
            issues.append(f"最高价<最低价的行数: {len(bad_hl)}")

    # 检查成交量<=0
    if "volume" in df.columns:
        bad_vol = df[df["volume"] <= 0]
        if len(bad_vol) > 0:
            issues.append(f"成交量为0的行数: {len(bad_vol)}")

    if issues:
        for i in issues:
            logger.warning(f"数据问题: {i}")
    else:
        logger.info("数据质量检查通过")

    return df

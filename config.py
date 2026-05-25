"""
A股个股波动做T套利分析 - 全局配置
"""
import os, re

# ---- 股票池 ----
STOCKS = {
    "600031": {
        "active": True,
        "name": "三一重工",
        "industry": "工程机械",
        "market_index": "sh000001",  # 上证指数
    },
    "601899": {
        "active": True,
        "name": "紫金矿业",
        "industry": "贵金属",
        "market_index": "sh000001",
    },
    "601225": {
        "active": True,
        "name": "陕西煤业",
        "industry": "煤炭",
        "market_index": "sh000001",
    },
    "600519": {
        "active": True,
        "name": "贵州茅台",
        "industry": "白酒",
        "market_index": "sh000001",
    },
    "300750": {
        "active": True,
        "name": "宁德时代",
        "industry": "电池",
        "market_index": "sz399006",  # 创业板指
    },
    "513050": {
        "active": True,
        "name": "中概互联ETF",
        "industry": "跨境ETF",
        "market_index": "sh000001",  # 上证指数
    },
    "002928": {
        "active": True,
        "name": "华夏航空",
        "industry": "航空运输",
        "market_index": "sz399001",  # 深证成指
    },
    "002318": {
        "active": True,
        "name": "久立特材",
        "industry": "特钢",
        "market_index": "sz399001",  # 深证成指
    },
    "603444": {
        "active": True,
        "name": "吉比特",
        "industry": "游戏",
        "market_index": "sh000001",
    },
    "601127": {
        "active": True,
        "name": "赛力斯",
        "industry": "汽车",
        "market_index": "sh000001",
    },
    "002182": {
        "active": True,
        "name": "宝武镁业",
        "industry": "有色金属",
        "market_index": "sz399001",
    },
    "601058": {
        "active": True,
        "name": "赛轮轮胎",
        "industry": "轮胎",
        "market_index": "sh000001",
    },
    "002281": {
        "active": True,
        "name": "光迅科技",
        "industry": "光通信",
        "market_index": "sz399001",
    },
    "002714": {
        "active": True,
        "name": "牧原股份",
        "industry": "畜牧业",
        "market_index": "sz399001",
    },
    "002028": {
        "active": True,
        "name": "思源电气",
        "industry": "电气设备",
        "market_index": "sz399001",
    },
    "603290": {
        "active": True,
        "name": "斯达半导",
        "industry": "半导体",
        "market_index": "sh000001",
    },
    "00700": {
        "active": True,
        "name": "腾讯控股",
        "industry": "互联网",
        "market_index": "HSI",
    },
}

# ---- 数据日期范围 ----
DEFAULT_START_DATE = "20210101"  # ~4年
DEFAULT_END_DATE = "20260525"

# ---- 数据缓存 ----
CACHE_DIR = "data/raw"
CACHE_REGISTRY = "data/cache_registry.json"

# ---- 分钟线 ----
USE_INTRADAY_DATA = True
INTRADAY_PERIOD = "15"  # 15分钟线
INTRADAY_BATCH_DAYS = 365

# ---- 特征工程窗口 ----
ROLLING_WINDOWS = [5, 10, 20, 60]

# ---- 回测参数 ----
T_COST = 0.0015  # 单边约0.15%（印花税0.05%+佣金0.025%+滑点）
ENTRY_THRESHOLDS = [-0.03, -0.025, -0.02, -0.015, -0.01]
EXIT_THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03]
POSITION_SIZE_PCT = 0.3  # 用仓位的30%做T

# ---- 信号权重 ----
SIGNAL_WEIGHTS = {
    "amplitude": 0.40,
    "liquidity": 0.20,
    "idio": 0.25,
    "regime": 0.15,
}
SIGNAL_THRESHOLD = 0.6  # 综合评分>0.6才发出信号

# ---- API重试 ----
MAX_RETRIES = 3
RETRY_DELAY = 1.0
FETCH_TIMEOUT = 30


# ---- 关注列表管理 ----
def active_stocks():
    """返回关注列表中的股票"""
    return {k: v for k, v in STOCKS.items() if v.get("active", True)}


def set_active(code, active):
    """设置股票的关注状态并持久化到 config.py"""
    STOCKS[code]["active"] = active
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    # 匹配: "CODE": {\n        "active": True/False,
    pattern = rf'("{re.escape(code)}":\s*\{{\s*\n\s*"active":\s*)(True|False)'
    replacement = rf'\g<1>{active}'
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)

"""
A股个股波动做T套利分析 - 全局配置
"""

# ---- 股票池 ----
STOCKS = {
    "600031": {
        "name": "三一重工",
        "industry": "工程机械",
        "market_index": "sh000001",  # 上证指数
    },
    "601899": {
        "name": "紫金矿业",
        "industry": "贵金属",
        "market_index": "sh000001",
    },
    "601225": {
        "name": "陕西煤业",
        "industry": "煤炭",
        "market_index": "sh000001",
    },
    "600519": {
        "name": "贵州茅台",
        "industry": "白酒",
        "market_index": "sh000001",
    },
    "300750": {
        "name": "宁德时代",
        "industry": "电池",
        "market_index": "sz399006",  # 创业板指
    },
    "513050": {
        "name": "中概互联ETF",
        "industry": "跨境ETF",
        "market_index": "sh000001",  # 上证指数
    },
}

# ---- 数据日期范围 ----
DEFAULT_START_DATE = "20210101"  # ~4年
DEFAULT_END_DATE = "20260522"

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

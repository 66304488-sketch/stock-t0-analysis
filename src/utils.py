"""
工具函数：重试、日志、中文字体设置
"""
import time
import logging
import functools
import matplotlib

# ---- 中文字体 ----
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 日志 ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("t_analysis")


def retry_with_backoff(max_retries=3, initial_delay=1.0):
    """带指数退避的重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(f"{func.__name__} 失败 (尝试 {attempt+1}/{max_retries+1}): {e}, {delay}s后重试")
                        time.sleep(delay)
                        delay *= 2
            logger.error(f"{func.__name__} 重试{max_retries}次后仍失败: {last_error}")
            raise last_error
        return wrapper
    return decorator

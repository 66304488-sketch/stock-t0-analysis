#!/usr/bin/env python3
"""
数据同步脚本 — 从 parquet 文件同步数据到 SQLite 数据库
用法:
    python sync_db.py                  # 同步全部股票
    python sync_db.py --stock 600031   # 同步单只
    python sync_db.py --force          # 强制重新计算统计
    python sync_db.py --stats-only     # 仅同步摘要统计（跳过日线）
"""
import sys, os, argparse, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STOCKS
from db import AnalysisDB
from generate_report import extract_stats, load_config, PROJECT_ROOT
import pandas as pd


def sync_daily(db, code):
    parquet_path = PROJECT_ROOT / f"data/processed/{code}_features.parquet"
    if not parquet_path.exists():
        print(f"  [跳过] {code} features.parquet 不存在")
        return 0
    df = pd.read_parquet(parquet_path)
    n = db.insert_daily(code, df)
    print(f"  daily: {n} 行")
    return n


def sync_summary(db, code, force=False):
    if not force and db.has_summary(code):
        print(f"  summary: 已存在 (--force 可强制刷新)")
        return True
    print(f"  计算摘要统计...", end=" ", flush=True)
    t0 = time.time()
    try:
        config = load_config()
        stats = extract_stats(code, config)
        db.upsert_summary(stats)
        db.upsert_seasonality(code, stats)
        print(f"完成 ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:
        print(f"失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="同步股票数据到 SQLite")
    parser.add_argument("--stock", help="仅同步指定股票代码")
    parser.add_argument("--force", action="store_true", help="强制重新计算统计")
    parser.add_argument("--stats-only", action="store_true", help="仅同步摘要统计")
    args = parser.parse_args()

    codes = [args.stock] if args.stock else list(STOCKS.keys())
    db = AnalysisDB()

    for i, code in enumerate(codes):
        stock = STOCKS.get(code, {"name": code, "industry": "default"})
        name = stock.get("name", code)
        print(f"[{i+1}/{len(codes)}] {name} ({code})")

        if not args.stats_only:
            sync_daily(db, code)

        sync_summary(db, code, force=args.force)

    s = db.stats()
    print(f"\n同步完成: {s['stock_count']} 只股票, {s['daily_rows']} 行日线数据")
    db.close()


if __name__ == "__main__":
    main()

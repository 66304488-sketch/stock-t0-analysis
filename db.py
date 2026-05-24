#!/usr/bin/env python3
"""SQLite 数据库模块 — 存储股票日线数据、摘要统计、季节效应"""
import sqlite3
import json
import os
from datetime import datetime

import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "analysis.db")


class AnalysisDB:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily (
            code        TEXT NOT NULL,
            date        TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            amount      REAL,
            amplitude   REAL,
            pct_change  REAL,
            PRIMARY KEY (code, date)
        );
        CREATE INDEX IF NOT EXISTS idx_daily_code ON daily(code);
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date);

        CREATE TABLE IF NOT EXISTS stock_summary (
            code              TEXT PRIMARY KEY,
            name              TEXT NOT NULL,
            industry          TEXT,
            market_index      TEXT,
            data_start        TEXT,
            data_end          TEXT,
            rows              INTEGER,
            updated_at        TEXT DEFAULT (datetime('now','localtime')),

            amp_mean          REAL, amp_median REAL, amp_std REAL,
            amp_min           REAL, amp_max  REAL,
            amp_p10 REAL, amp_p25 REAL, amp_p50 REAL, amp_p75 REAL,
            amp_p90 REAL, amp_p95 REAL, amp_p99 REAL,
            amp_skew          REAL, amp_kurtosis REAL,
            latest_amp        REAL, latest_close REAL,
            amp_ma_5_last     REAL, amp_ma_10_last REAL,
            amp_ma_20_last    REAL, amp_ma_60_last REAL,

            amount_mean       REAL, amount_last REAL,
            close_mean        REAL, close_last  REAL,
            daily_return_mean REAL, daily_return_std REAL,
            total_return_pct  REAL,

            beta_60_mean      REAL, beta_60_last  REAL,
            r_squared_mean    REAL, r_squared_last REAL,
            idio_ratio_mean   REAL, idio_ratio_last REAL,
            residual_vol_mean REAL, residual_vol_last REAL,

            garch_omega       REAL, garch_alpha REAL,
            garch_beta        REAL, garch_persistence REAL,
            garch_error       TEXT,

            signal_pct        REAL, signal_days  INTEGER,
            composite_score_mean REAL, composite_score_last REAL,
            signal_last       INTEGER,
            score_amplitude_mean REAL, score_amplitude_last REAL,
            score_liquidity_mean REAL, score_liquidity_last REAL,
            score_idio_mean   REAL, score_idio_last   REAL,
            score_regime_mean REAL, score_regime_last REAL,

            high_amp_pct      REAL, extreme_amp_pct REAL,
            extreme_amp_days  INTEGER,
            max_high_streak   INTEGER, avg_high_streak REAL,
            recent30_amp_mean REAL, recent30_amp_max REAL, recent30_amp_min REAL,

            dow_amp_json      TEXT, dow_best TEXT, dow_worst TEXT,
            month_amp_json    TEXT, month_best TEXT, month_worst TEXT,
            quarter_amp_json  TEXT,
            month_ret_json    TEXT, month_win_json TEXT,
            month_ret_best    TEXT, month_ret_worst TEXT,

            backtest_top5_json TEXT,

            dates_90_json     TEXT, amp_90_json TEXT,
            monthly_list_json TEXT, dow_list_json TEXT,
            month_ret_list_json TEXT, month_win_list_json TEXT
        );

        CREATE TABLE IF NOT EXISTS seasonality (
            code        TEXT NOT NULL,
            dimension   TEXT NOT NULL,
            label       TEXT NOT NULL,
            amp_mean    REAL,
            ret_mean    REAL,
            win_rate    REAL,
            PRIMARY KEY (code, dimension, label)
        );

        CREATE TABLE IF NOT EXISTS signal_log (
            code            TEXT NOT NULL,
            date            TEXT NOT NULL,
            composite_score REAL,
            signal          INTEGER DEFAULT 0,
            amp_score       REAL,
            liq_score       REAL,
            idio_score      REAL,
            regime_score    REAL,
            next_amplitude  REAL,
            next_pct_change REAL,
            is_win          INTEGER DEFAULT 0,
            profit_est      REAL,
            created_at      TEXT DEFAULT (datetime('now','localtime')),
            PRIMARY KEY (code, date)
        );
        """)

    # ── Daily ──────────────────────────────────────

    def insert_daily(self, code, df):
        cols = ["open", "high", "low", "close", "volume", "amount", "amplitude", "pct_change"]
        available = [c for c in cols if c in df.columns]
        if not available:
            return 0
        df = df.copy()
        if df.index.name is not None or not isinstance(df.index, (range,)):
            df = df.reset_index()
        date_col = df.columns[0] if df.columns[0] not in available else "date"
        if date_col not in df.columns:
            date_col = df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        vals = []
        for _, row in df.iterrows():
            vals.append(tuple([row["code"], row["date"]] + [float(row[c]) if pd.notna(row[c]) else None for c in available]))
        placeholders = ",".join(["?"] * (2 + len(available)))
        cols_sql = ",".join(["code", "date"] + available)
        sql = f"INSERT OR REPLACE INTO daily ({cols_sql}) VALUES ({placeholders})"
        self.conn.executemany(sql, vals)
        self.conn.commit()
        return len(df)

    def get_daily(self, code, start=None, end=None):
        sql = "SELECT * FROM daily WHERE code=? "
        params = [code]
        if start:
            sql += "AND date>=?"
            params.append(start)
        if end:
            sql += "AND date<=?"
            params.append(end)
        sql += " ORDER BY date"

        return pd.read_sql(sql, self.conn, params=params)

    def query_amplitude_above(self, code, threshold=0.05):

        return pd.read_sql(
            "SELECT * FROM daily WHERE code=? AND amplitude>=? ORDER BY date DESC",
            self.conn, params=[code, threshold])

    def get_all_daily(self, date):

        return pd.read_sql(
            "SELECT * FROM daily WHERE date=? ORDER BY code",
            self.conn, params=[date])

    # ── Summary ────────────────────────────────────

    def upsert_summary(self, s):
        scalar_cols = [c[1] for c in self.conn.execute("PRAGMA table_info(stock_summary)") if c[1] not in (
            "dow_amp_json", "month_amp_json", "quarter_amp_json", "month_ret_json", "month_win_json",
            "backtest_top5_json", "dates_90_json", "amp_90_json", "monthly_list_json",
            "dow_list_json", "month_ret_list_json", "month_win_list_json")]

        # Map extract_stats keys to DB column names
        key_map = {"date_start": "data_start", "date_end": "data_end"}
        json_cols = {
            "dow_amp_json": s.get("dow_amp"),
            "month_amp_json": s.get("month_amp"),
            "quarter_amp_json": s.get("quarter_amp"),
            "month_ret_json": s.get("month_ret"),
            "month_win_json": s.get("month_win"),
            "backtest_top5_json": s.get("backtest_top5"),
            "dates_90_json": s.get("dates_90"),
            "amp_90_json": s.get("amp_90"),
            "monthly_list_json": s.get("monthly_list"),
            "dow_list_json": s.get("dow_list"),
            "month_ret_list_json": s.get("monthly_ret_list"),
            "month_win_list_json": s.get("monthly_win_list"),
        }

        vals = {}
        for col in scalar_cols:
            src_key = col
            for ek, dk in key_map.items():
                if dk == col:
                    src_key = ek
                    break
            if src_key in s:
                vals[col] = s[src_key]
        for col, val in json_cols.items():
            if val is not None:
                vals[col] = json.dumps(val, ensure_ascii=False)

        vals["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cols_sql = ",".join(vals.keys())
        placeholders = ",".join(["?"] * len(vals))
        sql = f"INSERT OR REPLACE INTO stock_summary ({cols_sql}) VALUES ({placeholders})"
        self.conn.execute(sql, list(vals.values()))
        self.conn.commit()

    def get_summary(self, code):
        row = self.conn.execute("SELECT * FROM stock_summary WHERE code=?", [code]).fetchone()
        return dict(row) if row else None

    def get_all_summaries(self, codes=None):
        if codes:
            placeholders = ",".join(["?"] * len(codes))
            rows = self.conn.execute(
                f"SELECT * FROM stock_summary WHERE code IN ({placeholders})", codes).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM stock_summary ORDER BY code").fetchall()
        results = {}
        json_fields = {"dow_amp_json", "month_amp_json", "quarter_amp_json",
                       "month_ret_json", "month_win_json", "backtest_top5_json",
                       "dates_90_json", "amp_90_json", "monthly_list_json",
                       "dow_list_json", "month_ret_list_json", "month_win_list_json"}
        for row in rows:
            d = dict(row)
            for k in list(d.keys()):
                if k in json_fields and d[k]:
                    try:
                        d[k[:-5]] = json.loads(d[k])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results[d["code"]] = d
        return results

    def has_summary(self, code):
        r = self.conn.execute("SELECT 1 FROM stock_summary WHERE code=?", [code]).fetchone()
        return r is not None

    # ── Seasonality ────────────────────────────────

    def upsert_seasonality(self, code, s):
        rows = []
        for m, v in (s.get("month_amp") or {}).items():
            ret = (s.get("month_ret") or {}).get(m)
            win = (s.get("month_win") or {}).get(m)
            rows.append((code, "month", m, v, ret, win))
        for d, v in (s.get("dow_amp") or {}).items():
            rows.append((code, "dow", d, v, None, None))
        for q, v in (s.get("quarter_amp") or {}).items():
            rows.append((code, "quarter", q, v, None, None))
        self.conn.executemany(
            "INSERT OR REPLACE INTO seasonality VALUES (?,?,?,?,?,?)", rows)
        self.conn.commit()

    def get_seasonality(self, code, dimension):

        return pd.read_sql(
            "SELECT * FROM seasonality WHERE code=? AND dimension=? ORDER BY label",
            self.conn, params=[code, dimension])

    # ── Regime ─────────────────────────────────────

    def get_recent_regime(self, code, days=20):
        """计算单只股票近N日市场状态：震荡/单边+方向"""
        rows = self.conn.execute(
            "SELECT date, open, high, low, close, pct_change FROM daily WHERE code=? ORDER BY date DESC LIMIT ?",
            [code, days]
        ).fetchall()
        if len(rows) < 10:
            return {"label": "数据不足", "ratio": 0, "direction": "", "trend_pct": 0, "days": len(rows)}
        rows = list(reversed(rows))
        closes = [r["close"] for r in rows if r["close"]]
        pct_changes = [r["pct_change"] for r in rows if r["pct_change"] is not None]
        if not closes or not pct_changes:
            return {"label": "数据不足", "ratio": 0, "direction": "", "trend_pct": 0, "days": len(rows)}
        total_ret_pct = (closes[-1] - closes[0]) / closes[0] * 100
        path = sum(abs(pc) for pc in pct_changes)
        ratio = abs(total_ret_pct) / path if path > 0 else 0
        direction = "涨" if total_ret_pct > 0 else "跌"
        if ratio > 0.55:
            label = f"单边{direction}"
        elif ratio > 0.35:
            label = f"偏{direction}震荡"
        else:
            label = "窄幅震荡"
        return {"label": label, "ratio": round(ratio, 3), "direction": direction,
                "trend_pct": round(total_ret_pct, 2), "days": len(rows)}

    def get_all_recent_regimes(self, days=20):
        """批量获取所有股票的近N日市场状态"""
        codes = [r[0] for r in self.conn.execute("SELECT DISTINCT code FROM daily").fetchall()]
        result = {}
        for code in codes:
            result[code] = self.get_recent_regime(code, days)
        return result

    # ── Signal Log ────────────────────────────────

    def log_signal(self, code, date, composite_score, signal, amp_score=0, liq_score=0,
                   idio_score=0, regime_score=0, next_amplitude=None, next_pct_change=None):
        """记录单日信号及次日实际结果"""
        profit_est = None
        is_win = 0
        if next_amplitude is not None:
            profit_est = round(next_amplitude - 0.015, 4)  # 扣除~0.15%双边成本
            is_win = 1 if signal == 1 and profit_est > 0 else 0
        self.conn.execute("""
            INSERT OR REPLACE INTO signal_log
            (code, date, composite_score, signal, amp_score, liq_score, idio_score,
             regime_score, next_amplitude, next_pct_change, is_win, profit_est)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (code, date, composite_score, signal, amp_score, liq_score, idio_score,
              regime_score, next_amplitude, next_pct_change, is_win, profit_est))
        self.conn.commit()

    def backfill_signals(self, code, signal_df, daily_df):
        """批量回填信号日志：signal_df需含 composite_score/signal/各分项评分，daily_df需含 amplitude/pct_change"""
        import pandas as pd
        if signal_df.index.name is not None or not isinstance(signal_df.index, (range,)):
            signal_df = signal_df.reset_index()
        if daily_df.index.name is not None or not isinstance(daily_df.index, (range,)):
            daily_df = daily_df.reset_index()
        date_col_s = "date" if "date" in signal_df.columns else signal_df.columns[0]
        date_col_d = "date" if "date" in daily_df.columns else daily_df.columns[0]
        signal_df[date_col_s] = pd.to_datetime(signal_df[date_col_s])
        daily_df[date_col_d] = pd.to_datetime(daily_df[date_col_d])
        # shift: today's signal predicts tomorrow's amplitude
        daily_df["_next_amp"] = daily_df["amplitude"].shift(-1)
        daily_df["_next_pct"] = daily_df["pct_change"].shift(-1)
        merged = signal_df.merge(
            daily_df[[date_col_d, "_next_amp", "_next_pct"]],
            left_on=date_col_s, right_on=date_col_d, how="left")
        count = 0
        for _, row in merged.iterrows():
            d = pd.Timestamp(row[date_col_s]).strftime("%Y-%m-%d")
            sig = int(row.get("signal", 0))
            cs = float(row.get("composite_score", 0))
            self.log_signal(
                code, d, cs, sig,
                amp_score=float(row.get("score_amplitude", 0) or 0),
                liq_score=float(row.get("score_liquidity", 0) or 0),
                idio_score=float(row.get("score_idio", 0) or 0),
                regime_score=float(row.get("score_regime", 0) or 0),
                next_amplitude=float(row["_next_amp"]) if pd.notna(row.get("_next_amp")) else None,
                next_pct_change=float(row["_next_pct"]) if pd.notna(row.get("_next_pct")) else None,
            )
            count += 1
        self.conn.commit()
        return count

    def get_signal_log(self, code, limit=None):
        """获取某股票的信号日志"""
        sql = "SELECT * FROM signal_log WHERE code=? ORDER BY date DESC"
        params = [code]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return pd.read_sql(sql, self.conn, params=params)

    def get_signal_stats(self, code=None):
        """信号回测统计：胜率/累计收益等"""
        if code:
            rows = self.conn.execute(
                "SELECT * FROM signal_log WHERE code=? AND next_amplitude IS NOT NULL ORDER BY date",
                [code]).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM signal_log WHERE next_amplitude IS NOT NULL ORDER BY date").fetchall()
        if not rows:
            return None
        total = len(rows)
        signal_days = sum(1 for r in rows if r["signal"] == 1)
        wins = sum(1 for r in rows if r["is_win"] == 1)
        loss = sum(1 for r in rows if r["signal"] == 1 and r["is_win"] == 0)
        win_rate = round(wins / signal_days * 100, 1) if signal_days > 0 else 0
        total_profit = sum(r["profit_est"] or 0 for r in rows if r["signal"] == 1 and r["is_win"] == 1)
        total_loss = sum(abs(r["profit_est"] or 0) for r in rows if r["signal"] == 1 and r["is_win"] == 0)
        avg_profit_vals = [r["profit_est"] or 0 for r in rows if r["signal"] == 1]
        avg_p = round(sum(avg_profit_vals) / len(avg_profit_vals), 3) if avg_profit_vals else 0
        return {
            "total_days": total, "signal_days": signal_days, "signal_pct": round(signal_days/total*100,1) if total else 0,
            "wins": wins, "loss": loss, "win_rate": win_rate,
            "total_profit": round(total_profit, 3), "total_loss": round(total_loss, 3),
            "net_profit": round(total_profit - total_loss, 3),
            "avg_profit": avg_p,
        }

    # ── Utility ────────────────────────────────────

    def stats(self):
        codes = [r[0] for r in self.conn.execute("SELECT DISTINCT code FROM stock_summary").fetchall()]
        daily_count = self.conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
        return {"codes": codes, "stock_count": len(codes), "daily_rows": daily_count}

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    db = AnalysisDB()
    s = db.stats()
    print(f"数据库就绪: {s['stock_count']} 只股票, {s['daily_rows']} 行日线数据")
    if s["codes"]:
        print(f"股票: {', '.join(s['codes'])}")
    db.close()

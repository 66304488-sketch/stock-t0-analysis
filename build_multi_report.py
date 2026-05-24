#!/usr/bin/env python3
"""
多股票对比报告生成器
用法: python build_multi_report.py
输出: output/多股票对比报告.html
"""
import sys, os, json, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import STOCKS
from db import AnalysisDB
from generate_report import extract_stats, load_config

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STOCK_COLORS = [
    "#1a56db", "#dc2626", "#059669", "#7c3aed", "#0891b2",
    "#d97706", "#0d9488", "#be185d", "#2563eb", "#9333ea",
]


def get_processed_codes():
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    codes = []
    for code in STOCKS:
        if os.path.exists(os.path.join(processed_dir, f"{code}_features.parquet")):
            codes.append(code)
    return codes


def load_stock_data(code):
    """Load stock stats, prefer DB, fallback to extract_stats"""
    db = AnalysisDB()
    s = db.get_summary(code)
    if s:
        db.close()
        return s, STOCK_COLORS[len(STOCKS) % len(STOCK_COLORS)]

    print(f"  {code} 未在数据库中, 直接计算...")
    config = load_config()
    stats = extract_stats(code, config)
    db.upsert_summary(stats)
    db.upsert_seasonality(code, stats)
    db.close()
    return stats, STOCK_COLORS[len(STOCKS) % len(STOCK_COLORS)]


def make_compare_table(stocks_data):
    """Build comparison table HTML rows"""
    rows = []
    for i, (code, s) in enumerate(stocks_data.items()):
        color = STOCK_COLORS[i % len(STOCK_COLORS)]
        amp = s.get("amp_mean", "-")
        idio = round(s.get("idio_ratio_mean", 0) * 100, 1) if s.get("idio_ratio_mean") else "-"
        signal = s.get("signal_pct", "-")
        score = s.get("composite_score_mean", "-")
        garch = s.get("garch_persistence", "-")
        bt = s.get("backtest_top5")
        best_win = f"{bt[0]['win_rate']}%" if bt and len(bt) > 0 else "-"

        # 评价
        amp_v = float(amp) if amp != "-" else 0
        idio_v = float(idio) if idio != "-" else 0
        if amp_v >= 3.0 and idio_v >= 70:
            verdict = '<span class="badge badge-green">最佳</span>'
        elif amp_v >= 2.5 and idio_v >= 50:
            verdict = '<span class="badge badge-blue">推荐</span>'
        elif amp_v >= 2.2:
            verdict = '<span class="badge badge-yellow">谨慎</span>'
        else:
            verdict = '<span class="badge badge-red">不适合</span>'

        rows.append(f"""
        <tr>
          <td><span class="color-dot" style="background:{color}"></span>{s.get('name', code)} <small>({code})</small></td>
          <td>{s.get('industry', '-')}</td>
          <td class="num">{amp}%</td>
          <td class="num">{idio}%</td>
          <td class="num">{signal}%</td>
          <td class="num">{score}</td>
          <td class="num">{garch}</td>
          <td class="num">{best_win}</td>
          <td>{verdict}</td>
        </tr>""")
    return "\n".join(rows)


def make_js_stocks(stocks_data):
    """Generate the JS STOCKS array"""
    entries = []
    for i, (code, s) in enumerate(stocks_data.items()):
        color = STOCK_COLORS[i % len(STOCK_COLORS)]
        name = s.get("name", code)
        industry = s.get("industry", "")

        # Deserialize JSON fields
        def j(v):
            if isinstance(v, str):
                try: return json.loads(v)
                except: return v
            return v

        entry = {
            "code": code, "name": name, "industry": industry, "color": color,
            "summary": {
                "amp_mean": s.get("amp_mean"), "amp_median": s.get("amp_median"),
                "idio_ratio": round(s.get("idio_ratio_mean", 0) * 100, 1) if s.get("idio_ratio_mean") else 0,
                "signal_pct": s.get("signal_pct"),
                "composite_score": s.get("composite_score_mean"),
                "garch_persistence": s.get("garch_persistence"),
                "backtest_top5": j(s.get("backtest_top5_json")),
                "recent30_amp_mean": s.get("recent30_amp_mean"),
                "recent30_amp_max": s.get("recent30_amp_max"),
                "recent30_amp_min": s.get("recent30_amp_min"),
                "r_squared": s.get("r_squared_mean"),
                "signal_days": s.get("signal_days"),
                "rows": s.get("rows"),
            },
            "charts": {
                "ts90": {"labels": j(s.get("dates_90_json")), "amp": j(s.get("amp_90_json"))},
                "dow": {"labels": ["周一","周二","周三","周四","周五"], "data": j(s.get("dow_list_json"))},
                "month_amp": {"labels": [f"{i}月" for i in range(1,13)], "data": j(s.get("monthly_list_json"))},
                "month_ret": {"labels": [f"{i}月" for i in range(1,13)],
                               "data": j(s.get("month_ret_list_json")),
                               "win": j(s.get("month_win_list_json"))},
            }
        }
        # Compute win rate from backtest
        bt = j(s.get("backtest_top5_json"))
        entry["summary"]["best_win_rate"] = bt[0]["win_rate"] if bt and len(bt) > 0 else 0

        entries.append(entry)
    return json.dumps(entries, ensure_ascii=False)


def build_html(stocks_data):
    stocks_js = make_js_stocks(stocks_data)
    compare_rows = make_compare_table(stocks_data)

    # Build per-stock tabs
    tabs_html = ""
    panels_html = ""
    for i, (code, s) in enumerate(stocks_data.items()):
        name = s.get("name", code)
        color = STOCK_COLORS[i % len(STOCK_COLORS)]
        active = ' active' if i == 0 else ''
        tabs_html += f'<button class="tab-btn{active}" data-tab="{code}">{name}<br><small>{code}</small></button>\n'

        amp = s.get("amp_mean", "-")
        idio = round(s.get("idio_ratio_mean", 0) * 100, 1) if s.get("idio_ratio_mean") else "-"
        signal = s.get("signal_pct", "-")
        score = s.get("composite_score_mean", "-")
        garch = s.get("garch_persistence", "-")
        recent = s.get("recent30_amp_mean", "-")
        r2 = s.get("r_squared_mean", "-")

        panels_html += f"""
    <div id="tab-{code}" class="tab-panel">
      <div class="kpi-grid">
        <div class="kpi-card"><div class="kpi-label">日均振幅</div><div class="kpi-value" style="color:{color}">{amp}%</div><div class="kpi-sub">中位数 {s.get('amp_median', '-')}%</div></div>
        <div class="kpi-card"><div class="kpi-label">近30日振幅</div><div class="kpi-value" style="color:{color}">{recent}%</div><div class="kpi-sub">最高 {s.get('recent30_amp_max', '-')}% / 最低 {s.get('recent30_amp_min', '-')}%</div></div>
        <div class="kpi-card"><div class="kpi-label">特质波动占比</div><div class="kpi-value" style="color:{color}">{idio}%</div><div class="kpi-sub">R²_mkt = {r2}</div></div>
        <div class="kpi-card"><div class="kpi-label">GARCH 持续性</div><div class="kpi-value" style="color:{color}">{garch}</div><div class="kpi-sub">α+β</div></div>
        <div class="kpi-card"><div class="kpi-label">做T综合评分</div><div class="kpi-value" style="color:{color}">{score}</div><div class="kpi-sub">信号率 {signal}%</div></div>
        <div class="kpi-card"><div class="kpi-label">信号触发</div><div class="kpi-value" style="color:{color}">{signal}%</div><div class="kpi-sub">{s.get('signal_days', '-')} / {s.get('rows', '-')} 天</div></div>
      </div>
      <div class="chart-row">
        <div class="chart-card"><h3>近90日振幅走势</h3><div class="chart-wrap chart-md"><canvas id="ts90_{code}"></canvas></div></div>
        <div class="chart-card"><h3>周内效应</h3><div class="chart-wrap chart-md"><canvas id="dow_{code}"></canvas></div></div>
      </div>
      <div class="chart-row">
        <div class="chart-card"><h3>月度振幅</h3><div class="chart-wrap chart-md"><canvas id="monthAmp_{code}"></canvas></div></div>
        <div class="chart-card"><h3>月度涨跌</h3><div class="chart-wrap chart-md"><canvas id="monthRet_{code}"></canvas></div></div>
      </div>
    </div>"""

    today = time.strftime("%Y-%m-%d")
    stock_count = len(stocks_data)
    all_names = ", ".join(s.get("name", c) for c, s in stocks_data.items())
    date_range_start = min((s.get("data_start") or s.get("date_start") or "") for s in stocks_data.values())
    date_range_end = max((s.get("data_end") or s.get("date_end") or "") for s in stocks_data.values())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>A股做T多股票对比分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg: #f1f5f9; --card: #fff; --text: #1e293b; --text2: #64748b; --border: #e2e8f0; --r: 12px; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif; background:var(--bg); color:var(--text); line-height:1.6; -webkit-text-size-adjust:100%; }}
.masthead {{ background:linear-gradient(135deg,#0f172a,#1e3a5f,#1a56db); color:#fff; padding:40px 16px 32px; text-align:center; }}
.masthead h1 {{ font-size:clamp(1.3rem,3vw,1.8rem); font-weight:800; }}
.masthead .sub {{ font-size:0.85rem; opacity:0.7; margin-top:8px; }}
.container {{ max-width:1100px; margin:0 auto; padding:clamp(8px,1.5vw,16px); }}

.tabs {{ display:flex; gap:2px; margin-bottom:16px; border-bottom:2px solid var(--border); overflow-x:auto; -webkit-overflow-scrolling:touch; }}
.tab-btn {{ padding:8px 14px; border:none; background:none; font-size:0.85rem; font-weight:600; color:var(--text2); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px; white-space:nowrap; transition:all 0.2s; }}
.tab-btn small {{ font-size:0.7rem; opacity:0.6; }}
.tab-btn:hover {{ color:var(--text); }}
.tab-btn.active {{ color:#1a56db; border-bottom-color:#1a56db; }}
.tab-panel {{ display:none; }}
.tab-panel.active {{ display:block; }}

.kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:16px; }}
.kpi-card {{ background:var(--card); border-radius:var(--r); padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid var(--border); text-align:center; }}
.kpi-card .kpi-label {{ font-size:0.68rem; color:var(--text2); text-transform:uppercase; letter-spacing:0.3px; }}
.kpi-card .kpi-value {{ font-size:clamp(1.2rem,2vw,1.5rem); font-weight:800; margin:4px 0; }}
.kpi-card .kpi-sub {{ font-size:0.7rem; color:var(--text2); }}

.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }}
@media(max-width:700px){{ .chart-row {{ grid-template-columns:1fr; }} }}
.chart-card {{ background:var(--card); border-radius:var(--r); padding:14px 16px; box-shadow:0 1px 3px rgba(0,0,0,0.06); border:1px solid var(--border); }}
.chart-card h3 {{ font-size:0.9rem; font-weight:700; margin-bottom:8px; color:var(--text); }}
.chart-wrap {{ position:relative; width:100%; }}
.chart-md {{ height:260px; }}

.compare-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
.compare-table th {{ background:#f1f5f9; padding:10px 12px; text-align:left; font-weight:700; color:var(--text2); font-size:0.75rem; text-transform:uppercase; border-bottom:2px solid var(--border); }}
.compare-table td {{ padding:10px 12px; border-bottom:1px solid var(--border); }}
.compare-table .num {{ text-align:right; font-weight:600; font-feature-settings:'tnum'; }}
.compare-table tbody tr:hover {{ background:#f8fafc; }}
.compare-table .color-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}

.badge {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; }}
.badge-green {{ background:#d1fae5; color:#065f46; }}
.badge-blue {{ background:#dbeafe; color:#1e40af; }}
.badge-yellow {{ background:#fef3c7; color:#92400e; }}
.badge-red {{ background:#fee2e2; color:#991b1b; }}

.section-title {{ font-size:1.1rem; font-weight:700; margin:20px 0 12px; }}

footer {{ text-align:center; padding:30px 16px; font-size:0.75rem; color:#94a3b8; border-top:1px solid var(--border); margin-top:30px; }}
</style>
</head>
<body>

<div class="masthead">
  <h1>A股做T多股票对比分析报告</h1>
  <div class="sub">{stock_count} 只股票 · {date_range_start} ~ {date_range_end} · 数据来源: AKShare · 生成: {today}</div>
</div>

<div class="container">
  <div class="tabs">
    <button class="tab-btn active" data-tab="overview">对比总览</button>
    {tabs_html}
  </div>

  <div id="tab-overview" class="tab-panel active">
    <div class="chart-card" style="margin-bottom:14px;">
      <h3>{stock_count} 只股票核心指标对比</h3>
      <div style="overflow-x:auto;-webkit-overflow-scrolling:touch">
        <table class="compare-table">
          <thead><tr>
            <th>股票</th><th>行业</th><th>日均振幅</th><th>特质占比</th><th>信号率</th><th>综合评分</th><th>GARCH α+β</th><th>最佳胜率</th><th>评价</th>
          </tr></thead>
          <tbody>{compare_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="chart-row">
      <div class="chart-card"><h3>多维雷达对比</h3><div class="chart-wrap chart-md"><canvas id="radarChart"></canvas></div></div>
      <div class="chart-card"><h3>日均振幅排行</h3><div class="chart-wrap chart-md"><canvas id="ampBarChart"></canvas></div></div>
    </div>
    <div class="chart-row">
      <div class="chart-card"><h3>信号触发率排行</h3><div class="chart-wrap chart-md"><canvas id="signalBarChart"></canvas></div></div>
      <div class="chart-card"><h3>特质波动占比排行</h3><div class="chart-wrap chart-md"><canvas id="idioBarChart"></canvas></div></div>
    </div>
  </div>

  {panels_html}
</div>

<footer>A股个股波动做T套利分析系统 · 数据截止 {today} · 仅供参考不构成投资建议</footer>

<script>
const STOCKS = {stocks_js};

function rollingMean(arr, w) {{
  const o = [];
  for (let i = 0; i < arr.length; i++) {{
    if (i < w - 1) {{ o.push(null); continue; }}
    let s = 0;
    for (let j = i - w + 1; j <= i; j++) s += arr[j];
    o.push(s / w);
  }}
  return o;
}}

// ===== OVERVIEW CHARTS =====

// Radar
const radarCtx = document.getElementById('radarChart');
if (radarCtx) {{
  const maxAmp = Math.max(...STOCKS.map(s => s.summary.amp_mean));
  new Chart(radarCtx, {{
    type: 'radar',
    data: {{
      labels: ['日均振幅','特质占比','信号触发率','综合评分','GARCH持续性','最佳胜率'],
      datasets: STOCKS.map(s => ({{
        label: s.name,
        data: [
          (s.summary.amp_mean / (maxAmp || 5)) * 100,
          s.summary.idio_ratio,
          s.summary.signal_pct,
          s.summary.composite_score * 100,
          s.summary.garch_persistence * 100,
          s.summary.best_win_rate,
        ],
        borderColor: s.color,
        backgroundColor: s.color + '15',
        borderWidth: 2,
        pointRadius: 3,
      }})),
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ text: '做T多维度对比', display: true, font: {{ size: 13 }} }} }},
      scales: {{ r: {{ min: 0, max: 100, ticks: {{ stepSize: 20 }} }} }},
    }},
  }});
}}

// Bar helpers
function makeBarChart(id, label, getVal, title) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const sorted = [...STOCKS].sort((a, b) => getVal(b) - getVal(a));
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: sorted.map(s => s.name),
      datasets: [{{ label, data: sorted.map(getVal), backgroundColor: sorted.map(s => s.color + 'CC'), borderRadius: 4 }}],
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ title: {{ text: title, display: true, font: {{ size: 13 }} }}, legend: {{ display: false }} }},
      scales: {{ y: {{ beginAtZero: false }} }},
    }},
  }});
}}

makeBarChart('ampBarChart', '日均振幅 (%)', s => s.summary.amp_mean, '日均振幅排行');
makeBarChart('signalBarChart', '信号触发率 (%)', s => s.summary.signal_pct, '信号触发率排行');
makeBarChart('idioBarChart', '特质波动占比 (%)', s => s.summary.idio_ratio, '特质波动占比排行');

// ===== PER-STOCK CHARTS =====
STOCKS.forEach(s => {{
  const c = s.charts;

  // TS90
  const tsCtx = document.getElementById('ts90_' + s.code);
  if (tsCtx && c.ts90.labels) {{
    new Chart(tsCtx, {{
      type: 'line',
      data: {{
        labels: c.ts90.labels,
        datasets: [
          {{ label: '日振幅 (%)', data: c.ts90.amp, borderColor: s.color, borderWidth: 1, pointRadius: 0, tension: 0.2, fill: false }},
          {{ label: '20日均线', data: rollingMean(c.ts90.amp, 20), borderColor: '#dc2626', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: false }},
        ],
      }},
      options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ text: s.name + ' 近90日振幅走势', display: true, font: {{ size: 13 }} }} }} }},
    }});
  }}

  // DOW
  const dowCtx = document.getElementById('dow_' + s.code);
  if (dowCtx && c.dow.data) {{
    const mx = Math.max(...c.dow.data), mn = Math.min(...c.dow.data);
    new Chart(dowCtx, {{
      type: 'bar',
      data: {{
        labels: c.dow.labels,
        datasets: [{{ label: '日均振幅 (%)', data: c.dow.data,
          backgroundColor: c.dow.data.map(v => v === mx ? '#059669' : (v === mn ? '#dc2626' : s.color + 'BB')),
          borderRadius: 4 }}],
      }},
      options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ text: s.name + ' 周内振幅分布', display: true, font: {{ size: 13 }} }}, legend: {{ display: false }} }} }},
    }});
  }}

  // Month Amp
  const maCtx = document.getElementById('monthAmp_' + s.code);
  if (maCtx && c.month_amp.data) {{
    const mx2 = Math.max(...c.month_amp.data), mn2 = Math.min(...c.month_amp.data);
    new Chart(maCtx, {{
      type: 'bar',
      data: {{
        labels: c.month_amp.labels,
        datasets: [{{ label: '月均振幅 (%)', data: c.month_amp.data,
          backgroundColor: c.month_amp.data.map(v => v === mx2 ? '#059669' : (v === mn2 ? '#dc2626' : s.color + 'BB')),
          borderRadius: 4 }}],
      }},
      options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ text: s.name + ' 月度振幅分布', display: true, font: {{ size: 13 }} }}, legend: {{ display: false }} }} }},
    }});
  }}

  // Month Return
  const mrCtx = document.getElementById('monthRet_' + s.code);
  if (mrCtx && c.month_ret.data) {{
    new Chart(mrCtx, {{
      type: 'bar',
      data: {{
        labels: c.month_ret.labels,
        datasets: [{{ label: '月均涨跌幅 (%)', data: c.month_ret.data,
          backgroundColor: c.month_ret.data.map(v => v >= 0 ? '#059669' : '#dc2626'),
          borderRadius: 4 }}],
      }},
      options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ text: s.name + ' 月度涨跌分布', display: true, font: {{ size: 13 }} }}, legend: {{ display: false }} }} }},
    }});
  }}
}});

// ===== TAB SWITCHING =====
function switchTab(tabId) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('tab-' + tabId);
  if (panel) panel.classList.add('active');
  const btn = document.querySelector('[data-tab="' + tabId + '"]');
  if (btn) btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}

document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', function() {{ switchTab(this.dataset.tab); }});
}});
</script>
</body>
</html>"""
    return html


def main():
    codes = get_processed_codes()
    print(f"已处理股票: {len(codes)} 只")

    # Load all data from DB (with fallback)
    db = AnalysisDB()
    summaries = db.get_all_summaries(codes)
    db.close()

    if len(summaries) < len(codes):
        missing = [c for c in codes if c not in summaries]
        print(f"部分股票未同步到DB，直接计算: {missing}")
        config = load_config()
        db2 = AnalysisDB()
        for code in missing:
            try:
                stats = extract_stats(code, config)
                db2.upsert_summary(stats)
                db2.upsert_seasonality(code, stats)
                summaries[code] = stats
            except Exception as e:
                print(f"  {code} 失败: {e}")
        db2.close()

    # Build and sort: first by amp_mean desc, then add any missing stocks
    stocks_data = {}
    for code in codes:
        if code in summaries:
            s = summaries[code]
            stocks_data[code] = s

    print(f"生成报告 ({len(stocks_data)} 只股票)...")
    html = build_html(stocks_data)

    output_path = os.path.join(PROJECT_ROOT, "output", "多股票对比报告.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"报告已保存: {output_path} ({size_kb:.0f}KB)")
    return output_path


if __name__ == "__main__":
    t0 = time.time()
    path = main()
    print(f"耗时: {time.time()-t0:.1f}s")
    os.system(f"open '{path}'")

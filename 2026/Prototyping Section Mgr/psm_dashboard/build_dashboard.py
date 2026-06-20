#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
試作設備DXダッシュボード (スモールスケール版)

MemoPSM.md のMVP3本を1枚のHTMLに集約する。
  MVP1: 試作設備台帳 + 更新優先度スコアリング
  MVP2: 工程別ボトルネック可視化
  MVP3: 研究テーマ × 設備マップ / 投資候補ランキング

特徴:
  - Python標準ライブラリのみで動作 (pandas等のインストール不要)
  - グラフはPlotly.js(CDN)を埋め込んで描画
  - 入力はすべて data/*.csv 。CSVを書き換えれば再生成できる

使い方:
  python3 build_dashboard.py
  -> reports/dashboard.html を生成 (出力先は --out で変更可)
"""

import argparse
import csv
import json
import os
from datetime import datetime

# 更新優先度スコアの重み (MemoPSM.md のスコア式に準拠)
WEIGHTS = {
    "研究テーマ貢献度": 0.25,
    "ボトルネック度": 0.25,
    "老朽化度": 0.20,
    "故障頻度": 0.15,
    "安全リスク": 0.15,
}

HERE = os.path.dirname(os.path.abspath(__file__))


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def priority_label(score):
    if score >= 4.0:
        return "高"
    if score >= 3.0:
        return "中"
    return "低"


def compute_equipment(rows):
    """設備ごとに更新優先度スコア(5点満点換算)を計算する。"""
    result = []
    for r in rows:
        score = sum(to_float(r[k]) * w for k, w in WEIGHTS.items())
        item = dict(r)
        item["更新優先度スコア"] = round(score, 2)
        item["更新優先度"] = priority_label(score)
        result.append(item)
    result.sort(key=lambda x: x["更新優先度スコア"], reverse=True)
    return result


def compute_process(rows):
    """工程ごとの負荷率(=依頼件数/処理能力)を計算する。"""
    result = []
    for r in rows:
        cap = to_float(r["月間処理能力"], 1.0) or 1.0
        load = to_float(r["月間依頼件数"]) / cap * 100.0
        item = dict(r)
        item["負荷率"] = round(load, 1)
        if load >= 100:
            item["判定"] = "ボトルネック"
        elif load >= 85:
            item["判定"] = "逼迫"
        else:
            item["判定"] = "余力あり"
        result.append(item)
    result.sort(key=lambda x: x["負荷率"], reverse=True)
    return result


def color_for_load(load):
    if load >= 100:
        return "#d62728"
    if load >= 85:
        return "#ff7f0e"
    return "#2ca02c"


def color_for_priority(label):
    return {"高": "#d62728", "中": "#ff7f0e", "低": "#2ca02c"}.get(label, "#1f77b4")


def table_html(headers, rows_2d):
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = []
    for row in rows_2d:
        tds = "".join(f"<td>{c}</td>" for c in row)
        trs.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def build(equipment, process, themes):
    # --- アラートサマリ ---
    n_update = sum(1 for e in equipment if e["更新優先度"] == "高")
    n_fault = sum(1 for e in equipment if to_float(e["故障頻度"]) >= 3)
    n_safety = sum(1 for e in equipment if to_float(e["安全リスク"]) >= 4)
    n_bottleneck = sum(1 for p in process if p["判定"] == "ボトルネック")

    # --- グラフ1: 更新優先度ランキング ---
    eq_names = [e["設備名"] for e in equipment]
    eq_scores = [e["更新優先度スコア"] for e in equipment]
    eq_colors = [color_for_priority(e["更新優先度"]) for e in equipment]
    fig_priority = {
        "data": [{
            "type": "bar", "orientation": "h",
            "x": list(reversed(eq_scores)),
            "y": list(reversed(eq_names)),
            "marker": {"color": list(reversed(eq_colors))},
            "text": list(reversed(eq_scores)), "textposition": "auto",
        }],
        "layout": {"title": "設備 更新優先度スコア (5点満点)",
                   "xaxis": {"range": [0, 5]}, "margin": {"l": 140}, "height": 360},
    }

    # --- グラフ2: 工程別負荷率 ---
    pr_names = [p["工程"] for p in process]
    pr_loads = [p["負荷率"] for p in process]
    pr_colors = [color_for_load(l) for l in pr_loads]
    fig_load = {
        "data": [{
            "type": "bar",
            "x": pr_names, "y": pr_loads,
            "marker": {"color": pr_colors},
            "text": [f"{l}%" for l in pr_loads], "textposition": "auto",
        }],
        "layout": {"title": "工程別 負荷率 (100%超=ボトルネック)",
                   "yaxis": {"title": "負荷率 (%)"},
                   "shapes": [{"type": "line", "x0": -0.5, "x1": len(pr_names) - 0.5,
                               "y0": 100, "y1": 100,
                               "line": {"color": "#d62728", "dash": "dash"}}],
                   "height": 360},
    }

    # --- グラフ3: 工程別 待ち日数 ---
    fig_wait = {
        "data": [{
            "type": "bar",
            "x": pr_names, "y": [to_float(p["平均待ち日数"]) for p in process],
            "marker": {"color": "#1f77b4"},
        }],
        "layout": {"title": "工程別 平均待ち日数",
                   "yaxis": {"title": "日数"}, "height": 320},
    }

    # --- テーブル: 設備台帳 ---
    eq_headers = ["設備名", "工程", "老朽化度", "稼働率", "故障頻度",
                  "研究テーマ貢献度", "ボトルネック度", "安全リスク",
                  "更新優先度スコア", "更新優先度"]
    eq_rows = [[e[h] for h in eq_headers] for e in equipment]
    eq_table = table_html(eq_headers, eq_rows)

    # --- テーブル: テーマ×設備マップ ---
    th_headers = ["研究テーマ", "必要工程", "使用設備", "不足設備", "設備制約度"]
    th_rows = [[t[h] for h in th_headers] for t in themes]
    theme_table = table_html(th_headers, th_rows)

    # --- 投資候補ランキング (不足設備ベース + ボトルネック工程) ---
    top_bottleneck = [p["工程"] for p in process if p["判定"] == "ボトルネック"]
    invest = []
    for t in themes:
        if t["不足設備"]:
            invest.append((t["不足設備"], t["研究テーマ"], t["設備制約度"]))
    invest.sort(key=lambda x: {"大": 0, "中": 1, "小": 2}.get(x[2], 3))
    invest_items = "".join(
        f"<li><b>{name}</b> — テーマ: {theme} / 制約度: {lvl}</li>"
        for name, theme, lvl in invest)

    figs_json = json.dumps({
        "priority": fig_priority,
        "load": fig_load,
        "wait": fig_wait,
    }, ensure_ascii=False)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>試作設備DXダッシュボード</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: -apple-system, "Hiragino Sans", "Yu Gothic", sans-serif;
         margin: 0; background: #f4f6f8; color: #222; }}
  header {{ background: #0b3d61; color: #fff; padding: 18px 28px; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .sub {{ font-size: 12px; opacity: .8; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }}
  .card {{ flex: 1 1 180px; background: #fff; border-radius: 10px; padding: 16px;
          box-shadow: 0 1px 3px rgba(0,0,0,.1); border-left: 5px solid #0b3d61; }}
  .card .num {{ font-size: 30px; font-weight: bold; }}
  .card .lbl {{ font-size: 13px; color: #666; }}
  .card.red {{ border-color: #d62728; }}
  .card.orange {{ border-color: #ff7f0e; }}
  section {{ background: #fff; border-radius: 10px; padding: 18px; margin-bottom: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  section h2 {{ font-size: 16px; border-bottom: 2px solid #0b3d61;
               padding-bottom: 6px; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: center; }}
  th {{ background: #0b3d61; color: #fff; }}
  tbody tr:nth-child(even) {{ background: #f7f9fb; }}
  ul {{ line-height: 1.8; }}
</style>
</head>
<body>
<header>
  <h1>試作設備DXダッシュボード</h1>
  <div class="sub">Prototyping Section Manager / 生成日時: {now}</div>
</header>
<div class="wrap">

  <div class="cards">
    <div class="card red"><div class="num">{n_update}</div><div class="lbl">更新推奨 (優先度:高)</div></div>
    <div class="card orange"><div class="num">{n_fault}</div><div class="lbl">故障リスク高</div></div>
    <div class="card orange"><div class="num">{n_safety}</div><div class="lbl">安全対応要</div></div>
    <div class="card red"><div class="num">{n_bottleneck}</div><div class="lbl">ボトルネック工程</div></div>
  </div>

  <section>
    <h2>1. 設備投資優先順位 (MVP3)</h2>
    <div id="priority"></div>
  </section>

  <section>
    <h2>2. 工程別ボトルネック (MVP2)</h2>
    <div id="load"></div>
    <div id="wait"></div>
  </section>

  <section>
    <h2>3. 試作設備台帳 (MVP1)</h2>
    {eq_table}
  </section>

  <section>
    <h2>4. 研究テーマ × 設備マップ</h2>
    {theme_table}
  </section>

  <section>
    <h2>5. 投資候補ランキング (不足設備)</h2>
    <ol>{invest_items}</ol>
  </section>

</div>
<script>
  var FIGS = {figs_json};
  var cfg = {{responsive: true}};
  Plotly.newPlot('priority', FIGS.priority.data, FIGS.priority.layout, cfg);
  Plotly.newPlot('load', FIGS.load.data, FIGS.load.layout, cfg);
  Plotly.newPlot('wait', FIGS.wait.data, FIGS.wait.layout, cfg);
</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="試作設備DXダッシュボードHTMLを生成")
    ap.add_argument("--data", default=os.path.join(HERE, "data"),
                    help="入力CSVフォルダ")
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "dashboard.html"),
                    help="出力HTMLパス")
    args = ap.parse_args()

    equipment = compute_equipment(read_csv(os.path.join(args.data, "equipment.csv")))
    process = compute_process(read_csv(os.path.join(args.data, "process_load.csv")))
    themes = read_csv(os.path.join(args.data, "theme_map.csv"))

    html = build(equipment, process, themes)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] ダッシュボードを生成しました: {args.out}")
    print(f"  設備数: {len(equipment)} / 工程数: {len(process)} / テーマ数: {len(themes)}")
    bottleneck = [p['工程'] for p in process if p['判定'] == 'ボトルネック']
    print(f"  ボトルネック工程: {', '.join(bottleneck) if bottleneck else 'なし'}")


if __name__ == "__main__":
    main()

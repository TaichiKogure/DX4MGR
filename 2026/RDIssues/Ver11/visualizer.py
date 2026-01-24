import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
import pandas as pd

# Prefer an installed Japanese-capable font to avoid tofu in charts.
def _pick_japanese_font():
    candidates = [
        "Yu Gothic",
        "Meiryo",
        "BIZ UDGothic",
        "MS Gothic",
        "MS PGothic",
        "Hiragino Sans",
        "IPAexGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_jp_font = _pick_japanese_font()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    _jp_font if _jp_font else "DejaVu Sans",
    "DejaVu Sans",
    "sans-serif",
]
plt.rcParams["axes.unicode_minus"] = False

def plot_comparison_with_ci(comparison_results, title="シナリオ比較 (95%信頼区間)"):
    """
    平均値と信頼区間を表示する棒グラフ
    comparison_results: { label: {"mean": m, "ci": [low, high]}, ... }
    """
    labels = list(comparison_results.keys())
    means = [comparison_results[l]["mean"] for l in labels]
    lows = [comparison_results[l]["ci"][0] for l in labels]
    highs = [comparison_results[l]["ci"][1] for l in labels]
    
    yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, means, yerr=yerr, capsize=5, color='skyblue', alpha=0.8)
    
    plt.title(title)
    plt.ylabel("スループット (VP/day)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", 
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()

def plot_doe_analysis(doe_df, target_col="throughput", title="DOE (実験計画法) 感度分析"):
    """
    パラメータとターゲット変数の相関を示す散布図
    """
    param_cols = [c for c in doe_df.columns if c != target_col]
    n_params = len(param_cols)
    
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 5), sharey=True)
    if n_params == 1: axes = [axes]
    
    for i, col in enumerate(param_cols):
        sns.regplot(x=col, y=target_col, data=doe_df, ax=axes[i], scatter_kws={'alpha':0.5})
        axes[i].set_title(f"{col} vs {target_col}")
        axes[i].grid(True, alpha=0.3)
        
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def plot_quality_gate_status(gate_results, title="検証ゲート (Quality Gates) ステータス"):
    """
    検証ゲートの結果をテーブル形式で可視化
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    rows = gate_results["gates"]
    data = [[r["name"], f"{r['value']:.4f}", r["threshold"], r["status"]] for r in rows]
    
    colors = []
    for r in rows:
        colors.append(['white', 'white', 'white', 'lightgreen' if r["status"] == "PASS" else 'tomato'])
    
    table = ax.table(cellText=data, 
                    colLabels=["ゲート名", "現在値", "閾値", "判定"],
                    cellColours=colors,
                    loc='center',
                    cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    overall_color = 'green' if gate_results["overall_status"] == "PASS" else 'red'
    plt.title(f"{title}\n総合判定: {gate_results['overall_status']}", 
              color=overall_color, fontsize=14, fontweight='bold')
    plt.tight_layout()

def plot_gate_wait_heatmap(gate_stats_by_scenario: dict, title="ゲート別 平均待ち時間ヒートマップ"):
    """
    gate_stats_by_scenario:
      { scenario_name: [ {"node_id": "...", "avg_wait_time": ...}, ... ], ... }
    """
    rows = []
    for scenario, stats in gate_stats_by_scenario.items():
        for s in stats:
            rows.append({
                "Scenario": scenario,
                "Gate": s.get("node_id"),
                "AvgWait": s.get("avg_wait_time", 0.0)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    pivot = df.pivot_table(index="Scenario", columns="Gate", values="AvgWait", aggfunc="mean").fillna(0.0)

    plt.figure(figsize=(max(8, 1.2 * pivot.shape[1]), max(4, 0.7 * pivot.shape[0])))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title(title)
    plt.xlabel("Gate")
    plt.ylabel("Scenario")
    plt.tight_layout()

def plot_gate_wip_heatmap(avg_wip_by_node_by_scenario: dict, title="ゲート別 平均WIPヒートマップ"):
    """
    avg_wip_by_node_by_scenario:
      { scenario_name: { node_id: avg_wip, ... }, ... }
    """
    rows = []
    for scenario, d in avg_wip_by_node_by_scenario.items():
        for node_id, v in (d or {}).items():
            rows.append({
                "Scenario": scenario,
                "Gate": node_id,
                "AvgWIP": float(v)
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    pivot = df.pivot_table(index="Scenario", columns="Gate", values="AvgWIP", aggfunc="mean").fillna(0.0)

    plt.figure(figsize=(max(8, 1.2 * pivot.shape[1]), max(4, 0.7 * pivot.shape[0])))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="PuBu")
    plt.title(title)
    plt.xlabel("Gate")
    plt.ylabel("Scenario")
    plt.tight_layout()
def plot_wait_time_distribution(results_dict, title="待ち時間の分布 (P90強調)"):
    data = []
    for label, waits in results_dict.items():
        for w in waits:
            data.append({"Case": label, "Wait Time": w})
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Case", y="Wait Time", data=df, inner=None, color="lightgray")
    sns.stripplot(x="Case", y="Wait Time", data=df, alpha=0.3, jitter=True)
    for i, label in enumerate(results_dict.keys()):
        waits = results_dict[label]
        if not waits: continue
        p90 = np.percentile(waits, 90)
        p95 = np.percentile(waits, 95)
        plt.hlines(p90, i-0.2, i+0.2, colors='red', linestyles='solid')
        plt.hlines(p95, i-0.2, i+0.2, colors='darkred', linestyles='dashed')
        plt.text(i+0.22, p90, f"P90:{p90:.1f}d", color='red', va='center')
    plt.title(title)
    plt.ylabel("待ち時間 (日)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_ccdf(results_dict, title="超過確率カーブ (CCDF)"):
    plt.figure(figsize=(10, 6))
    for label, waits in results_dict.items():
        if not waits: continue
        sorted_waits = np.sort(waits)
        ccdf = 1.0 - np.arange(1, len(sorted_waits) + 1) / len(sorted_waits)
        plt.step(sorted_waits, ccdf, label=label, where='post')
    plt.title(title)
    plt.xlabel("待ち時間 x (日)")
    plt.ylabel("P(Wait Time > x)")
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()

def plot_rework_distribution(results_dict, title="差し戻し回数の分布"):
    """
    各シナリオの差し戻し回数の分布を可視化
    results_dict: { label: [rework_count1, rework_count2, ...], ... }
    """
    data = []
    for label, counts in results_dict.items():
        for c in counts:
            data.append({"Case": label, "Rework Count": c})
    
    if not data:
        return
        
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    
    # 箱ひげ図とストリッププロットの組み合わせ
    sns.boxplot(x="Case", y="Rework Count", data=df, color="lightgray", showfliers=False)
    sns.stripplot(x="Case", y="Rework Count", data=df, alpha=0.3, jitter=True)
    
    for i, label in enumerate(results_dict.keys()):
        counts = results_dict[label]
        if not counts: continue
        avg = np.mean(counts)
        plt.text(i, avg, f"Avg:{avg:.2f}", color='blue', va='bottom', ha='center', fontweight='bold')
    
    plt.title(title)
    plt.ylabel("差し戻し回数")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_waterfall(base_val, skills_inc, dep_dec, final_val, title="実効処理能力の内訳"):
    labels = ["基準能力", "スキル向上", "依存/調整ロス", "実効能力"]
    values = [base_val, skills_inc, -dep_dec, final_val]
    current = 0
    bottom = []
    for v in values:
        if v >= 0:
            bottom.append(current)
            current += v
        else:
            current += v
            bottom.append(current)
    bottom[-1] = 0
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    bars = plt.bar(labels, values, bottom=bottom, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_y() + yval/2, 
                 f"{yval:+.2f}" if bar.get_label() != "実効能力" else f"{yval:.2f}",
                 ha='center', va='center', color='white', fontweight='bold')
    plt.title(title)
    plt.ylabel("処理能力 (VP/day)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

# Step 7: 新規追加の可視化関数
def plot_wip_time_series(wip_history_by_scenario: dict, title="WIP (仕掛品) の時系列推移"):
    """
    wip_history_by_scenario: { scenario_name: [ {"time": t, "total_wip": w}, ... ], ... }
    """
    plt.figure(figsize=(12, 6))
    for label, history in wip_history_by_scenario.items():
        if not history: continue
        df = pd.DataFrame(history)
        plt.plot(df["time"], df["total_wip"], label=label, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("経過日数")
    plt.ylabel("WIP数")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def plot_rework_weight_distribution(weights_dict, title="差し戻し重みの分布"):
    data = []
    for label, weights in weights_dict.items():
        for w in weights:
            data.append({"Case": label, "Weight": w})
    if not data: return
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Case", y="Weight", data=df, color="lightgray")
    plt.title(title)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

def plot_proliferated_tasks_distribution(tasks_dict, title="増殖した小実験数の分布"):
    data = []
    for label, counts in tasks_dict.items():
        for c in counts:
            data.append({"Case": label, "Count": c})
    if not data: return
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="Count", hue="Case", multiple="dodge", discrete=True, shrink=0.8)
    plt.title(title)
    plt.xlabel("増殖タスク数")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

def plot_scorecard(scenario_metrics: dict, baseline_name: str, title="シナリオ性能スコアカード"):
    """
    各シナリオの主要指標を並べたスコアカード（表形式の画像）を生成。
    指標: LT(P90), TP, AvgWIP, Rework
    """
    import pandas as pd
    
    if baseline_name not in scenario_metrics:
        return
    
    base = scenario_metrics[baseline_name]
    
    rows = []
    for name, m in scenario_metrics.items():
        row = {
            "Scenario": name,
            "LT(P90)": f"{m['p90']:.1f}",
            "LT(P90) vs Base": f"{m['p90'] / (base['p90'] + 1e-9):.2f}x",
            "Throughput": f"{m['tp']:.3f}",
            "TP vs Base": f"{m['tp'] / (base['tp'] + 1e-9):.2f}x",
            "AvgWIP": f"{m['wip']:.1f}",
            "AvgWIP vs Base": f"{m['wip'] / (base['wip'] + 1e-9):.2f}x",
            "Rework": f"{m['rework']:.2f}",
            "Rework vs Base": f"{m['rework'] / (base['rework'] + 1e-9):.2f}x"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(rows)))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # ヘッダーの色設定
    for j in range(len(df.columns)):
        table[0, j].set_facecolor('#40466e')
        table[0, j].get_text().set_color('white')
        table[0, j].get_text().set_weight('bold')

    plt.title(title, pad=30, fontsize=14, fontweight='bold')
    plt.tight_layout()

def plot_job_gantt(job_log_df, title="ジョブ進行（サンプル）", max_jobs=30):
    """
    Jobの処理履歴を簡易ガントで可視化（先頭 max_jobs 件）
    job_log_df: job_details_{scenario}.csv の DataFrame
    """
    if job_log_df is None or job_log_df.empty:
        return
    required = {"job_id", "node_id", "start_time", "effective_duration"}
    if not required.issubset(set(job_log_df.columns)):
        return

    df = job_log_df.dropna(subset=["job_id", "node_id", "start_time", "effective_duration"]).copy()
    if df.empty:
        return

    df = df.sort_values("start_time")
    job_ids = df["job_id"].drop_duplicates().head(max_jobs).tolist()
    df = df[df["job_id"].isin(job_ids)]
    if df.empty:
        return

    y_pos = {job_id: i for i, job_id in enumerate(job_ids)}
    nodes = df["node_id"].dropna().unique().tolist()
    palette = sns.color_palette("tab10", n_colors=max(1, len(nodes)))
    node_colors = {node: palette[i % len(palette)] for i, node in enumerate(nodes)}

    plt.figure(figsize=(12, max(4, 0.35 * len(job_ids))))
    for _, row in df.iterrows():
        y = y_pos[row["job_id"]]
        start = float(row["start_time"])
        duration = float(row["effective_duration"])
        wait = row.get("wait_time", np.nan)

        if pd.notna(wait) and wait > 0:
            wait_start = max(0.0, start - float(wait))
            plt.barh(y, float(wait), left=wait_start, height=0.5, color="lightgray", alpha=0.5)

        plt.barh(y, duration, left=start, height=0.5,
                 color=node_colors.get(row["node_id"]), alpha=0.9)

    plt.yticks(range(len(job_ids)), [str(j) for j in job_ids])
    plt.xlabel("時間(日)")
    plt.ylabel("ジョブ")
    plt.title(title)
    legend_items = [Patch(color=c, label=n) for n, c in node_colors.items()]
    legend_items.append(Patch(color="lightgray", label="待ち(キュー)"))
    plt.legend(handles=legend_items, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

def plot_gate_wait_heatmap_by_time(job_log_df, bin_days=30, title="ゲート別待ち時間（時間帯）"):
    """
    ゲート別の待ち時間を時間帯ごとのヒートマップで表示
    """
    if job_log_df is None or job_log_df.empty:
        return
    required = {"node_id", "start_time", "wait_time"}
    if not required.issubset(set(job_log_df.columns)):
        return

    df = job_log_df.dropna(subset=["node_id", "start_time", "wait_time"]).copy()
    if df.empty:
        return

    bin_days = float(bin_days) if bin_days else 30.0
    df["time_bin"] = (df["start_time"] // bin_days) * bin_days

    pivot = df.pivot_table(index="time_bin", columns="node_id", values="wait_time", aggfunc="mean").fillna(0.0)
    if pivot.empty:
        return

    plt.figure(figsize=(max(8, 1.2 * pivot.shape[1]), max(4, 0.4 * pivot.shape[0])))
    sns.heatmap(pivot, annot=False, fmt=".1f", cmap="YlOrRd")
    plt.title(f"{title}（bin={int(bin_days)}日）")
    plt.xlabel("Gate")
    plt.ylabel("時間帯(開始日)")
    plt.tight_layout()

def plot_gate_wait_distribution(job_log_df, title="ゲート別待ち時間分布"):
    """
    ゲート別の待ち時間分布（箱ひげ＋散布）
    """
    if job_log_df is None or job_log_df.empty:
        return
    required = {"node_id", "wait_time"}
    if not required.issubset(set(job_log_df.columns)):
        return

    df = job_log_df.dropna(subset=["node_id", "wait_time"]).copy()
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="node_id", y="wait_time", data=df, color="lightgray", showfliers=False)
    sns.stripplot(x="node_id", y="wait_time", data=df, alpha=0.3, jitter=True)
    plt.title(title)
    plt.xlabel("Gate")
    plt.ylabel("待ち時間(日)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

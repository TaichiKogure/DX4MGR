import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 日本語フォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'IPAexGothic', 'sans-serif']

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

# Ver4までの関数を継承・互換性維持
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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 日本語フォント設定 (環境によって異なるが、一般的なものを試す)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans', 'IPAexGothic', 'sans-serif']

def plot_wait_time_distribution(results_dict, title="待ち時間の分布 (P90強調)"):
    """
    results_dict: {label: [wait_times, ...]}
    """
    data = []
    for label, waits in results_dict.items():
        for w in waits:
            data.append({"Case": label, "Wait Time": w})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Case", y="Wait Time", data=df, inner=None, color="lightgray")
    sns.stripplot(x="Case", y="Wait Time", data=df, alpha=0.3, jitter=True)
    
    # P90, P95の線を引く
    for i, label in enumerate(results_dict.keys()):
        waits = results_dict[label]
        if not waits: continue
        p50 = np.percentile(waits, 50)
        p90 = np.percentile(waits, 90)
        p95 = np.percentile(waits, 95)
        
        plt.hlines(p90, i-0.2, i+0.2, colors='red', linestyles='solid', label="P90" if i==0 else "")
        plt.hlines(p95, i-0.2, i+0.2, colors='darkred', linestyles='dashed', label="P95" if i==0 else "")
        plt.text(i+0.22, p90, f"P90:{p90:.1f}d", color='red', va='center')
        plt.text(i+0.22, p95, f"P95:{p95:.1f}d", color='darkred', va='center')

    plt.title(title)
    plt.ylabel("待ち時間 (日)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

def plot_ccdf(results_dict, title="超過確率カーブ (CCDF)"):
    """
    results_dict: {label: [wait_times, ...]}
    """
    plt.figure(figsize=(10, 6))
    
    for label, waits in results_dict.items():
        if not waits: continue
        sorted_waits = np.sort(waits)
        ccdf = 1.0 - np.arange(1, len(sorted_waits) + 1) / len(sorted_waits)
        plt.step(sorted_waits, ccdf, label=label, where='post')
    
    plt.title(title)
    plt.xlabel("待ち時間 x (日)")
    plt.ylabel("P(Wait Time > x)")
    plt.yscale('log') # 対数スケールが見やすい
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()

def plot_wip_timeseries(results_logs_dict, title="WIP (滞留数) の推移"):
    """
    results_logs_dict: {label: [(t, wip), ...]}
    """
    plt.figure(figsize=(12, 6))
    
    for label, history in results_logs_dict.items():
        t = [x[0] for x in history]
        wip = [x[1] for x in history]
        plt.plot(t, wip, label=label, alpha=0.8)
    
    plt.title(title)
    plt.xlabel("経過日数")
    plt.ylabel("WIP (案件数)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

def plot_waterfall(base_val, skills_inc, dep_dec, final_val, title="実効処理能力の内訳"):
    """
    簡易版ウォーターフォール図
    """
    labels = ["基準能力", "スキル向上", "依存/調整ロス", "実効能力"]
    values = [base_val, skills_inc, -dep_dec, final_val]
    
    # 累積値を計算
    current = 0
    bottom = []
    for v in values:
        if v >= 0:
            bottom.append(current)
            current += v
        else:
            current += v
            bottom.append(current)
    
    # 実効能力(最後)は0から立ち上げる
    bottom[-1] = 0
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange']
    bars = plt.bar(labels, values, bottom=bottom, color=colors)
    
    # 値を表示
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_y() + yval/2, 
                 f"{yval:+.2f}" if bar.get_label() != "実効能力" else f"{yval:.2f}",
                 ha='center', va='center', color='white', fontweight='bold')

    plt.title(title)
    plt.ylabel("処理能力 (VP/day)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_tornado(labels, changes, title="感度分析 (トルネードチャート)"):
    """
    labels: パラメータ名
    changes: 各パラメータを変化させた時のスループット変化量
    """
    df = pd.DataFrame({"Parameter": labels, "Change": changes})
    df = df.reindex(df.Change.abs().sort_values().index) # 絶対値でソート
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'blue' for x in df.Change]
    plt.barh(df.Parameter, df.Change, color=colors, alpha=0.7)
    plt.axvline(0, color='black', linewidth=1)
    
    plt.title(title)
    plt.xlabel("スループットの変化量 (VP/day)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

def plot_executive_summary(results_dict, threshold=10, title="上層部向けサマリ: 遅延リスクと納期保証"):
    """
    上層部向けに、P90と閾値超過確率をまとめた表形式+棒グラフ
    """
    rows = []
    for label, waits in results_dict.items():
        if not waits: continue
        p50 = np.percentile(waits, 50)
        p90 = np.percentile(waits, 90)
        risk = np.mean(np.array(waits) > threshold) * 100
        rows.append({"ケース": label, "平均納期(P50)": p50, "安心納期(P90)": p90, "遅延リスク(%)": risk})
    
    df = pd.DataFrame(rows)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 納期比較 (P50 vs P90)
    df_melt = df.melt(id_vars="ケース", value_vars=["平均納期(P50)", "安心納期(P90)"], var_name="指標", value_name="日数")
    sns.barplot(x="ケース", y="日数", hue="指標", data=df_melt, ax=ax1)
    ax1.set_title("納期予測 (P50:平均 / P90:ほぼ確実)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. 遅延リスク (信号機)
    colors = ['green' if x < 5 else 'orange' if x < 20 else 'red' for x in df["遅延リスク(%)"]]
    sns.barplot(x="ケース", y="遅延リスク(%)", data=df, ax=ax2, palette=colors)
    ax2.set_title(f"炎上リスク (待ち時間が{threshold}日を超える確率)")
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

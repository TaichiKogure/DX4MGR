import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from simulator_v4 import simulate_flow_v3, run_monte_carlo
import visualizer_v4 as viz

# 出力ディレクトリ
OUT_DIR = "2026/RDIssues/Ver4/output"
os.makedirs(OUT_DIR, exist_ok=True)

def load_scenarios(file_path):
    return pd.read_csv(file_path)

def run_all_scenarios():
    csv_path = "2026/RDIssues/Ver4/scenarios.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_scenarios = load_scenarios(csv_path)
    print(f"{len(df_scenarios)} 個のシナリオを読み込みました。")

    all_results = {}
    wait_times_dict = {}
    wip_logs_dict = {}

    for _, row in df_scenarios.iterrows():
        name = row['scenario_name']
        print(f"シナリオ実行中: {name} ...")
        
        params = {
            "days": row['days'],
            "arrival_vp_per_day": row['arrival_vp_per_day'],
            "approvers": int(row['approvers']),
            "service_vp_per_day": row['service_vp_per_day'],
            "p_rework": row['p_rework'],
            "skill_factor": row['skill_factor'],
            "skill_at_dependency_resolution": row['skill_at_dependency_resolution'],
            "dependency_level": row['dependency_level'],
            "coordination_penalty": row['coordination_penalty']
        }
        
        n_trials = int(row['n_trials'])
        trials = run_monte_carlo(n_trials=n_trials, **params)
        
        all_waits = []
        for r in trials:
            all_waits.extend(r["logs"]["wait_times"])
        
        wait_times_dict[name] = all_waits
        wip_logs_dict[name] = trials[0]["logs"]["wip_history"] # 最初の一回を代表に
        
        # 個別シナリオのWaterfall（寄与分解）
        # waterfallはひとつの状態を説明するものなので個別に作成
        base = row['service_vp_per_day']
        with_skill = base * row['skill_factor']
        skill_inc = with_skill - base
        adj_dep = row['dependency_level'] / (1.0 + row['skill_at_dependency_resolution'])
        coord_factor = 1.0 / (1.0 + row['coordination_penalty'] * adj_dep)
        final = with_skill * coord_factor
        dep_dec = with_skill - final
        
        viz.plot_waterfall(base, skill_inc, dep_dec, final, title=f"能力内訳: {name}")
        plt.savefig(f"{OUT_DIR}/waterfall_{name}.png")
        plt.close()

    # 全シナリオ比較: 待ち時間分布
    print("全シナリオ比較図を作成中...")
    viz.plot_wait_time_distribution(wait_times_dict, title="全シナリオ比較: 待ち時間分布 (P90強調)")
    plt.savefig(f"{OUT_DIR}/compare_violin.png")
    plt.close()

    # 全シナリオ比較: CCDF
    viz.plot_ccdf(wait_times_dict, title="全シナリオ比較: 超過確率カーブ (CCDF)")
    plt.savefig(f"{OUT_DIR}/compare_ccdf.png")
    plt.close()

    # 全シナリオ比較: WIP
    viz.plot_wip_timeseries(wip_logs_dict, title="全シナリオ比較: WIP推移 (代表試行)")
    plt.savefig(f"{OUT_DIR}/compare_wip.png")
    plt.close()

    # エグゼクティブ・サマリ (全シナリオだと多いので、上位/重要なものに絞ることも可能だが、一旦全部出す)
    viz.plot_executive_summary(wait_times_dict, threshold=14, title="Ver4 エグゼクティブ・ダッシュボード")
    plt.savefig(f"{OUT_DIR}/compare_dashboard.png")
    plt.close()

    print(f"\n全てのシミュレーションと可視化が完了しました。")
    print(f"結果は {OUT_DIR} に保存されました。")

if __name__ == "__main__":
    run_all_scenarios()

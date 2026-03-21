import os
import json
import pandas as pd
import numpy as np
from core.simulation import Simulation
from analysis.viz import plot_tech_history, plot_wip_heatmap, plot_all_results, plot_kpi_comparison

def run_doe_01():
    doe_id = "DOE01"
    output_dir = f"reports/doe/{doe_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = []
    
    # 1. Sensitivity Analysis: Engineers vs DR Period
    for n_eng in [2, 5, 8]:
        for dr_p in [7, 14, 21]:
            scenarios.append({
                "name": f"Sens_Eng{n_eng}_DR{dr_p}",
                "params": {
                    "res_n_servers": n_eng,
                    "proto_n_servers": max(1, n_eng // 2),
                    "dr1_period": dr_p,
                    "dr2_period": dr_p * 2,
                    "dr3_period": dr_p * 3,
                    "steps": 1000
                }
            })

    # 2. Extreme and Solid Examples
    scenarios.append({
        "name": "Extreme_HighPerf",
        "params": {
            "res_n_servers": 15,
            "proto_n_servers": 10,
            "dr1_period": 5,
            "dr2_period": 10,
            "dr3_period": 15,
            "dr_threshold": 0.2,
            "steps": 1000
        }
    })
    
    scenarios.append({
        "name": "Extreme_Struggling",
        "params": {
            "res_n_servers": 2,
            "proto_n_servers": 1,
            "dr1_period": 30,
            "dr2_period": 60,
            "dr3_period": 90,
            "dr_threshold": 0.8,
            "steps": 1000
        }
    })

    scenarios.append({
        "name": "Solid_Conservative",
        "params": {
            "res_n_servers": 5,
            "proto_n_servers": 3,
            "dr1_period": 14,
            "dr2_period": 28,
            "dr3_period": 42,
            "dr_threshold": 0.4,
            "steps": 1000
        }
    })

    all_results = []
    iterations = 5 # 統計的安定性のために5回ずつ実行

    print(f"Starting {doe_id}...")
    
    plot_dir = f"{output_dir}/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    for sc in scenarios:
        print(f"Running scenario: {sc['name']}...")
        scenario_results = []
        for i in range(iterations):
            params = sc['params'].copy()
            params['seed'] = 42 + i
            
            sim = Simulation()
            sim.setup_with_params(params)
            kpis = sim.run(steps=params['steps'])
            
            # 最初のイテレーションについてのみ詳細グラフを保存
            if i == 0:
                prefix = f"{plot_dir}/{doe_id}_{sc['name']}"
                plot_tech_history(sim.tech_history, f"{prefix}_tech.png")
                plot_wip_heatmap(sim.engine.results.get("wip_history", []), f"{prefix}_wip.png")
            
            res = {
                "doe_id": doe_id,
                "scenario": sc['name'],
                "iteration": i,
                **params,
                **kpis
            }
            # Remove complex objects from results if any
            if 'tech_items' in res: del res['tech_items']
            if 'teams' in res: del res['teams']
            
            scenario_results.append(res)
            all_results.append(res)
        
        # 保存：シナリオごとの設定ファイル
        with open(f"{output_dir}/{doe_id}_{sc['name']}_config.json", 'w', encoding='utf-8') as f:
            json.dump(sc['params'], f, indent=4, ensure_ascii=False)

    # 統合結果の保存
    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/{doe_id}_results.csv", index=False)
    
    # サマリの作成 (平均値)
    summary_cols = ['total_gain', 'completed_jobs', 'total_experiments', 'technical_failures', 'dr_cost']
    df_summary = df.groupby('scenario')[summary_cols].mean().reset_index()
    df_summary.to_csv(f"{output_dir}/{doe_id}_summary_table.csv", index=False)
    
    # 全体比較グラフ
    plot_kpi_comparison(all_results, f"{plot_dir}/{doe_id}_comparison.png")
    
    print(f"Completed {doe_id}. Results saved to {output_dir}")

if __name__ == "__main__":
    run_doe_01()

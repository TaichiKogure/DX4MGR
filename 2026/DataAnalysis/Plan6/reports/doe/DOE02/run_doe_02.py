import os
import json
import pandas as pd
import numpy as np
from core.simulation import Simulation
from analysis.viz import plot_tech_history, plot_wip_heatmap, plot_kpi_comparison

def run_doe_02():
    doe_id = "DOE02"
    output_dir = f"reports/doe/{doe_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = []
    
    # 1. Sensitivity Analysis: Technical Threshold vs Knowledge Decay
    for u_thr in [0.2, 0.4, 0.6]:
        for dec in [0.5, 0.8, 0.95]:
            scenarios.append({
                "name": f"Sens_Uncert{u_thr}_Decay{dec}",
                "params": {
                    "dr_threshold": u_thr, # Using as proxy for passing criteria
                    "decay": dec,
                    "res_n_servers": 5,
                    "proto_n_servers": 3,
                    "dr1_period": 14,
                    "steps": 1000
                }
            })

    all_results = []
    iterations = 5

    print(f"Starting {doe_id}...")
    
    plot_dir = f"{output_dir}/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    for sc in scenarios:
        print(f"Running scenario: {sc['name']}...")
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
            if 'tech_items' in res: del res['tech_items']
            if 'teams' in res: del res['teams']
            
            all_results.append(res)
        
        with open(f"{output_dir}/{doe_id}_{sc['name']}_config.json", 'w', encoding='utf-8') as f:
            json.dump(sc['params'], f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(all_results)
    df.to_csv(f"{output_dir}/{doe_id}_results.csv", index=False)
    
    summary_cols = ['total_gain', 'completed_jobs', 'total_experiments', 'technical_failures', 'dr_cost']
    df_summary = df.groupby('scenario')[summary_cols].mean().reset_index()
    df_summary.to_csv(f"{output_dir}/{doe_id}_summary_table.csv", index=False)
    
    # 全体比較グラフ
    plot_kpi_comparison(all_results, f"{plot_dir}/{doe_id}_comparison.png")
    
    print(f"Completed {doe_id}. Results saved to {output_dir}")

if __name__ == "__main__":
    run_doe_02()

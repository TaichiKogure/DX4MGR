import os
import sys
import pandas as pd
import numpy as np
from core.simulation import Simulation

def run_doe_simulation(num_runs=100):
    output_dir = 'reports/doe'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_results = []
    
    # 乱数シードの固定
    np.random.seed(42)

    for i in range(1, num_runs + 1):
        test_no = f"TestNo_{i:03d}"
        
        # パラメータのランダム生成 (探索範囲を設定)
        params = {
            'test_no': test_no,
            'dr1_period': int(np.random.uniform(7, 30)),
            'dr_threshold': float(np.random.uniform(0.1, 0.6)),
            'cost_per_review': float(np.random.uniform(50.0, 300.0)),
            'rework_load_factor': float(np.random.uniform(0.3, 0.9)),
            'decay': float(np.random.uniform(0.5, 0.95)),
            'uncertainty_threshold': float(np.random.uniform(0.2, 0.5)),
            'maturity_threshold': float(np.random.uniform(0.5, 0.9)),
            'steps': 1500 # 十分な期間
        }
        
        # シミュレーション実行
        sim = Simulation()
        sim.setup_with_params(params)
        kpis = sim.run(steps=params['steps'])
        
        # パラメータと結果を統合
        result_entry = {**params, **kpis}
        all_results.append(result_entry)
        
        # 個別CSV保存
        df_single = pd.DataFrame([result_entry])
        df_single.to_csv(f"{output_dir}/{test_no}.csv", index=False)
        
        if i % 10 == 0:
            print(f"Completed {i}/{num_runs} runs...")

    # 統合レポートの保存
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv('reports/doe_summary.csv', index=False)
    print("\nDOE Completed. Summary saved to reports/doe_summary.csv")

if __name__ == "__main__":
    run_doe_simulation(100)

import os
import sys
import pandas as pd
from core.simulation import Simulation

def run_batch_simulation():
    # パラメータの設定
    params = {
        'scenario_id': 'Batch_Run_MultiStage',
        'steps': 1000,
        'dr_threshold': 4.0,
        'strategy': 'strategic',
        'dr1_period': 20,
        'rework_load_factor': 0.6,
        'max_rework_cycles': 3,
        'decay': 0.8,
        'use_digital_twin': False,
        'loss_res_proto': 0.15,
        'dist_proto_ana': 0.1,
        'delay_res_proto': 3,
        'delay_proto_ana': 2,
        'delay_ana_res': 3
    }
    
    print("Starting Multi-Stage Simulation...")
    sim = Simulation()
    sim.setup_with_params(params)
    
    kpis = sim.run(steps=params['steps'])
    
    print("\n--- Simulation Results ---")
    print(f"Total Gain: {kpis['total_gain']:.2f}")
    print(f"Completed Projects: {kpis['completed_jobs']}")
    print(f"Total Experiments: {kpis['total_experiments']}")
    print(f"Technical Failures: {kpis['technical_failures']}")
    print(f"Operational Failures: {kpis['operational_failures']}")
    print(f"DR Total Cost: {kpis['dr_cost']:.2f}")
    
    print("\n--- Technology Maturity Status ---")
    tech_status = sim.get_tech_status()
    for name, status in tech_status.items():
        print(f"  {name}: Maturity={status['maturity']:.2f}, Uncertainty={status['uncertainty']:.2f}")

    # 結果の保存
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    df = pd.DataFrame([kpis])
    df.to_csv('reports/batch_results.csv', index=False)
    print("\nResults saved to reports/batch_results.csv")

if __name__ == "__main__":
    run_batch_simulation()

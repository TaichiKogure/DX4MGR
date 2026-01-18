import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.getcwd(), "2026/RDIssues/Ver6"))

from runner.adapters import setup_standard_flow
from runner.experiment import run_single_trial
from analysis.metrics import calculate_metrics
from analysis.viz import plot_all_results
import numpy as np

def main():
    params = {
        "days": 500,
        "arrival_rate": 0.2, # 5日に1件
        "approvers": 2,
        "small_exp_duration": 10,
        "proto_duration": 30,
        "bundle_size": 3,
        "dr_period": 90,
        "rework_load_factor": 2.0,
        "max_rework_cycles": 3
    }
    
    print("Starting simulation with new engine...")
    
    # 単一試行の実行
    engine = run_single_trial(setup_standard_flow, trial_id=0, base_seed=42, **params)
    
    # 指標の計算
    nodes_stats = [node.stats() for node in engine.nodes.values()]
    metrics = calculate_metrics(engine.results["completed_jobs"], nodes_stats, params["days"])
    
    print("\n--- Summary ---")
    print(f"Completed Jobs: {metrics['summary']['completed_count']}")
    print(f"Throughput: {metrics['summary']['throughput']:.4f} jobs/day")
    print(f"Lead Time P50: {metrics['summary']['lead_time_p50']:.2f} days")
    print(f"Lead Time P90: {metrics['summary']['lead_time_p90']:.2f} days")
    print(f"Avg Reworks: {metrics['summary']['avg_reworks']:.2f}")
    
    # 可視化
    output_dir = "2026/RDIssues/Ver6/output"
    os.makedirs(output_dir, exist_ok=True)
    plot_all_results(metrics, output_path=f"{output_dir}/v6_new_engine_verify.png")

if __name__ == "__main__":
    main()

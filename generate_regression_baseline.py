import json
import os
import sys
import numpy as np
import pandas as pd

# プロジェクトルートとVer9のパスを通す
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VER9_DIR = os.path.join(CURRENT_DIR, "2026/RDIssues/archive/Ver9")
if VER9_DIR not in sys.path:
    sys.path.append(VER9_DIR)

from simulator_v8 import run_monte_carlo_v8

def generate_baseline():
    print("Generating baseline data from Ver8 code...")
    
    # 3つのパラメータセット (小/中/大負荷)
    scenarios = [
        {
            "name": "low_load",
            "params": {
                "days": 180,
                "arrival_rate": 0.2,
                "small_exp_duration": 5,
                "proto_duration": 15,
                "bundle_size": 2,
                "dr_period": 30,
                "dr_capacity": 10,
                "dr_quality": 0.9,
                "rework_load_factor": 0.5,
                "max_rework_cycles": 3,
                "decay": 0.8,
                "approvers": 2
            }
        },
        {
            "name": "mid_load",
            "params": {
                "days": 180,
                "arrival_rate": 0.5,
                "small_exp_duration": 5,
                "proto_duration": 20,
                "bundle_size": 3,
                "dr_period": 60,
                "dr_capacity": 10,
                "dr_quality": 0.8,
                "rework_load_factor": 1.0,
                "max_rework_cycles": 5,
                "decay": 0.7,
                "approvers": 2
            }
        },
        {
            "name": "high_load",
            "params": {
                "days": 180,
                "arrival_rate": 0.8,
                "small_exp_duration": 7,
                "proto_duration": 30,
                "bundle_size": 5,
                "dr_period": 90,
                "dr_capacity": 5,
                "dr_quality": 0.7,
                "rework_load_factor": 2.0,
                "max_rework_cycles": 5,
                "decay": 0.6,
                "approvers": 1
            }
        }
    ]
    
    baseline_results = {}
    n_trials = 10
    seed = 42
    
    for scn in scenarios:
        print(f"Running scenario: {scn['name']}")
        # Ver8のrun_monte_carlo_v8はseedを引数に取らないが、内部でrange(n_trials)をseedとして使っている
        # 決定論的に動作するか確認が必要
        trials = run_monte_carlo_v8(n_trials=n_trials, use_parallel=False, **scn['params'])
        
        throughputs = [t["summary"]["throughput"] for t in trials]
        lead_times = [lt for t in trials for lt in t["logs"]["lead_times"]]
        rework_counts = [rc for t in trials for rc in t["logs"]["rework_counts"]]
        
        baseline_results[scn['name']] = {
            "throughput": {
                "mean": float(np.mean(throughputs)),
                "std": float(np.std(throughputs))
            },
            "lead_time": {
                "p50": float(np.percentile(lead_times, 50)) if lead_times else 0,
                "p90": float(np.percentile(lead_times, 90)) if lead_times else 0,
                "p95": float(np.percentile(lead_times, 95)) if lead_times else 0
            },
            "rework_count": {
                "mean": float(np.mean(rework_counts)) if rework_counts else 0
            }
        }
    
    output_path = os.path.join(CURRENT_DIR, "ver9_regression_baseline.json")
    with open(output_path, "w") as f:
        json.dump(baseline_results, f, indent=4)
    print(f"Baseline saved to {output_path}")

if __name__ == "__main__":
    generate_baseline()

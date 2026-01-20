import json
import os
import sys
import numpy as np

# プロジェクトルートとVer9のパスを通す
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VER9_DIR = os.path.join(CURRENT_DIR, "../archive/Ver9")
if VER9_DIR not in sys.path:
    sys.path.append(VER9_DIR)

from simulator_v8 import run_monte_carlo_v8

def run_regression():
    print("Running regression test for Ver9 (Step 3)...")
    
    baseline_path = os.path.join(CURRENT_DIR, "ver9_regression_baseline.json")
    if not os.path.exists(baseline_path):
        print("Error: Baseline file not found.")
        return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

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
    
    n_trials = 10
    all_passed = True
    
    for scn in scenarios:
        print(f"Testing scenario: {scn['name']}")
        # Ver9の回帰テストでは、ベースラインが simulator_v8 に基づいているため、意図的に simulator_v8 を使用
        trials = run_monte_carlo_v8(n_trials=n_trials, use_parallel=False, **scn['params'])
        
        throughputs = [t["summary"]["throughput"] for t in trials]
        current_mean_tp = float(np.mean(throughputs))
        
        base_mean_tp = baseline[scn['name']]["throughput"]["mean"]
        
        # 許容誤差 1e-6 (ベースラインとの厳密な一致を確認)
        diff_tp = abs(current_mean_tp - base_mean_tp)
        if diff_tp > 1e-6:
            print(f"  [FAIL] Throughput mismatch: Current={current_mean_tp}, Baseline={base_mean_tp}, Diff={diff_tp}")
            all_passed = False
        else:
            print(f"  [PASS] Throughput matches baseline.")
            
    if all_passed:
        print("\nALL REGRESSION TESTS PASSED.")
    else:
        print("\nREGRESSION TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_regression()

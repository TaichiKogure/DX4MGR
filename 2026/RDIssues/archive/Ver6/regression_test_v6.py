import json
import os
import numpy as np
from simulator_v6 import simulate_flow_v6

def get_test_params():
    return {
        "low_load": {
            "days": 100,
            "arrival_vp_per_day": 2.0,
            "approvers": 2,
            "service_vp_per_day": 5.0,
            "p_rework": 0.1,
            "rework_load_factor": 0.5,
            "seed": 42
        },
        "mid_load": {
            "days": 100,
            "arrival_vp_per_day": 5.0,
            "approvers": 2,
            "service_vp_per_day": 5.0,
            "p_rework": 0.2,
            "rework_load_factor": 1.0,
            "seed": 42
        },
        "high_load": {
            "days": 100,
            "arrival_vp_per_day": 10.0,
            "approvers": 2,
            "service_vp_per_day": 5.0,
            "p_rework": 0.3,
            "rework_load_factor": 2.0,
            "seed": 42
        }
    }

def run_and_serialize(params):
    res = simulate_flow_v6(**params)
    summary = res["summary"]
    # JSON化可能なように型変換
    serializable_summary = {}
    for k, v in summary.items():
        if isinstance(v, (np.float64, np.float32)):
            serializable_summary[k] = float(v)
        elif isinstance(v, (np.int64, np.int32)):
            serializable_summary[k] = int(v)
        else:
            serializable_summary[k] = v
    return serializable_summary

def generate_baseline(filepath):
    params_set = get_test_params()
    baseline = {}
    for name, params in params_set.items():
        print(f"Running simulation for {name}...")
        baseline[name] = run_and_serialize(params)
    
    with open(filepath, 'w') as f:
        json.dump(baseline, f, indent=4)
    print(f"Baseline saved to {filepath}")

def test_regression(filepath):
    if not os.path.exists(filepath):
        print(f"Baseline file {filepath} not found. Generating...")
        generate_baseline(filepath)
        return

    params_set = get_test_params()
    with open(filepath, 'r') as f:
        baseline = json.load(f)

    for name, params in params_set.items():
        print(f"Testing {name}...")
        current = run_and_serialize(params)
        base = baseline[name]
        
        for k in base.keys():
            if k not in current:
                print(f"  FAILED: {k} missing in current results")
                continue
            
            # 浮動小数点の誤差を許容
            if not np.isclose(current[k], base[k], rtol=1e-5):
                print(f"  FAILED: {k} mismatch. Expected {base[k]}, got {current[k]}")
            else:
                pass
                # print(f"  PASSED: {k}")

if __name__ == "__main__":
    baseline_path = "2026/RDIssues/Ver6/regression_baseline.json"
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        generate_baseline(baseline_path)
    else:
        test_regression(baseline_path)

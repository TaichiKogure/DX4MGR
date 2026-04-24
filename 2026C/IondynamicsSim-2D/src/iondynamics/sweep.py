import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from iondynamics.config import SimConfig, load_config
from iondynamics.simulate import run_simulation
from iondynamics.breakdown import compute_breakdown

def run_sweep(base_cfg: SimConfig, sweep_spec: Dict[str, List[Any]]) -> pd.DataFrame:
    """
    sweep_spec: { "electrode.thickness_um": [40, 80, 120] }
    """
    import itertools
    keys = list(sweep_spec.keys())
    values = list(sweep_spec.values())
    combinations = list(itertools.product(*values))
    
    results = []
    
    for combo in combinations:
        cfg = base_cfg # Note: 深いコピーが必要かもしれないが、今は手動で上書き
        params_str = ""
        for k, v in zip(keys, combo):
            # dict の階層を辿ってセット
            parts = k.split('.')
            target = cfg
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], v)
            params_str += f"{k}={v}_"
            
        print(f"Running simulation for {params_str}")
        res = run_simulation(cfg)
        breakdown = compute_breakdown(res)
        
        # 統計量の計算
        # 放電容量 (Ah) = Current(A) * Time(s) / 3600
        # PyBaMMのCurrentは A で、正の方向が放電
        capacity = np.trapezoid(res.current, res.time) / 3600
        avg_voltage = np.mean(res.voltage)
        
        entry = {
            "capacity_Ah": capacity,
            "avg_voltage_V": avg_voltage
        }
        for k, v in zip(keys, combo):
            entry[k] = v
            
        # 抵抗成分の時間平均
        for k_res, v_res in breakdown.items():
            if k_res != "total":
                entry[f"avg_{k_res}_V"] = np.mean(v_res)
        
        results.append(entry)
        
    return pd.DataFrame(results)

def plot_sensitivity_heatmap(df: pd.DataFrame, x_key: str, y_key: str, z_key: str, out_path: str):
    pivot_df = df.pivot(index=y_key, columns=x_key, values=z_key)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis")
    plt.title(f"Sensitivity Analysis: {z_key}")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.savefig(out_path)
    plt.close()

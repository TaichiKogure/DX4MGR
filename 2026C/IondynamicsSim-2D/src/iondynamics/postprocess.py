import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import os

@dataclass
class PostProcessResult:
    delta_ce: np.ndarray             # Δc_e_through_thickness [mol/m3]
    max_grad_ce: np.ndarray         # max concentration gradient [mol/m4]
    resistance_index: np.ndarray    # effective ionic resistance index
    ce_ratio: np.ndarray            # separator/collector ratio [-]
    depletion_time: Optional[float] # depletion onset time [s]
    
    # サマリ値 (最終値)
    final_values: Dict[str, float]

def compute_kpis(sim_result, depletion_threshold=0.1) -> PostProcessResult:
    """
    シミュレーション結果からKPIを算出する
    sim_result.c_e は (Nx, Nt)
    x=0 がセパレータ側, x=L が集電体側と想定
    """
    c_e = sim_result.c_e
    x = sim_result.x
    t = sim_result.time
    
    dx = x[1] - x[0] if len(x) > 1 else 1e-6
    
    # 1. Δc_e_through_thickness (セパレータ側 - 集電体側)
    # 放電時はセパレータ側で濃度が高く、集電体側で低くなる傾向
    c_sep = c_e[0, :]
    c_coll = c_e[-1, :]
    delta_ce = c_sep - c_coll
    
    # 2. max concentration gradient
    # 勾配の絶対値の最大値
    grad_ce = np.abs(np.diff(c_e, axis=0) / dx)
    max_grad_ce = np.max(grad_ce, axis=0)
    
    # 3. effective ionic resistance index
    # 簡易定義: 濃度差を電流(C-rate)で正規化したもの
    # 厚膜化・低空隙率化で増大する指標
    c_rate = sim_result.config.operation.c_rate
    resistance_index = delta_ce / (c_rate + 1e-9)
    
    # 4. separator-side / current-collector-side concentration ratio
    ce_ratio = c_sep / (c_coll + 1e-9)
    
    # 5. depletion onset time
    # 集電体側（最も低くなる場所）が閾値を下回る時刻
    # threshold は初期濃度に対する比率または絶対値
    # ここでは簡易的に絶対値または初期値からの割合とする
    c_init = c_e[:, 0].mean()
    threshold = depletion_threshold * c_init if depletion_threshold < 1.0 else depletion_threshold
    
    depleted_mask = c_coll < threshold
    if np.any(depleted_mask):
        depletion_time = t[np.where(depleted_mask)[0][0]]
    else:
        depletion_time = None
        
    final_values = {
        "delta_ce_final": float(delta_ce[-1]),
        "max_grad_ce_final": float(max_grad_ce[-1]),
        "resistance_index_final": float(resistance_index[-1]),
        "ce_ratio_final": float(ce_ratio[-1]),
        "depletion_time": float(depletion_time) if depletion_time is not None else -1.0
    }
    
    return PostProcessResult(
        delta_ce=delta_ce,
        max_grad_ce=max_grad_ce,
        resistance_index=resistance_index,
        ce_ratio=ce_ratio,
        depletion_time=depletion_time,
        final_values=final_values
    )

def save_thickness_data(sim_result, run_dir):
    """厚み方向の濃度分布をCSV保存する"""
    x_um = sim_result.x * 1e6
    t = sim_result.time
    c_e = sim_result.c_e
    
    # 行が座標、列が時刻の形式
    df = pd.DataFrame(c_e, index=x_um)
    df.index.name = "x_um"
    df.columns = [f"t_{time:.2f}s" for time in t]
    
    df.to_csv(os.path.join(run_dir, "thickness_ce_distribution.csv"))

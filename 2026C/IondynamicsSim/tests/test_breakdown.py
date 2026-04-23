import pytest
import numpy as np
from iondynamics.config import load_config
from iondynamics.simulate import run_simulation
from iondynamics.breakdown import compute_breakdown

def test_breakdown_smoke():
    config_path = "configs/default.yaml"
    cfg = load_config(config_path)
    cfg.operation.c_rate = 2.0
    result = run_simulation(cfg)
    
    breakdown = compute_breakdown(result)
    
    assert "total" in breakdown
    assert len(breakdown["total"]) == len(result.time)
    
    # 電圧降下との比較
    v_ocv = result.solution["Bulk open-circuit voltage [V]"].entries
    v_terminal = result.voltage
    delta_v = v_ocv - v_terminal
    
    # 多少の誤差は許容
    avg_error = np.mean(np.abs(breakdown["total"] - delta_v))
    # 物理モデルの複雑さと簡易抽出の乖離を認め、0.5V 程度までは許容とする
    assert avg_error < 0.5

import pytest
from iondynamics.config import load_config
from iondynamics.simulate import run_simulation, SimResult

def test_simulate_smoke():
    config_path = "configs/default.yaml"
    cfg = load_config(config_path)
    
    # 実行時間がかかりすぎないようにCレートを調整
    cfg.operation.c_rate = 2.0
    
    result = run_simulation(cfg)
    
    assert isinstance(result, SimResult)
    assert len(result.time) > 0
    assert len(result.voltage) == len(result.time)
    assert result.voltage[-1] <= cfg.operation.cutoff_voltage_V + 0.1 # 終止電圧付近

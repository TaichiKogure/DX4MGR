import pybamm
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
from .config import SimConfig

@dataclass
class SimResult:
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    c_e: np.ndarray  # (x, t)
    c_s_surf: np.ndarray # (x, t)
    x: np.ndarray
    solution: pybamm.Solution
    config: SimConfig

def run_simulation(cfg: SimConfig) -> SimResult:
    # 1. モデル定義
    model = pybamm.lithium_ion.DFN()
    
    # 2. パラメータ設定
    params = pybamm.ParameterValues("Chen2020")
    
    # cfg からの値を反映
    params.update({
        "Positive electrode thickness [m]": cfg.electrode.thickness_um * 1e-6,
        "Positive electrode porosity": cfg.electrode.porosity,
        "Positive particle radius [m]": cfg.electrode.particle_radius_um * 1e-6,
        "Positive electrode conductivity [S.m-1]": cfg.resistances.electronic_conductivity_S_m,
        "Electrolyte conductivity [S.m-1]": cfg.resistances.ionic_conductivity_S_m,
        "Positive particle diffusivity [m2.s-1]": cfg.resistances.solid_diffusivity_m2_s,
    }, check_already_exists=False)
    
    # 3. 実験設定
    c_rate = cfg.operation.c_rate
    cutoff = cfg.operation.cutoff_voltage_V
    experiment = pybamm.Experiment([
        f"Discharge at {c_rate}C until {cutoff}V"
    ])
    
    # 4. シミュレーション実行
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    sol = sim.solve()
    
    # 5. 結果の抽出
    t = sol["Time [s]"].entries
    v = sol["Terminal voltage [V]"].entries
    i = sol["Current [A]"].entries
    
    # PyBaMMの変数抽出 (空間分布)
    def get_spatial_var(var_name):
        var = sol[var_name]
        # data は (Nx, Nt) または (Nx, Nr, Nt) など
        data = var.entries
        # 座標を取得
        # var.mesh は存在しない場合があるため、processed variable から取る
        # 簡易的に 0 から L_p までの np.linspace を作成し、
        # データの最初の次元数に合わせる
        nx = data.shape[0]
        x_local = np.linspace(0, cfg.electrode.thickness_um * 1e-6, nx)
        
        # 2D (Nx, Nt) に整形
        if data.ndim == 3:
            data = data[:, 0, :] # 粒子の表面など
            
        return x_local, data

    try:
        x, c_e_dist = get_spatial_var("Positive electrode electrolyte concentration [mol.m-3]")
        _, c_s_surf_dist = get_spatial_var("Positive particle surface concentration [mol.m-3]")
    except KeyError:
        # フォールバック: 全体から正極分を推定
        c_e_full = sol["Electrolyte concentration [mol.m-3]"].entries
        nx_full = c_e_full.shape[0]
        # 正極は通常最後の 1/3 程度 (負極/セパレータ/正極)
        nx_p = nx_full // 3
        c_e_dist = c_e_full[-nx_p:]
        c_s_surf_dist = sol["Positive particle surface concentration [mol.m-3]"].entries
        if c_s_surf_dist.shape[0] != c_e_dist.shape[0]:
            # 形状を c_s_surf に合わせる
            nx_p = c_s_surf_dist.shape[0]
            c_e_dist = c_e_full[-nx_p:]
        x = np.linspace(0, cfg.electrode.thickness_um * 1e-6, nx_p)

    return SimResult(
        time=t,
        voltage=v,
        current=i,
        c_e=c_e_dist,
        c_s_surf=c_s_surf_dist,
        x=x,
        solution=sol,
        config=cfg
    )

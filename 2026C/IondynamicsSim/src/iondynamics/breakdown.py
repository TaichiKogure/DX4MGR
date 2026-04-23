import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def compute_breakdown(result) -> Dict[str, np.ndarray]:
    sol = result.solution
    
    # 抵抗成分の抽出 (V)
    # PyBaMMの変数名はバージョンによって異なる場合があるため try/except で対応
    
    def get_var(name):
        try:
            return sol[name].entries
        except KeyError:
            return np.zeros_like(sol["Time [s]"].entries)

    # 1. 電子伝導抵抗
    eta_ohm_s = get_var("X-averaged positive electrode ohmic losses [V]")
    
    # 2. イオン輸送抵抗
    # 電解液オーム損失と拡散損失の合計を試す
    eta_ohm_e = get_var("X-averaged electrolyte ohmic losses [V]")
    
    # 3. 反応過電圧
    eta_ct = get_var("X-averaged positive electrode reaction overpotential [V]")
    
    # 4. 電解液拡散過電圧
    eta_conc = get_var("X-averaged electrolyte concentration overpotential [V]")
    
    # 5. 固体内拡散過電圧
    eta_s_diff = get_var("X-averaged positive particle concentration overpotential [V]")
    
    # 6. 負極側も考慮が必要かもしれない (DFNなので)
    eta_ohm_s_n = get_var("X-averaged negative electrode ohmic losses [V]")
    eta_ct_n = get_var("X-averaged negative electrode reaction overpotential [V]")
    eta_s_diff_n = get_var("X-averaged negative particle concentration overpotential [V]")

    # 合計 (負極も加味)
    total = eta_ohm_s + eta_ohm_e + eta_ct + eta_conc + eta_s_diff + eta_ohm_s_n + eta_ct_n + eta_s_diff_n
    
    return {
        "ohmic_electronic": eta_ohm_s + eta_ohm_s_n,
        "ohmic_ionic": eta_ohm_e,
        "reaction": eta_ct + eta_ct_n,
        "electrolyte_diffusion": eta_conc,
        "solid_diffusion": eta_s_diff + eta_s_diff_n,
        "total": total
    }

def plot_breakdown_bar(breakdown: Dict[str, np.ndarray], t_index: int, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    keys = ["ohmic_electronic", "ohmic_ionic", "reaction", "electrolyte_diffusion", "solid_diffusion"]
    labels = ["Ohmic (s)", "Ohmic (e)", "Reaction", "Diff (e)", "Diff (s)"]
    values = [breakdown[k][t_index] for k in keys]
    
    ax.bar(labels, values)
    ax.set_ylabel("Overpotential [V]")
    ax.set_title(f"Resistance Breakdown at t={t_index}")
    return ax

def plot_breakdown_stack(breakdown: Dict[str, np.ndarray], times: np.ndarray, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        
    keys = ["ohmic_electronic", "ohmic_ionic", "reaction", "electrolyte_diffusion", "solid_diffusion"]
    labels = ["Ohmic (s)", "Ohmic (e)", "Reaction", "Diff (e)", "Diff (s)"]
    
    y = np.row_stack([breakdown[k] for k in keys])
    ax.stackplot(times, y, labels=labels)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Overpotential [V]")
    ax.legend(loc='upper left')
    ax.set_title("Resistance Breakdown Over Time")
    return ax

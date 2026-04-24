import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from iondynamics.breakdown import compute_breakdown, plot_breakdown_bar, plot_breakdown_stack

def plot_voltage_time(result, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(result.time, result.voltage)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Voltage vs Time")
    return ax

def plot_ce_profile(result, t_index, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(result.x * 1e6, result.c_e[:, t_index])
    ax.set_xlabel("x [um] (0: separator side)")
    ax.set_ylabel("Electrolyte conc [mol/m3]")
    ax.set_title(f"Electrolyte concentration at t={result.time[t_index]:.1f}s")
    return ax

def plot_ce_time_series(result, ax=None, num_profiles=5):
    """厚み方向濃度の時系列プロファイルを表示"""
    if ax is None:
        fig, ax = plt.subplots()
    
    nt = len(result.time)
    indices = np.linspace(0, nt-1, num_profiles, dtype=int)
    
    for idx in indices:
        t_val = result.time[idx]
        ax.plot(result.x * 1e6, result.c_e[:, idx], label=f"t={t_val:.1f}s")
    
    ax.set_xlabel("x [um] (0: separator side)")
    ax.set_ylabel("Electrolyte conc [mol/m3]")
    ax.set_title("Electrolyte Concentration Profile")
    ax.legend()
    return ax

def plot_kpi_time_series(result, kpis, ax=None):
    """KPIの時系列変化を表示"""
    if ax is None:
        fig, ax = plt.subplots()
    
    ax2 = ax.twinx()
    
    ln1 = ax.plot(result.time, kpis.delta_ce, 'b-', label="Δc_e [mol/m3]")
    ln2 = ax2.plot(result.time, kpis.ce_ratio, 'r--', label="c_e ratio [-]")
    
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Δc_e [mol/m3]")
    ax2.set_ylabel("Ratio [-]")
    
    lns = ln1 + ln2
    
    # 手動で合成
    ax.legend(lns, [l.get_label() for l in lns], loc='best')
    ax.set_title("KPI Time Series")
    return ax

import matplotlib.pyplot as plt
import numpy as np
from .breakdown import compute_breakdown, plot_breakdown_bar, plot_breakdown_stack

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
    ax.set_xlabel("x [um]")
    ax.set_ylabel("Electrolyte conc [mol/m3]")
    ax.set_title(f"Electrolyte concentration at t={result.time[t_index]:.1f}s")
    return ax

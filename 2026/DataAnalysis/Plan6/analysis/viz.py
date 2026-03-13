import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List

def plot_all_results(metrics: Dict[str, Any], output_path: str = "output_new.png"):
    if "error" in metrics:
        print(f"Error in metrics: {metrics['error']}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Lead Time Distribution (Violin/Box)
    sns.violinplot(y=metrics["raw_lead_times"], ax=axes[0, 0])
    axes[0, 0].set_title("Lead Time Distribution")
    axes[0, 0].set_ylabel("Days")
    
    # 2. CCDF
    axes[0, 1].step(metrics["ccdf"]["x"], metrics["ccdf"]["y"], where='post')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_title("Lead Time CCDF (Log Scale)")
    axes[0, 1].set_xlabel("Days")
    axes[0, 1].set_ylabel("P(X > x)")
    axes[0, 1].grid(True, which="both", ls="-", alpha=0.5)
    
    # 3. Gate Stats (Avg Wait Time)
    gate_names = [s["node_id"] for s in metrics["gate_stats"]]
    avg_waits = [s["avg_wait_time"] for s in metrics["gate_stats"]]
    axes[1, 0].bar(gate_names, avg_waits)
    axes[1, 0].set_title("Avg Wait Time per Gate")
    axes[1, 0].set_ylabel("Days")
    
    # 4. Rework Count Distribution
    sns.histplot(metrics["raw_rework_counts"], discrete=True, ax=axes[1, 1])
    axes[1, 1].set_title("Rework Count Distribution")
    axes[1, 1].set_xlabel("Rework Count")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def plot_tech_history(tech_history: List[Dict[str, Any]], output_path: str = "tech_history.png"):
    if not tech_history:
        return
    
    times = [h['time'] for h in tech_history]
    tech_names = tech_history[0]['tech_items'].keys()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for name in tech_names:
        maturities = [h['tech_items'][name]['maturity'] for h in tech_history]
        uncertainties = [h['tech_items'][name]['uncertainty'] for h in tech_history]
        
        ax1.plot(times, maturities, label=f"{name} (Maturity)")
        ax2.plot(times, uncertainties, label=f"{name} (Uncertainty)", linestyle='--')
    
    ax1.set_ylabel("Maturity")
    ax1.set_title("Technology Maturity Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Uncertainty")
    ax2.set_xlabel("Days")
    ax2.set_title("Technology Uncertainty Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Tech history plot saved to {output_path}")

def plot_wip_heatmap(wip_history: List[Dict[str, Any]], output_path: str = "wip_heatmap.png"):
    if not wip_history:
        return
    
    import pandas as pd
    data = []
    for h in wip_history:
        for node_id, wip in h['node_wip'].items():
            data.append({"time": h['time'], "node": node_id, "wip": wip})
    
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index="node", columns="time", values="wip")
    
    plt.figure(figsize=(15, 8))
    sns.heatmap(df_pivot, cmap="YlOrRd", cbar_kws={'label': 'WIP (Queue + Service)'})
    plt.title("WIP Heatmap over Time")
    plt.xlabel("Time (Days)")
    plt.ylabel("Gate Node")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"WIP heatmap saved to {output_path}")

def plot_kpi_comparison(results: List[Dict[str, Any]], output_path: str = "kpi_comparison.png"):
    if not results:
        return
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Select numeric columns for scatter/correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return
    
    # 散布図: Experiment vs Gain
    plt.figure(figsize=(10, 6))
    if 'total_experiments' in df.columns and 'total_gain' in df.columns:
        sns.scatterplot(data=df, x='total_experiments', y='total_gain', hue='scenario_id' if 'scenario_id' in df.columns else None)
        plt.title("Total Experiments vs Total Gain")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"KPI comparison plot saved to {output_path}")

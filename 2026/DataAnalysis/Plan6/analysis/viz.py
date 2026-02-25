import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any

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
    print(f"Plot saved to {output_path}")

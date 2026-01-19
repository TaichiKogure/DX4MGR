import numpy as np
from typing import List, Dict, Any
from core.entities import Job

def calculate_metrics(completed_jobs: List[Job], nodes_stats: List[Dict[str, Any]], total_days: float):
    if not completed_jobs:
        return {"error": "No jobs completed"}

    lead_times = [job.history[-1]["time"] - job.created_at for job in completed_jobs]
    rework_counts = [job.rework_count for job in completed_jobs]
    
    # CCDFの計算用データ
    sorted_lt = np.sort(lead_times)
    ccdf_y = 1.0 - np.arange(1, len(sorted_lt) + 1) / len(sorted_lt)
    
    metrics = {
        "summary": {
            "completed_count": len(completed_jobs),
            "throughput": len(completed_jobs) / total_days,
            "lead_time_p50": float(np.percentile(lead_times, 50)),
            "lead_time_p90": float(np.percentile(lead_times, 90)),
            "lead_time_p95": float(np.percentile(lead_times, 95)),
            "avg_reworks": float(np.mean(rework_counts)),
            "max_reworks": int(np.max(rework_counts))
        },
        "gate_stats": nodes_stats,
        "ccdf": {
            "x": sorted_lt.tolist(),
            "y": ccdf_y.tolist()
        },
        "raw_lead_times": lead_times,
        "raw_rework_counts": rework_counts
    }
    
    return metrics

import numpy as np
from typing import List, Dict, Any
from core.entities import Job

def calculate_metrics(completed_jobs: List[Job], nodes_stats: List[Dict[str, Any]], total_days: float, wip_history: List[Dict[str, Any]] = None):
    if not completed_jobs:
        return {"error": "No jobs completed"}

    lead_times = [job.history[-1]["time"] - job.created_at for job in completed_jobs]
    rework_counts = [job.rework_count for job in completed_jobs]

    # CCDFの計算用データ
    sorted_lt = np.sort(lead_times)
    ccdf_y = 1.0 - np.arange(1, len(sorted_lt) + 1) / len(sorted_lt)

    # WIP集計（ノード別平均WIP）
    avg_wip_total = 0.0
    avg_wip_by_node: Dict[str, float] = {}
    if wip_history:
        totals = [x.get("total_wip", 0) for x in wip_history]
        avg_wip_total = float(np.mean(totals)) if totals else 0.0

        # node_wipを縦持ちにして平均
        acc: Dict[str, List[int]] = {}
        for row in wip_history:
            node_wip = row.get("node_wip", {}) or {}
            for node_id, v in node_wip.items():
                acc.setdefault(node_id, []).append(int(v))
        avg_wip_by_node = {k: float(np.mean(vs)) for k, vs in acc.items()}

    metrics = {
        "summary": {
            "completed_count": len(completed_jobs),
            "throughput": len(completed_jobs) / total_days,
            "lead_time_p50": float(np.percentile(lead_times, 50)),
            "lead_time_p90": float(np.percentile(lead_times, 90)),
            "lead_time_p95": float(np.percentile(lead_times, 95)),
            "avg_reworks": float(np.mean(rework_counts)),
            "max_reworks": int(np.max(rework_counts)),
            "avg_wip": avg_wip_total
        },
        "gate_stats": nodes_stats,
        "wip": {
            "avg_total": avg_wip_total,
            "avg_by_node": avg_wip_by_node,
            "history": wip_history or []
        },
        "ccdf": {
            "x": sorted_lt.tolist(),
            "y": ccdf_y.tolist()
        },
        "raw_lead_times": lead_times,
        "raw_rework_counts": rework_counts
    }

    return metrics

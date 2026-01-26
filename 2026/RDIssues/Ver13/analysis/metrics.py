import numpy as np
from typing import List, Dict, Any
from core.entities import Job

def calculate_metrics(completed_jobs: List[Job], nodes_stats: List[Dict[str, Any]], total_days: float, wip_history: List[Dict[str, Any]] = None):
    primary_jobs = [job for job in completed_jobs if not getattr(job, "is_rework_task", False)]
    rework_jobs = [job for job in completed_jobs if getattr(job, "is_rework_task", False)]

    def _lead_time_stats(values):
        if not values:
            return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0}
        return {
            "avg": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        }

    rework_lead_times = [job.history[-1]["time"] - job.created_at for job in rework_jobs if job.history]
    rework_stats = _lead_time_stats(rework_lead_times)

    if not primary_jobs:
        return {
            "error": "No primary jobs completed",
            "rework_jobs_completed": len(rework_jobs),
            "rework_summary": rework_stats,
            "raw_rework_lead_times": rework_lead_times,
        }

    # Step 7: リードタイムと差し戻しの集計
    lead_times = [job.history[-1]["time"] - job.created_at for job in primary_jobs]
    rework_counts = [job.rework_count for job in primary_jobs]
    rework_weights = [job.rework_weight for job in primary_jobs]
    
    # 増殖した小実験数
    proliferated_tasks = [sum(1 for t in job.tasks if t.generated_by == "REWORK") for job in primary_jobs]

    # CCDFの計算用データ (Step 7: CCDFを固定で出す)
    sorted_lt = np.sort(lead_times)
    ccdf_y = 1.0 - np.arange(1, len(sorted_lt) + 1) / len(sorted_lt)

    # WIP集計（ノード別平均WIP） (Step 7: WIP時系列)
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
            "completed_count": len(primary_jobs),
            "throughput": len(primary_jobs) / total_days,
            "lead_time_p50": float(np.percentile(lead_times, 50)),
            "lead_time_p90": float(np.percentile(lead_times, 90)),
            "lead_time_p95": float(np.percentile(lead_times, 95)),
            "avg_reworks": float(np.mean(rework_counts)),
            "max_reworks": int(np.max(rework_counts)),
            "avg_rework_weight": float(np.mean(rework_weights)),
            "avg_proliferated_tasks": float(np.mean(proliferated_tasks)),
            "avg_wip": avg_wip_total,
            "rework_jobs_completed": len(rework_jobs),
            "rework_completed_count": len(rework_jobs),
            "rework_throughput": len(rework_jobs) / total_days,
            "rework_avg_lead_time": rework_stats["avg"],
            "rework_lead_time_p50": rework_stats["p50"],
            "rework_lead_time_p90": rework_stats["p90"],
            "rework_lead_time_p95": rework_stats["p95"],
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
        "raw_rework_counts": rework_counts,
        "raw_rework_weights": rework_weights,
        "raw_proliferated_tasks": proliferated_tasks,
        "raw_rework_lead_times": rework_lead_times,
    }

    return metrics

import numpy as np
from typing import List, Dict, Any
from core.entities import Job

def calculate_metrics(completed_jobs: List[Job], nodes_stats: List[Dict[str, Any]], total_days: float, wip_history: List[Dict[str, Any]] = None):
    primary_jobs = [job for job in completed_jobs if not getattr(job, "is_rework_task", False) and not getattr(job, "is_rejected", False)]
    rework_jobs = [job for job in completed_jobs if getattr(job, "is_rework_task", False)]
    rejected_jobs = [job for job in completed_jobs if getattr(job, "is_rejected", False)]

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

    def _init_gate_bucket():
        return {
            "cycle_times": [],
            "wait_times": [],
            "decision_latencies": [],
            "outcomes": {"GO": 0, "CONDITIONAL": 0, "NO_GO": 0}
        }

    def _sum_job_components(job: Job, gate_buckets: Dict[str, Any], rework_source_bucket: Dict[str, Any]):
        work_time = 0.0
        wait_time = 0.0
        decision_latency = 0.0
        review_cost = 0.0

        if not getattr(job, "history", None):
            return work_time, wait_time, decision_latency, review_cost

        # Sort by time to stabilize ENQUEUE -> DECISION matching
        history = sorted(job.history, key=lambda h: (h.get("time", 0.0), h.get("event", "")))
        last_enqueue = {}
        last_review_wait = {}

        for h in history:
            event = h.get("event")
            node_id = h.get("node_id")

            if event == "START_WORK":
                work_time += float(h.get("effective_duration", 0.0) or 0.0)
                wait_time += float(h.get("wait_time", 0.0) or 0.0)
                continue

            if event == "REVIEW":
                wait_time += float(h.get("wait_time", 0.0) or 0.0)
                review_cost += float(h.get("cost_per_review", 0.0) or 0.0)
                if node_id:
                    last_review_wait[node_id] = float(h.get("wait_time", 0.0) or 0.0)
                continue

            if event == "BUNDLED":
                wait_time += float(h.get("wait_time", 0.0) or 0.0)
                continue

            if event == "ENQUEUE" and isinstance(node_id, str) and node_id.startswith("DR"):
                last_enqueue[node_id] = float(h.get("time", 0.0))
                continue

            if event == "DECISION" and isinstance(node_id, str) and node_id.startswith("DR"):
                decision_latency += float(h.get("decision_latency", 0.0) or 0.0)
                gate = node_id
                bucket = gate_buckets.setdefault(gate, _init_gate_bucket())
                start = last_enqueue.get(gate)
                if start is not None:
                    bucket["cycle_times"].append(float(h.get("time", 0.0)) - float(start))
                wait_val = last_review_wait.get(gate)
                if wait_val is not None:
                    bucket["wait_times"].append(float(wait_val))
                bucket["decision_latencies"].append(float(h.get("decision_latency", 0.0) or 0.0))
                outcome = h.get("outcome")
                if outcome in bucket["outcomes"]:
                    bucket["outcomes"][outcome] += 1

        if getattr(job, "is_rework_task", False):
            source_gate = getattr(job, "rework_source_gate", None) or "UNKNOWN"
            src_bucket = rework_source_bucket.setdefault(source_gate, {
                "count": 0,
                "work_total": 0.0,
                "wait_total": 0.0,
                "decision_total": 0.0,
                "flow_total": 0.0
            })
            src_bucket["count"] += 1
            src_bucket["work_total"] += work_time
            src_bucket["wait_total"] += wait_time
            src_bucket["decision_total"] += decision_latency
            if job.history:
                src_bucket["flow_total"] += float(job.history[-1]["time"]) - float(job.created_at)

        return work_time, wait_time, decision_latency, review_cost

    def _summarize_gate_bucket(bucket: Dict[str, Any]):
        cycle = bucket.get("cycle_times", [])
        waits = bucket.get("wait_times", [])
        decisions = bucket.get("decision_latencies", [])
        stats = {
            "cycle_avg": float(np.mean(cycle)) if cycle else 0.0,
            "cycle_p50": float(np.percentile(cycle, 50)) if cycle else 0.0,
            "cycle_p90": float(np.percentile(cycle, 90)) if cycle else 0.0,
            "cycle_p95": float(np.percentile(cycle, 95)) if cycle else 0.0,
            "wait_avg": float(np.mean(waits)) if waits else 0.0,
            "wait_p50": float(np.percentile(waits, 50)) if waits else 0.0,
            "wait_p90": float(np.percentile(waits, 90)) if waits else 0.0,
            "decision_avg": float(np.mean(decisions)) if decisions else 0.0,
            "decision_p50": float(np.percentile(decisions, 50)) if decisions else 0.0,
            "decision_p90": float(np.percentile(decisions, 90)) if decisions else 0.0,
            "count": int(len(cycle))
        }
        outcomes = bucket.get("outcomes", {}) or {}
        total = float(sum(outcomes.values()))
        if total > 0:
            stats["pass_rate"] = outcomes.get("GO", 0) / total
            stats["conditional_rate"] = outcomes.get("CONDITIONAL", 0) / total
            stats["nogo_rate"] = outcomes.get("NO_GO", 0) / total
        else:
            stats["pass_rate"] = 0.0
            stats["conditional_rate"] = 0.0
            stats["nogo_rate"] = 0.0
        return stats

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
            "rejected_count": len(rejected_jobs),
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
        "dr_gate": {},
        "loss": {},
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

    # --- Loss decomposition & DR cycle times ---
    primary_gate_buckets: Dict[str, Any] = {}
    rework_gate_buckets: Dict[str, Any] = {}
    rework_by_source_gate: Dict[str, Any] = {}

    primary_work = primary_wait = primary_decision = primary_review_cost = 0.0
    rework_work = rework_wait = rework_decision = rework_review_cost = 0.0
    primary_flow_total = 0.0
    rework_flow_total = 0.0

    for job in primary_jobs:
        w, wt, dlat, rcost = _sum_job_components(job, primary_gate_buckets, rework_by_source_gate)
        primary_work += w
        primary_wait += wt
        primary_decision += dlat
        primary_review_cost += rcost
        if job.history:
            primary_flow_total += float(job.history[-1]["time"]) - float(job.created_at)

    for job in rework_jobs:
        w, wt, dlat, rcost = _sum_job_components(job, rework_gate_buckets, rework_by_source_gate)
        rework_work += w
        rework_wait += wt
        rework_decision += dlat
        rework_review_cost += rcost
        if job.history:
            rework_flow_total += float(job.history[-1]["time"]) - float(job.created_at)

    primary_count = len(primary_jobs)
    rework_count = len(rework_jobs)

    time_primary_avg = {
        "count": primary_count,
        "work_total": primary_work,
        "wait_total": primary_wait,
        "decision_total": primary_decision,
        "flow_total": primary_flow_total,
        "avg_work": primary_work / primary_count if primary_count else 0.0,
        "avg_wait": primary_wait / primary_count if primary_count else 0.0,
        "avg_decision": primary_decision / primary_count if primary_count else 0.0,
        "avg_flow": primary_flow_total / primary_count if primary_count else 0.0,
    }
    time_rework_avg = {
        "count": rework_count,
        "work_total": rework_work,
        "wait_total": rework_wait,
        "decision_total": rework_decision,
        "flow_total": rework_flow_total,
        "avg_work": rework_work / rework_count if rework_count else 0.0,
        "avg_wait": rework_wait / rework_count if rework_count else 0.0,
        "avg_decision": rework_decision / rework_count if rework_count else 0.0,
        "avg_flow": rework_flow_total / rework_count if rework_count else 0.0,
    }

    loss_per_primary = 0.0
    rework_time_per_primary = 0.0
    if primary_count > 0:
        loss_per_primary = (primary_wait + primary_decision + rework_flow_total) / primary_count
        rework_time_per_primary = rework_flow_total / primary_count

    metrics["loss"] = {
        "time": {
            "primary": time_primary_avg,
            "rework": time_rework_avg,
            "loss_per_primary": loss_per_primary,
            "rework_time_per_primary": rework_time_per_primary,
            "loss_time_total": primary_wait + primary_decision + rework_flow_total
        },
        "cost": {
            "review_total": primary_review_cost + rework_review_cost,
            "review_primary": primary_review_cost,
            "review_rework": rework_review_cost,
            "review_per_primary": (primary_review_cost + rework_review_cost) / primary_count if primary_count else 0.0,
            "rework_review_per_primary": rework_review_cost / primary_count if primary_count else 0.0,
            "rework_review_ratio": rework_review_cost / (primary_review_cost + rework_review_cost) if (primary_review_cost + rework_review_cost) > 0 else 0.0
        },
        "rework_by_source_gate": rework_by_source_gate
    }

    dr_gate_summary_primary = {}
    for gate, bucket in primary_gate_buckets.items():
        dr_gate_summary_primary[gate] = _summarize_gate_bucket(bucket)
    dr_gate_summary_rework = {}
    for gate, bucket in rework_gate_buckets.items():
        dr_gate_summary_rework[gate] = _summarize_gate_bucket(bucket)

    metrics["dr_gate"] = {
        "primary": dr_gate_summary_primary,
        "rework": dr_gate_summary_rework
    }

    return metrics

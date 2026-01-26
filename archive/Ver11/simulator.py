import os
import sys
import numpy as np
import concurrent.futures
import multiprocessing

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.entities import Job, Approver
from core.engine import SimulationEngine
from core.gates import WorkGate, MeetingGate
from core.policies import ReworkPolicy
from analysis.metrics import calculate_metrics
from runner.adapters import setup_standard_flow

def simulate_standard_flow(
    days=365,
    arrival_rate=0.5,
    small_exp_duration=5,
    proto_duration=20,
    bundle_size=3,
    dr_period=90,
    dr_capacity=10,
    dr_quality=0.8,
    rework_load_factor=1.0,
    max_rework_cycles=5,
    decay=0.7,
    n_senior=1,
    n_coordinator=0,
    n_new=0,
    seed=None,
    sampling_interval=1.0,
    **kwargs
):
    """
    標準フロー版（SMALL_EXP→PROTO→BUNDLE→DR_GATE）
    """
    rng = np.random.default_rng(seed)

    engine = setup_standard_flow(
        rng,
        days=days,
        arrival_rate=arrival_rate,
        small_exp_duration=small_exp_duration,
        proto_duration=proto_duration,
        bundle_size=bundle_size,
        dr_period=dr_period,
        dr_capacity=dr_capacity,
        dr_quality=dr_quality,
        rework_load_factor=rework_load_factor,
        max_rework_cycles=max_rework_cycles,
        decay=decay,
        n_senior=n_senior,
        n_coordinator=n_coordinator,
        n_new=n_new,
        **kwargs
    )

    # sampling_interval を engine に反映（setup側が engine を生成しているため後付け）
    engine.sampling_interval = float(sampling_interval) if sampling_interval else 0.0
    engine.next_sample_time = 0.0

    engine.run(days)

    nodes_stats = [node.stats() for node in engine.nodes.values()]
    m = calculate_metrics(
        engine.results["completed_jobs"],
        nodes_stats,
        days,
        wip_history=engine.results.get("wip_history", [])
    )

    if "error" in m:
        summary = {
            "approved_count": 0,
            "throughput": 0,
            "avg_wait": 0,
            "p90_wait": 0,
            "p95_wait": 0,
            "avg_wip": 0,
            "avg_reworks": 0,
            "max_reworks": 0,
        }
    else:
        summary = {
            "approved_count": m["summary"]["completed_count"],
            "throughput": m["summary"]["throughput"],
            "avg_wait": m["summary"]["lead_time_p50"],
            "p90_wait": m["summary"]["lead_time_p90"],
            "p95_wait": m["summary"]["lead_time_p95"],
            "avg_wip": m["summary"].get("avg_wip", 0),
            "avg_reworks": m["summary"]["avg_reworks"],
            "max_reworks": m["summary"]["max_reworks"],
        }

    # Job詳細ログ (摩擦の検証用など)
    job_logs = []
    
    def collect_job_logs(job_list):
        for job in job_list:
            # 自身の履歴から抽出
            for h in job.history:
                if h.get("event") == "START_WORK":
                    job_logs.append({
                        "job_id": job.job_id,
                        "node_id": h.get("node_id"),
                        "start_time": h.get("time"),
                        "base_duration": h.get("base_duration"),
                        "effective_duration": h.get("effective_duration"),
                        "friction_multiplier": h.get("friction_multiplier"),
                        "n_servers": h.get("n_servers"),
                        "wait_time": h.get("wait_time")
                    })
            # バンドルされている場合は子要素も探索
            if hasattr(job, "bundle_items") and job.bundle_items:
                collect_job_logs(job.bundle_items)

    collect_job_logs(engine.results["completed_jobs"])
    completed_count = len(engine.results["completed_jobs"])
    

    return {
        "summary": summary,
        "metrics": m,
        "logs": {
            "lead_times": m.get("raw_lead_times", []),
            "rework_counts": m.get("raw_rework_counts", []),
            "rework_weights": m.get("raw_rework_weights", []),
            "proliferated_tasks": m.get("raw_proliferated_tasks", []),
            "wip_history": m.get("wip", {}).get("history", []),
            "job_logs": job_logs,
        }
    }

def _single_trial(args):
    params, seed = args
    return simulate_standard_flow(**params, seed=seed)

def run_monte_carlo(n_trials=100, use_parallel=True, base_seed=42, **params):
    """
    モンテカルロ（標準フロー版）
    Step 8.2: MonteCarlo - 各シナリオで n_trials 回実行して分布を得る
    seedは base_seed + trial_id (並列化対応の独立RNG)
    """
    args = [(params, base_seed + i) for i in range(n_trials)]

    if use_parallel and n_trials > 1:
        cpu_count = multiprocessing.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            results = list(executor.map(_single_trial, args))
    else:
        results = [_single_trial(arg) for arg in args]

    return results

def latin_hypercube_sampling(n_samples, param_ranges):
    """
    ラティス法 (Latin Hypercube Sampling) によるパラメータ空間のサンプリング
    param_ranges: { "param_name": (min, max), ... }
    """
    n_params = len(param_ranges)
    lower_limits = np.arange(0, n_samples) / n_samples
    upper_limits = np.arange(1, n_samples + 1) / n_samples
    
    u_samples = np.zeros((n_samples, n_params))
    for i in range(n_params):
        samples = np.random.uniform(low=lower_limits, high=upper_limits, size=n_samples)
        np.random.shuffle(samples)
        u_samples[:, i] = samples
    
    # 実際の値の範囲にスケーリング
    scaled_samples = []
    param_names = list(param_ranges.keys())
    for j in range(n_samples):
        sample_dict = {}
        for i, name in enumerate(param_names):
            p_min, p_max = param_ranges[name]
            sample_dict[name] = p_min + u_samples[j, i] * (p_max - p_min)
        scaled_samples.append(sample_dict)
        
    return scaled_samples

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

def _const_func(val):
    return val

def _exp_sample(rng, rate):
    return rng.exponential(1.0 / rate)

def simulate_flow_v6(
    days=90,
    arrival_vp_per_day=5.0,
    approvers=1,
    service_vp_per_day=6.0,
    p_rework=0.1,
    rework_load_factor=0.5,
    skill_factor=1.0,
    skill_at_dependency_resolution=0.0,
    dependency_level=0.1,
    coordination_penalty=1.0,
    seed=None,
    sampling_interval=1.0
):
    """
    DX4MGR Sim Ver6 - 新エンジンへのアダプタ実装
    """
    rng = np.random.default_rng(seed)
    engine = SimulationEngine(rng=rng)
    
    # 1. パラメータ変換 (旧Ver6のロジックを一部借用して実効レートを出す)
    adjusted_dependency = dependency_level / (1.0 + skill_at_dependency_resolution)
    coordination_factor = 1.0 / (1.0 + coordination_penalty * adjusted_dependency)
    effective_service_rate = service_vp_per_day * skill_factor * coordination_factor
    
    from functools import partial
    # 2. ゲート構築 (旧モデルに近い単一ゲート構成)
    # 差し戻しポリシー
    rework_policy = ReworkPolicy(
        rework_load_factor=rework_load_factor,
        weight_dist_func=partial(_const_func, 1.0), # 旧モデルは固定負荷増大
        max_rework_cycles=10,
        decay=1.0 # 旧モデルは減衰なし
    )
    
    # 承認者
    apps = [Approver(f"app_{i}", "Standard", 1, 1.0 - p_rework) for i in range(approvers)]
    
    # 旧モデルは「作業時間」があるDESだったので、WorkGateとして実装するのが近い
    # ただし、旧モデルは「サービス終了時に判定」していたので、
    # MeetingGate (period=0) に近いが、WorkGateの後に判定ノードがある構成にする
    
    # ここでは指示書の「GateNode」を活かすため、最小構成のワークワークフローを作る
    gate = WorkGate(
        "MAIN_GATE", engine,
        n_servers=approvers,
        duration_dist=partial(_exp_sample, rng, effective_service_rate),
        next_node_id="DR"
    )
    
    # 判定ゲート (即時実行されるように period は極小)
    dr = MeetingGate(
        "DR", engine,
        period_days=0.001, 
        approvers=[Approver("eval", "Eval", 999, 1.0 - p_rework)],
        next_node_id=None,
        rework_node_id="MAIN_GATE",
        rework_policy=rework_policy
    )
    
    engine.add_node(gate)
    engine.add_node(dr)
    
    # 3. 流入スケジュール
    t = 0
    while t < days:
        t += rng.exponential(1.0 / arrival_vp_per_day)
        if t < days:
            job = Job(job_id=f"j_{t:.2f}", created_at=t)
            engine.schedule_event(t, "ARRIVAL", {"job": job, "target_node": "MAIN_GATE"})
            
    # 4. 実行
    engine.run(days)
    
    # 5. 指標変換
    nodes_stats = [node.stats() for node in engine.nodes.values()]
    m = calculate_metrics(engine.results["completed_jobs"], nodes_stats, days)
    
    if "error" in m:
        summary = {
            "approved_count": 0, "throughput": 0, "avg_wait": 0, "p90_wait": 0, "p95_wait": 0,
            "avg_wip": 0, "avg_reworks": 0, "max_reworks": 0, "utilization": 0
        }
    else:
        summary = {
            "approved_count": m["summary"]["completed_count"],
            "throughput": m["summary"]["throughput"],
            "avg_wait": m["summary"]["lead_time_p50"], # 簡易的に
            "p90_wait": m["summary"]["lead_time_p90"],
            "p95_wait": m["summary"]["lead_time_p95"],
            "avg_wip": 0, # TODO
            "avg_reworks": m["summary"]["avg_reworks"],
            "max_reworks": m["summary"]["max_reworks"],
            "utilization": 0 # TODO
        }
        
    return {
        "summary": summary,
        "logs": {
            "wait_times": [], # TODO
            "lead_times": m.get("raw_lead_times", []),
            "wip_history": [],
            "rework_counts": m.get("raw_rework_counts", [])
        },
        "params": {
            "effective_service_rate": effective_service_rate,
            "coordination_factor": coordination_factor,
            "adjusted_dependency": adjusted_dependency,
            "rework_load_factor": rework_load_factor
        }
    }

def _single_trial(args):
    params, seed = args
    return simulate_flow_v6(**params, seed=seed)

def run_monte_carlo_v6(n_trials=100, use_parallel=True, **params):
    """
    モンテカルロシミュレーションの実行 (並列処理対応) - Ver6
    """
    args = [(params, i) for i in range(n_trials)]
    
    if use_parallel and n_trials > 1:
        # 並列実行
        cpu_count = multiprocessing.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            results = list(executor.map(_single_trial, args))
    else:
        # 逐次実行
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

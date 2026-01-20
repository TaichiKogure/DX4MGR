import random
import heapq
import numpy as np
import concurrent.futures
import multiprocessing

def simulate_flow_v5(
    days=90,
    arrival_vp_per_day=5.0,
    approvers=1,
    service_vp_per_day=6.0,
    p_rework=0.1,
    skill_factor=1.0,
    skill_at_dependency_resolution=0.0,
    dependency_level=0.1,
    coordination_penalty=1.0,
    seed=None,
    sampling_interval=1.0
):
    """
    離散イベントシミュレーション (Discrete Event Simulation: DES)
    """
    if seed is not None:
        random.seed(seed)
    
    # 1. 相互作用による実効依存度 (理論モデル: memo3 7行目参照)
    adjusted_dependency = dependency_level / (1.0 + skill_at_dependency_resolution)
    
    # 2. 調整コスト因子 (待ち行列理論の拡張)
    coordination_factor = 1.0 / (1.0 + coordination_penalty * adjusted_dependency)
    
    # 3. 実効的な処理能力 (スキルと組織制約の合成)
    effective_service_rate = service_vp_per_day * skill_factor * coordination_factor
    
    # 4. シミュレーション実行
    t = 0.0
    approver_available = [0.0 for _ in range(approvers)]
    heapq.heapify(approver_available)
    
    wait_times = []
    lead_times = []
    wip_log = [] # (time, wip_count)
    
    events = [] # (time, event_type, data)
    
    if arrival_vp_per_day > 0:
        first_arrival = random.expovariate(arrival_vp_per_day)
        heapq.heappush(events, (first_arrival, 0, {"arrival_time": first_arrival}))
    
    current_wip = 0
    waiting_queue = [] 
    next_sample_t = 0.0
    
    while events:
        event_t, event_type, data = heapq.heappop(events)
        if event_t > days:
            break
            
        while next_sample_t <= event_t:
            wip_log.append((next_sample_t, current_wip))
            next_sample_t += sampling_interval

        if event_type == 0: # 到着
            current_wip += 1
            arrival_t = data["arrival_time"]
            next_arrival_t = event_t + random.expovariate(arrival_vp_per_day)
            heapq.heappush(events, (next_arrival_t, 0, {"arrival_time": next_arrival_t}))
            
            if len(approver_available) > 0:
                free_time = heapq.heappop(approver_available)
                start_time = max(event_t, free_time)
                service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
                finish_time = start_time + service_time
                heapq.heappush(events, (finish_time, 1, {"arrival_time": arrival_t, "start_time": start_time}))
            else:
                waiting_queue.append(arrival_t)
            
        elif event_type == 1: # サービス終了
            if random.random() < p_rework:
                waiting_queue.insert(0, data["arrival_time"])
            else:
                wait_time = data["start_time"] - data["arrival_time"]
                lead_time = event_t - data["arrival_time"]
                wait_times.append(wait_time)
                lead_times.append(lead_time)
                current_wip -= 1
            
            if waiting_queue:
                next_job_arrival_t = waiting_queue.pop(0)
                start_time = event_t 
                service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
                finish_time = start_time + service_time
                heapq.heappush(events, (finish_time, 1, {"arrival_time": next_job_arrival_t, "start_time": start_time}))
            else:
                heapq.heappush(approver_available, event_t)

    while next_sample_t <= days:
        wip_log.append((next_sample_t, current_wip))
        next_sample_t += sampling_interval

    # 統計情報の要約
    summary = {
        "approved_count": len(lead_times),
        "throughput": len(lead_times) / days,
        "avg_wait": np.mean(wait_times) if wait_times else 0,
        "p90_wait": np.percentile(wait_times, 90) if wait_times else 0,
        "p95_wait": np.percentile(wait_times, 95) if wait_times else 0,
        "avg_wip": np.mean([w for t, w in wip_log]),
        "utilization": arrival_vp_per_day / (approvers * effective_service_rate) if effective_service_rate > 0 else 0
    }
    
    return {
        "summary": summary,
        "logs": {
            "wait_times": wait_times,
            "lead_times": lead_times,
            "wip_history": wip_log
        },
        "params": {
            "effective_service_rate": effective_service_rate,
            "coordination_factor": coordination_factor,
            "adjusted_dependency": adjusted_dependency
        }
    }

def _single_trial(args):
    params, seed = args
    return simulate_flow_v5(**params, seed=seed)

def run_monte_carlo_v5(n_trials=100, use_parallel=True, **params):
    """
    モンテカルロシミュレーションの実行 (並列処理対応)
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

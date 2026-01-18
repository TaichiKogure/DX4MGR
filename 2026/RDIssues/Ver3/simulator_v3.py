import random
import heapq
import numpy as np

def simulate_flow_v3(
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
    sampling_interval=1.0  # WIPをサンプリングする間隔
):
    if seed is not None:
        random.seed(seed)
    
    # 1. 相互作用による実効依存度
    adjusted_dependency = dependency_level / (1.0 + skill_at_dependency_resolution)
    
    # 2. 調整コスト因子
    coordination_factor = 1.0 / (1.0 + coordination_penalty * adjusted_dependency)
    
    # 3. 実効的な処理能力
    effective_service_rate = service_vp_per_day * skill_factor * coordination_factor
    
    # 4. 離散イベントシミュレーション
    t = 0.0
    approver_available = [0.0 for _ in range(approvers)]
    heapq.heapify(approver_available)
    
    wait_times = []
    lead_times = []
    wip_log = [] # (time, wip_count)
    
    # イベント管理用のヒープ (time, event_type, data)
    # event_type: 0 = arrival, 1 = service_finish
    events = []
    
    # 最初の到着をスケジュール
    if arrival_vp_per_day > 0:
        first_arrival = random.expovariate(arrival_vp_per_day)
        heapq.heappush(events, (first_arrival, 0, {"arrival_time": first_arrival}))
    
    # WIP追跡用
    current_wip = 0
    waiting_queue = [] # (arrival_time)
    
    # サンプリング用のタイマー
    next_sample_t = 0.0
    
    while events:
        event_t, event_type, data = heapq.heappop(events)
        if event_t > days:
            break
            
        # サンプリング
        while next_sample_t <= event_t:
            wip_log.append((next_sample_t, current_wip))
            next_sample_t += sampling_interval

        if event_type == 0: # 到着
            current_wip += 1
            arrival_t = data["arrival_time"]
            
            # 次の到着をスケジュール
            next_arrival_t = event_t + random.expovariate(arrival_vp_per_day)
            heapq.heappush(events, (next_arrival_t, 0, {"arrival_time": next_arrival_t}))
            
            # 空いている承認者がいるか確認
            if len(approver_available) > 0:
                # 承認者に割り当て
                free_time = heapq.heappop(approver_available)
                start_time = max(event_t, free_time)
                
                # サービス終了をスケジュール
                service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
                finish_time = start_time + service_time
                heapq.heappush(events, (finish_time, 1, {"arrival_time": arrival_t, "start_time": start_time}))
            else:
                # 待ち行列に入れる
                waiting_queue.append(arrival_t)
            
        elif event_type == 1: # サービス終了
            # 完了処理 (再作業判定)
            if random.random() < p_rework:
                # 再作業として再度並ぶ
                # 承認者が空いたので、即座に自分または次の人を割り当てる
                # ただし自分は再度並び直す必要がある。
                # 簡略化のため、再作業は「現在の終了時刻」を「新しい到着時刻」として待ち行列の先頭に入れる
                waiting_queue.insert(0, data["arrival_time"])
            else:
                # 完了
                wait_time = data["start_time"] - data["arrival_time"]
                lead_time = event_t - data["arrival_time"]
                wait_times.append(wait_time)
                lead_times.append(lead_time)
                current_wip -= 1
            
            # 次のジョブがあれば開始
            if waiting_queue:
                next_job_arrival_t = waiting_queue.pop(0)
                start_time = event_t # 現在時刻
                service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
                finish_time = start_time + service_time
                heapq.heappush(events, (finish_time, 1, {"arrival_time": next_job_arrival_t, "start_time": start_time}))
            else:
                # 承認者を空き状態に戻す
                heapq.heappush(approver_available, event_t)

    # 残りの期間をサンプリング
    while next_sample_t <= days:
        wip_log.append((next_sample_t, current_wip))
        next_sample_t += sampling_interval

    return {
        "summary": {
            "approved_count": len(lead_times),
            "throughput": len(lead_times) / days,
            "avg_wait": np.mean(wait_times) if wait_times else 0,
            "p90_wait": np.percentile(wait_times, 90) if wait_times else 0,
            "p95_wait": np.percentile(wait_times, 95) if wait_times else 0,
            "avg_wip": np.mean([w for t, w in wip_log]),
            "utilization": arrival_vp_per_day / (approvers * effective_service_rate) if effective_service_rate > 0 else 0
        },
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

def run_monte_carlo(n_trials=100, **params):
    results = []
    for i in range(n_trials):
        res = simulate_flow_v3(**params, seed=i)
        results.append(res)
    return results

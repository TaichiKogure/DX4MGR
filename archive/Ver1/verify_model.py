import random
import heapq
import math

def simulate_flow_model(
    days=90,
    arrival_vp_per_day=14.0,
    approvers=2,
    service_vp_per_day=9.0,
    p_rework=0.1,
    dependency_level=0.1,
    coordination_penalty=1.0,
    seed=42
):
    """
    R&Dフロー健全性診断のためのパラメータスタディ用統合モデル
    
    【モデルの構成要素】
    1. 待ち行列モデル (Queueing): 承認者がボトルネックになる「待ち」を再現
    2. 再作業ループ (Rework): 承認での差し戻しによるロスを再現
    3. 調整コスト (Coordination): 依存関係による効率低下を再現
    """
    random.seed(seed)
    
    # 1. 調整コストによる実効キャパシティの減衰 (モデル3の適用)
    # 依存関係が強いほど、チーム全体の動けるキャパシティが削られる
    coordination_factor = 1.0 / (1.0 + coordination_penalty * dependency_level)
    
    # 実効的な処理能力
    effective_service_rate = service_vp_per_day * coordination_factor
    
    # 2. 離散イベントシミュレーション (モデル1: 待ち行列)
    t = 0.0
    approver_available = [0.0 for _ in range(approvers)]
    heapq.heapify(approver_available)
    
    approved_count = 0
    total_waits = 0.0
    total_rework_attempts = 0
    
    # シミュレーションループ
    while True:
        # 次の仕事（価値ポイント: VP）が到着するまでの時間
        if arrival_vp_per_day <= 0: break
        t += random.expovariate(arrival_vp_per_day)
        if t > days: break
        
        current_job_t = t
        is_done = False
        
        # 3. 再作業ループ (モデル2)
        # 承認されるまでループする（実際には処理を繰り返す）
        while not is_done:
            # 最も早く空く承認者を選択
            free_time = heapq.heappop(approver_available)
            start_time = max(current_job_t, free_time)
            
            wait_time = start_time - current_job_t
            total_waits += wait_time
            
            # 承認処理時間
            service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
            finish_time = start_time + service_time
            
            if finish_time > days:
                heapq.heappush(approver_available, finish_time)
                break # 期間終了
            
            # 承認判定 (差し戻し確率 p_rework)
            if random.random() > p_rework:
                # 承認成功
                approved_count += 1
                is_done = True
                heapq.heappush(approver_available, finish_time)
            else:
                # 差し戻し発生
                total_rework_attempts += 1
                current_job_t = finish_time # 次の試行は今の終了後から
                heapq.heappush(approver_available, finish_time)

    avg_wait = total_waits / approved_count if approved_count > 0 else 0.0
    throughput = approved_count / days
    
    return {
        "期間(日)": days,
        "承認された価値ポイント(VP)": approved_count,
        "1日あたりのスループット(VP/day)": round(throughput, 2),
        "平均待ち時間(日)": round(avg_wait, 2),
        "再作業発生回数": total_rework_attempts,
        "稼働率(目安)": round(arrival_vp_per_day / (approvers * effective_service_rate), 2) if effective_service_rate > 0 else "inf"
    }

def run_parameter_study():
    print("=== R&D Flow 健全性診断 パラメータスタディ (Ver1) ===")
    
    # ケース1: 基本設定 (標準的なチーム)
    print("\n【ケース1: 基本設定】")
    res1 = simulate_flow_model(arrival_vp_per_day=5, approvers=1, service_vp_per_day=6, p_rework=0.1)
    for k, v in res1.items(): print(f"  {k}: {v}")
    
    # ケース2: 承認ボトルネック (仕事量に対して承認能力がギリギリ)
    print("\n【ケース2: 承認ボトルネック (仕事量増加)】")
    res2 = simulate_flow_model(arrival_vp_per_day=5.8, approvers=1, service_vp_per_day=6, p_rework=0.1)
    for k, v in res2.items(): print(f"  {k}: {v}")
    
    # ケース3: 手戻りが多い (品質や定義が不明確)
    print("\n【ケース3: 手戻りが多い (再作業率30%)】")
    res3 = simulate_flow_model(arrival_vp_per_day=5, approvers=1, service_vp_per_day=6, p_rework=0.3)
    for k, v in res3.items(): print(f"  {k}: {v}")

    # ケース4: 組織の肥大化 (依存関係と調整コストの増大)
    print("\n【ケース4: 組織の肥大化 (依存関係増、調整効率低下)】")
    res4 = simulate_flow_model(arrival_vp_per_day=5, approvers=1, service_vp_per_day=6, p_rework=0.1, dependency_level=0.5, coordination_penalty=2.0)
    for k, v in res4.items(): print(f"  {k}: {v}")

if __name__ == "__main__":
    run_parameter_study()

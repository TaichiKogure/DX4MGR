import random
import heapq
import math

def simulate_flow_model_v2(
    days=90,
    arrival_vp_per_day=5.0,
    approvers=1,
    service_vp_per_day=6.0,
    p_rework=0.1,
    # --- Step2: パラメータ補強 ---
    skill_factor=1.0,               # 意思決定・専門能力 (1.0が標準)
    skill_at_dependency_resolution=0.0, # 依存関係解消能力 (0.0〜)
    dependency_level=0.1,           # 組織的な依存度合い (0.0〜1.0)
    coordination_penalty=1.0,       # 依存によるオーバーヘッドの重さ
    seed=42
):
    """
    R&Dフロー健全性診断 拡張モデル (Ver2: スキル×依存性)
    
    【拡張要素】
    1. Skill Factor: 承認者のスキルが高いと、1件あたりの判断速度(service_rate)が向上する
    2. Dependency Resolution: スキルが高い人は依存関係による待ちや調整コストを軽減できる
    3. Interaction: スキルと依存性の相互作用を数式化
    """
    random.seed(seed)
    
    # 1. スキルと依存性の相互作用による実効依存度の計算
    # 高スキルの人が依存を解く役割を果たす構造 (memo2.txt 699行目参照)
    adjusted_dependency = dependency_level / (1.0 + skill_at_dependency_resolution)
    
    # 2. 調整コストによる実効キャパシティの減衰
    coordination_factor = 1.0 / (1.0 + coordination_penalty * adjusted_dependency)
    
    # 3. 実効的な処理能力 (スキル因子を乗算)
    effective_service_rate = service_vp_per_day * skill_factor * coordination_factor
    
    # 4. 離散イベントシミュレーション (待ち行列 + 再作業)
    t = 0.0
    approver_available = [0.0 for _ in range(approvers)]
    heapq.heapify(approver_available)
    
    approved_count = 0
    total_waits = 0.0
    total_rework_attempts = 0
    
    while True:
        if arrival_vp_per_day <= 0: break
        t += random.expovariate(arrival_vp_per_day)
        if t > days: break
        
        current_job_t = t
        is_done = False
        
        while not is_done:
            free_time = heapq.heappop(approver_available)
            start_time = max(current_job_t, free_time)
            
            wait_time = start_time - current_job_t
            total_waits += wait_time
            
            # 承認処理時間
            service_time = random.expovariate(effective_service_rate) if effective_service_rate > 0 else 1e9
            finish_time = start_time + service_time
            
            if finish_time > days:
                heapq.heappush(approver_available, finish_time)
                break
            
            if random.random() > p_rework:
                approved_count += 1
                is_done = True
                heapq.heappush(approver_available, finish_time)
            else:
                total_rework_attempts += 1
                current_job_t = finish_time
                heapq.heappush(approver_available, finish_time)

    avg_wait = total_waits / approved_count if approved_count > 0 else 0.0
    throughput = approved_count / days
    
    return {
        "パラメータ": {
            "skill_factor": skill_factor,
            "dependency_resolution_skill": skill_at_dependency_resolution,
            "dependency_level": dependency_level,
            "effective_service_rate": round(effective_service_rate, 2)
        },
        "結果": {
            "承認された価値ポイント(VP)": approved_count,
            "1日あたりのスループット(VP/day)": round(throughput, 2),
            "平均待ち時間(日)": round(avg_wait, 2),
            "再作業発生回数": total_rework_attempts,
            "稼働率(目安)": round(arrival_vp_per_day / (approvers * effective_service_rate), 2) if effective_service_rate > 0 else "inf"
        }
    }

def run_ver2_study():
    print("=== R&D Flow 健全性診断 パラメータスタディ (Ver2: パラメータ補強版) ===")
    
    # A: 高スキル・低依存 (理想的なエキスパートチーム)
    print("\n【ケースA: 高スキル・低依存 (エキスパートチーム)】")
    resA = simulate_flow_model_v2(skill_factor=1.5, skill_at_dependency_resolution=1.0, dependency_level=0.1)
    print_results(resA)
    
    # B: 低スキル・高依存 (新人中心/組織分断)
    print("\n【ケースB: 低スキル・高依存 (組織の壁とスキル不足)】")
    resB = simulate_flow_model_v2(skill_factor=0.7, skill_at_dependency_resolution=0.0, dependency_level=0.5, coordination_penalty=2.0)
    print_results(resB)
    
    # C: 依存は高いが、調整スキルでカバーしている場合
    print("\n【ケースC: 高依存を「調整スキル」で突破している】")
    resC = simulate_flow_model_v2(skill_factor=1.0, skill_at_dependency_resolution=2.0, dependency_level=0.5, coordination_penalty=2.0)
    print_results(resC)

def print_results(res):
    print("  [パラメータ]")
    for k, v in res["パラメータ"].items(): print(f"    {k}: {v}")
    print("  [シミュレーション結果]")
    for k, v in res["結果"].items(): print(f"    {k}: {v}")

if __name__ == "__main__":
    run_ver2_study()

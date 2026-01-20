import numpy as np
from core.entities import Job, Approver, TaskType
from core.engine import SimulationEngine
from core.gates import WorkGate, BundleGate, MeetingGate
from core.policies import ReworkPolicy

def _exp_dist(rng, scale):
    return rng.exponential(scale)

def _beta_dist(rng, a, b):
    return rng.beta(a, b)

def _constant_dist(val):
    return val

def setup_standard_flow(rng, **params):
    engine = SimulationEngine(rng=rng)
    
    # パラメータの抽出
    days = params.get("days", 365)
    arrival_rate = params.get("arrival_rate", 0.5)
    
    # Step 5: 差し戻しポリシー
    from functools import partial
    # Beta分布のパラメータを抽出 (default 2, 5)
    rework_beta_a = params.get("rework_beta_a", 2.0)
    rework_beta_b = params.get("rework_beta_b", 5.0)
    
    rework_policy = ReworkPolicy(
        rework_load_factor=params.get("rework_load_factor", 0.5),
        weight_dist_func=partial(_beta_dist, rng, rework_beta_a, rework_beta_b),
        max_rework_cycles=params.get("max_rework_cycles", 5),
        decay=params.get("decay", 0.7),
        task_type_mix=params.get("rework_task_type_mix", 1.0)
    )
    
    # Step 6: 承認者の構成
    from core.entities import APPROVER_TYPES
    approvers = []
    # 役割タイプとして扱う（匿名化）
    n_senior = int(params.get("n_senior", 1))
    n_coordinator = int(params.get("n_coordinator", 0))
    n_new = int(params.get("n_new", 0))
    
    for i in range(n_senior):
        approvers.append(Approver(f"senior_{i}", "Senior", APPROVER_TYPES["Senior"]["capacity"], APPROVER_TYPES["Senior"]["quality"]))
    for i in range(n_coordinator):
        approvers.append(Approver(f"coord_{i}", "Coordinator", APPROVER_TYPES["Coordinator"]["capacity"], APPROVER_TYPES["Coordinator"]["quality"]))
    for i in range(n_new):
        approvers.append(Approver(f"new_{i}", "New", APPROVER_TYPES["New"]["capacity"], APPROVER_TYPES["New"]["quality"]))
    
    # もし誰もいなければデフォルトでSenior1人を追加（Ver9.1互換性のため）
    if not approvers:
        approvers.append(Approver("default", "Senior", 10, 0.9))
    
    # ノード構築
    # 1. 小実験
    small_exp_gate = WorkGate(
        "SMALL_EXP", engine, 
        n_servers=999, # 無制限
        duration_dist=partial(_exp_dist, rng, params.get("small_exp_duration", 5)),
        next_node_id="PROTO",
        task_type=TaskType.SMALL_EXP
    )
    
    # 2. 試作
    proto_gate = WorkGate(
        "PROTO", engine,
        n_servers=5, # 試作ラインは有限
        duration_dist=partial(_exp_dist, rng, params.get("proto_duration", 20)),
        next_node_id="BUNDLE",
        task_type=TaskType.PROTO_TEST
    )
    
    # 3. バンドル
    bundle_gate = BundleGate(
        "BUNDLE", engine,
        bundle_size_dist=partial(_constant_dist, params.get("bundle_size", 3)),
        next_node_id="DR_GATE"
    )
    
    # 4. DR
    dr_gate = MeetingGate(
        "DR_GATE", engine,
        period_days=params.get("dr_period", 90),
        approvers=approvers,
        next_node_id=None, # 完了
        rework_node_id="SMALL_EXP",
        rework_policy=rework_policy,
        conditional_prob_ratio=params.get("conditional_prob_ratio", 0.8),
        decision_latency_days=params.get("decision_latency_days", 0.0)
    )
    
    engine.add_node(small_exp_gate)
    engine.add_node(proto_gate)
    engine.add_node(bundle_gate)
    engine.add_node(dr_gate)
    
    # 流入イベントのスケジュール
    t = 0
    while t < days:
        t += rng.exponential(1.0 / arrival_rate)
        if t < days:
            job = Job(job_id=f"job_{t:.2f}", created_at=t)
            engine.schedule_event(t, "ARRIVAL", {"job": job, "target_node": "SMALL_EXP"})
            
    return engine

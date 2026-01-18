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
    
    # 差し戻しポリシー
    from functools import partial
    rework_policy = ReworkPolicy(
        rework_load_factor=params.get("rework_load_factor", 0.5),
        weight_dist_func=partial(_beta_dist, rng, 2, 5),
        max_rework_cycles=params.get("max_rework_cycles", 5),
        decay=params.get("decay", 0.7)
    )
    
    # 承認者 (Step 6)
    approvers = [
        Approver(f"app_{i}", "Senior", 10, 0.9) for i in range(params.get("approvers", 1))
    ]
    
    # ノード構築
    # 1. 小実験
    small_exp_gate = WorkGate(
        "SMALL_EXP", engine, 
        n_servers=999, # 無制限
        duration_dist=partial(_exp_dist, rng, params.get("small_exp_duration", 5)),
        next_node_id="PROTO"
    )
    
    # 2. 試作
    proto_gate = WorkGate(
        "PROTO", engine,
        n_servers=5, # 試作ラインは有限
        duration_dist=partial(_exp_dist, rng, params.get("proto_duration", 20)),
        next_node_id="BUNDLE"
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
        rework_policy=rework_policy
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

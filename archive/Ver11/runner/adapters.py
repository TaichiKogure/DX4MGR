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

def _parse_periods(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value if v is not None]
    if isinstance(value, (int, float, np.integer, np.floating)) and not np.isnan(value):
        return [float(value)]
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        for sep in ["|", ";", ",", " "]:
            cleaned = cleaned.replace(sep, " ")
        parts = [p for p in cleaned.split(" ") if p]
        periods = []
        for p in parts:
            try:
                periods.append(float(p))
            except ValueError:
                continue
        return periods
    return []


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
    
    # 摩擦設定
    friction_model = params.get("friction_model", "linear")
    friction_alpha = float(params.get("friction_alpha", 0.05))

    # ノード構築
    # 1. 小実験
    small_exp_gate = WorkGate(
        "SMALL_EXP", engine, 
        n_servers=999, # 無制限
        duration_dist=partial(_exp_dist, rng, params.get("small_exp_duration", 5)),
        next_node_id="PROTO",
        task_type=TaskType.SMALL_EXP,
        friction_model=friction_model,
        friction_alpha=friction_alpha
    )
    
    # 2. 試作
    proto_gate = WorkGate(
        "PROTO", engine,
        n_servers=int(params.get("n_servers_proto", 5)), # パラメータ化
        duration_dist=partial(_exp_dist, rng, params.get("proto_duration", 20)),
        next_node_id="BUNDLE",
        task_type=TaskType.PROTO_TEST,
        friction_model=friction_model,
        friction_alpha=friction_alpha
    )
    
    # 3. バンドル
    # 3. DR periods (support multi-gate via CSV: "30|65|90")
    raw_periods = params.get("dr_periods", None)
    periods = _parse_periods(raw_periods)
    if not periods:
        periods = _parse_periods(params.get("dr_period", 90))
    if not periods:
        periods = [90.0]

    if len(periods) == 1:
        dr_gate_ids = ["DR_GATE"]
    else:
        dr_gate_ids = [f"DR_GATE_{i}" for i in range(1, len(periods) + 1)]

    # 4. BUNDLE -> first DR gate
    bundle_gate = BundleGate(
        "BUNDLE", engine,
        bundle_size_dist=partial(_constant_dist, params.get("bundle_size", 3)),
        next_node_id=dr_gate_ids[0]
    )

    # 5. DR gates chain
    dr_gates = []
    for i, period in enumerate(periods):
        node_id = dr_gate_ids[i]
        next_node = dr_gate_ids[i + 1] if i + 1 < len(dr_gate_ids) else None
        rework_node = "SMALL_EXP" if i == 0 else dr_gate_ids[i - 1]
        dr_gates.append(MeetingGate(
            node_id, engine,
            period_days=period,
            approvers=approvers,
            next_node_id=next_node, # final gate ends
            # Multi-stage DR: rework returns to previous DR gate (not always to SMALL_EXP).
            rework_node_id=rework_node,
            rework_policy=rework_policy,
            conditional_prob_ratio=params.get("conditional_prob_ratio", 0.8),
            decision_latency_days=params.get("decision_latency_days", 0.0)
        ))

    
    engine.add_node(small_exp_gate)
    engine.add_node(proto_gate)
    engine.add_node(bundle_gate)
    for gate in dr_gates:
        engine.add_node(gate)
    
    # 流入イベントのスケジュール
    t = 0
    while t < days:
        t += rng.exponential(1.0 / arrival_rate)
        if t < days:
            job = Job(job_id=f"job_{t:.2f}", created_at=t)
            engine.schedule_event(t, "ARRIVAL", {"job": job, "target_node": "SMALL_EXP"})
            
    return engine

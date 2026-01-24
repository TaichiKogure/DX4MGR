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

    days = params.get("days", 365)
    arrival_rate = params.get("arrival_rate", 0.5)

    from functools import partial

    rework_beta_a = params.get("rework_beta_a", 2.0)
    rework_beta_b = params.get("rework_beta_b", 5.0)

    rework_load_factor = float(params.get("rework_load_factor", 0.5))
    rework_policy = ReworkPolicy(
        rework_load_factor=rework_load_factor,
        weight_dist_func=partial(_beta_dist, rng, rework_beta_a, rework_beta_b),
        max_rework_cycles=params.get("max_rework_cycles", 5),
        decay=params.get("decay", 0.7),
        task_type_mix=params.get("rework_task_type_mix", 1.0)
    )
    dr2_rework_multiplier = float(params.get("dr2_rework_multiplier", 1.0))
    if dr2_rework_multiplier != 1.0:
        rework_policy_dr2 = ReworkPolicy(
            rework_load_factor=rework_load_factor * dr2_rework_multiplier,
            weight_dist_func=partial(_beta_dist, rng, rework_beta_a, rework_beta_b),
            max_rework_cycles=params.get("max_rework_cycles", 5),
            decay=params.get("decay", 0.7),
            task_type_mix=params.get("rework_task_type_mix", 1.0)
        )
    else:
        rework_policy_dr2 = rework_policy

    from core.entities import APPROVER_TYPES
    approvers = []
    n_senior = int(params.get("n_senior", 1))
    n_coordinator = int(params.get("n_coordinator", 0))
    n_new = int(params.get("n_new", 0))

    for i in range(n_senior):
        approvers.append(Approver(f"senior_{i}", "Senior", APPROVER_TYPES["Senior"]["capacity"], APPROVER_TYPES["Senior"]["quality"]))
    for i in range(n_coordinator):
        approvers.append(Approver(f"coord_{i}", "Coordinator", APPROVER_TYPES["Coordinator"]["capacity"], APPROVER_TYPES["Coordinator"]["quality"]))
    for i in range(n_new):
        approvers.append(Approver(f"new_{i}", "New", APPROVER_TYPES["New"]["capacity"], APPROVER_TYPES["New"]["quality"]))

    if not approvers:
        approvers.append(Approver("default", "Senior", 10, 0.9))

    friction_model = params.get("friction_model", "linear")
    friction_alpha = float(params.get("friction_alpha", 0.05))
    def _optional_int(value):
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    dr1_capacity = _optional_int(params.get("dr1_capacity", params.get("dr_capacity")))
    dr2_capacity = _optional_int(params.get("dr2_capacity", dr1_capacity))
    dr3_capacity = _optional_int(params.get("dr3_capacity", dr2_capacity))

    dr1_cost = float(params.get("dr1_cost_per_review", 1.0))
    dr2_cost = float(params.get("dr2_cost_per_review", dr1_cost * 2.0))
    dr3_cost = float(params.get("dr3_cost_per_review", dr1_cost * 3.0))

    # Flow: SMALL_EXP -> BUNDLE_SMALL -> DR1 -> MID_EXP -> BUNDLE_MID -> DR2 -> FIN_EXP -> BUNDLE_FIN -> DR3 -> COMPLETE
    small_exp_gate = WorkGate(
        "SMALL_EXP", engine,
        n_servers=999,
        duration_dist=partial(_exp_dist, rng, params.get("small_exp_duration", 5)),
        next_node_id="BUNDLE_SMALL",
        task_type=TaskType.SMALL_EXP,
        friction_model=friction_model,
        friction_alpha=friction_alpha
    )

    bundle_small_gate = BundleGate(
        "BUNDLE_SMALL", engine,
        bundle_size_dist=partial(_constant_dist, params.get("bundle_size_small", 5)),
        next_node_id="DR1"
    )

    dr1_gate = MeetingGate(
        "DR1", engine,
        period_days=params.get("dr1_period", 30),
        approvers=approvers,
        next_node_id="MID_EXP",
        rework_node_id="SMALL_EXP",
        rework_policy=rework_policy,
        nogo_node_id="SMALL_EXP",
        conditional_prob_ratio=params.get("conditional_prob_ratio", 0.8),
        decision_latency_days=params.get("decision_latency_days", 0.0),
        capacity_override=dr1_capacity,
        cost_per_review=dr1_cost
    )

    mid_exp_gate = WorkGate(
        "MID_EXP", engine,
        n_servers=int(params.get("n_servers_mid", 5)),
        duration_dist=partial(_exp_dist, rng, params.get("mid_exp_duration", 20)),
        next_node_id="BUNDLE_MID",
        task_type=TaskType.MID_EXP,
        friction_model=friction_model,
        friction_alpha=friction_alpha
    )

    bundle_mid_gate = BundleGate(
        "BUNDLE_MID", engine,
        bundle_size_dist=partial(_constant_dist, params.get("bundle_size_mid", 3)),
        next_node_id="DR2"
    )

    dr2_gate = MeetingGate(
        "DR2", engine,
        period_days=params.get("dr2_period", 60),
        approvers=approvers,
        next_node_id="FIN_EXP",
        rework_node_id="MID_EXP",
        rework_policy=rework_policy_dr2,
        nogo_node_id="MID_EXP",
        conditional_prob_ratio=params.get("conditional_prob_ratio", 0.8),
        decision_latency_days=params.get("decision_latency_days", 0.0),
        capacity_override=dr2_capacity,
        cost_per_review=dr2_cost
    )

    fin_exp_gate = WorkGate(
        "FIN_EXP", engine,
        n_servers=int(params.get("n_servers_fin", 3)),
        duration_dist=partial(_exp_dist, rng, params.get("fin_exp_duration", 30)),
        next_node_id="BUNDLE_FIN",
        task_type=TaskType.FIN_EXP,
        friction_model=friction_model,
        friction_alpha=friction_alpha
    )

    bundle_fin_gate = BundleGate(
        "BUNDLE_FIN", engine,
        bundle_size_dist=partial(_constant_dist, params.get("bundle_size_fin", 3)),
        next_node_id="DR3"
    )

    dr3_gate = MeetingGate(
        "DR3", engine,
        period_days=params.get("dr3_period", 90),
        approvers=approvers,
        next_node_id=None,
        rework_node_id="FIN_EXP",
        rework_policy=rework_policy,
        nogo_node_id="FIN_EXP",
        conditional_prob_ratio=params.get("conditional_prob_ratio", 0.8),
        decision_latency_days=params.get("decision_latency_days", 0.0),
        capacity_override=dr3_capacity,
        cost_per_review=dr3_cost
    )

    engine.add_node(small_exp_gate)
    engine.add_node(bundle_small_gate)
    engine.add_node(dr1_gate)
    engine.add_node(mid_exp_gate)
    engine.add_node(bundle_mid_gate)
    engine.add_node(dr2_gate)
    engine.add_node(fin_exp_gate)
    engine.add_node(bundle_fin_gate)
    engine.add_node(dr3_gate)

    t = 0
    while t < days:
        t += rng.exponential(1.0 / arrival_rate)
        if t < days:
            job = Job(job_id=f"job_{t:.2f}", created_at=t)
            engine.schedule_event(t, "ARRIVAL", {"job": job, "target_node": "SMALL_EXP"})

    return engine

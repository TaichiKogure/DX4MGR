import numpy as np
import pandas as pd
import os
from .technology import TechnologyItem
from .team import Team
from .experiment import Experiment, Resource
from .engine import SimulationEngine
from .entities import Job, TaskType, Department, Handoff, CrossDeptMeeting, WorkItem, DecisionLogic
from .gates import WorkGate, BundleGate, MeetingGate, AdoptionGate
from .policies import ReworkPolicy
from .planning import Scheduler, LatentRisk

class Simulation:
    def __init__(self, config=None):
        self.current_time = 0
        self.tech_items = []
        self.teams = {}
        self.resources = {}
        self.engine = None
        self.history = []
        self.tech_history = []  # Added for visualization
        self.gui_callback = None
        
        self.dr_threshold = 5.0
        self.dr_passed = False
        self.production_started = False
        self.strategy = 'strategic'
        
        self.kpis = {
            'total_experiments': 0,
            'technical_failures': 0,
            'operational_failures': 0,
            'total_gain': 0,
            'rework_count': 0,
            'integration_days': 0,
            'market_complaints': 0,
            'dr_cost': 0.0,
            'completed_jobs': 0,
            # Ver2追加KPI
            'sla_violations': 0,
            'handoff_events': 0,
            'dept_cost_time': 0.0
        }

    def setup_with_params(self, params):
        from .entities import Approver, TaskType, APPROVER_TYPES
        from functools import partial
        self.params = params
        self.dr_threshold = params.get('dr_threshold', 5.0)
        self.strategy = params.get('strategy', 'strategic')
        seed = params.get('seed', 42)
        rng = np.random.default_rng(seed)
        # Ver2: 複数部署関連の設定
        self.consider_departments = bool(params.get('consider_departments', False))
        self.consider_handoffs = bool(params.get('consider_handoffs', False))
        self.consider_cross_meetings = bool(params.get('consider_cross_meetings', False))
        self.consider_rework_rules = bool(params.get('consider_rework_rules', False))

        # Departments
        self.departments = []
        self.dept_map = {}
        for d in (params.get('departments') or []):
            try:
                dept = Department(
                    dept_id=d.get('id') or d.get('dept_id') or d.get('name'),
                    name=d.get('name') or d.get('dept_id') or d.get('id'),
                    calendar=d.get('calendar'),
                    cost_factor=float(d.get('cost_factor', 1.0) or 1.0),
                    sla=d.get('sla') or {"avg_response": 0.0, "max_wait": 0.0}
                )
                self.departments.append(dept)
                self.dept_map[dept.dept_id] = dept
            except Exception:
                continue

        # Handoffs
        self.handoffs = []
        self.handoff_map = {}
        for h in (params.get('handoffs') or []):
            try:
                ho = Handoff(
                    from_dept=h.get('from') or h.get('from_dept'),
                    to_dept=h.get('to') or h.get('to_dept'),
                    q_if=float(h.get('q_if', 1.0) or 1.0),
                    info_loss_lambda=float(h.get('lambda', h.get('info_loss_lambda', 0.0)) or 0.0),
                    transfer_time_dist=h.get('wait_dist') or h.get('transfer_time_dist')
                )
                self.handoffs.append(ho)
                self.handoff_map[(ho.from_dept, ho.to_dept)] = ho
            except Exception:
                continue

        # Cross-dept meetings / WorkItems (保持のみ: 最小実装)
        self.cross_meetings = []
        for cm in (params.get('cross_meetings') or []):
            try:
                logic = cm.get('logic', 'GO')
                logic_enum = DecisionLogic[logic] if isinstance(logic, str) else logic
                self.cross_meetings.append(CrossDeptMeeting(
                    departments=cm.get('departments') or cm.get('depts') or [],
                    interval_days=float(cm.get('interval', cm.get('interval_days', 14.0)) or 14.0),
                    threshold=float(cm.get('threshold', 0.0) or 0.0),
                    logic=logic_enum
                ))
            except Exception:
                continue
        self.work_items = []
        for wi in (params.get('work_items') or []):
            try:
                self.work_items.append(WorkItem(
                    work_id=str(wi.get('id') or wi.get('work_id') or f"W{len(self.work_items)}"),
                    steps=wi.get('steps') or [],
                    owners=wi.get('owners'),
                    rework_rules=wi.get('rework_rules')
                ))
            except Exception:
                continue

        # チームの部署マッピング
        self.team_department = {}
        
        sampling_interval = params.get('sampling_interval', 1.0)
        self.engine = SimulationEngine(rng=rng, sampling_interval=sampling_interval)
        self.engine.sampling_callback = self._record_snapshot
        
        # 1. 技術要素の作成 (Plan5x)
        # 設定ファイルから技術要素を定義可能にする
        tech_configs = params.get('tech_items', [
            {"name": "Material Mix", "tacitness": 0.7},
            {"name": "Process Conditions", "tacitness": 0.4},
            {"name": "Evaluation Criteria", "tacitness": 0.3}
        ])
        self.tech_items = [TechnologyItem(tc['name'], tacitness=tc['tacitness']) for tc in tech_configs]
        
        # トレードオフ設定 (オプション)
        for tc in tech_configs:
            if 'trade_offs' in tc:
                for target_name, intensity in tc['trade_offs'].items():
                    item = next((t for t in self.tech_items if t.name == tc['name']), None)
                    if item:
                        item.add_trade_off(target_name, intensity=intensity)
        
        if not tech_configs: # デフォルト
            self.tech_items[0].add_trade_off("Process Conditions", intensity=0.2)
            self.tech_items[1].add_trade_off("Evaluation Criteria", intensity=0.1)

        # 2. チームの作成 (Plan5x)
        team_configs = params.get('teams', [
            {"name": "Research", "wip_limit": 2},
            {"name": "Prototype", "wip_limit": 1},
            {"name": "Analysis", "wip_limit": 2},
            {"name": "MassProduction", "wip_limit": 5}
        ])
        self.teams = {}
        for tc in team_configs:
            t = Team(tc['name'], wip_limit=tc.get('wip_limit', 3),
                     department=tc.get('dept'), utilization_cap=tc.get('utilization_cap'),
                     num_agents=int(tc.get('num_agents', 3) or 3))
            self.teams[tc['name']] = t
        for tc in team_configs:
            if 'dept' in tc:
                self.team_department[tc['name']] = tc['dept']
        
        if params.get('use_digital_twin'):
            self.calibrate_from_logs(list(self.teams.values()), params.get('past_logs_file'))

        # チーム間のエッジ設定 (オプション)
        edge_configs = params.get('team_edges', [
            {"from": "Research", "to": "Prototype", "delay": params.get('delay_res_proto', 2), "loss": params.get('loss_res_proto', 0.1), "dist": params.get('dist_res_proto', 0.1)},
            {"from": "Prototype", "to": "Analysis", "delay": params.get('delay_proto_ana', 1), "loss": params.get('loss_proto_ana', 0.05), "dist": params.get('dist_proto_ana', 0.05)},
            {"from": "Analysis", "to": "Research", "delay": params.get('delay_ana_res', 2), "loss": params.get('loss_ana_res', 0.1), "dist": params.get('dist_ana_res', 0.2)},
            {"from": "Prototype", "to": "MassProduction", "delay": 5, "loss": 0.2, "dist": 0.2}
        ])
        for ec in edge_configs:
            if ec['from'] in self.teams and ec['to'] in self.teams:
                self.teams[ec['from']].add_edge(ec['to'], delay=ec['delay'], loss_prob=ec['loss'], distortion_prob=ec['dist'])
        
        # 3. Ver14 Flowの構築 (DES)
        def _get_dist(dist_config, r):
            if isinstance(dist_config, (int, float)):
                return lambda: float(dist_config)
            
            dist_type = dist_config.get('type', 'exponential')
            params = dist_config.get('params', {'scale': 5})
            
            if dist_type == 'exponential':
                return lambda: r.exponential(params.get('scale', 5))
            elif dist_type == 'uniform':
                return lambda: r.uniform(params.get('low', 0), params.get('high', 10))
            elif dist_type == 'triangular':
                return lambda: r.triangular(params.get('left', 0), params.get('mode', 5), params.get('right', 10))
            elif dist_type == 'constant':
                return lambda: float(params.get('value', 5))
            return lambda: r.exponential(5)

        rework_policy_config = params.get('rework_policy', {
            "rework_load_factor": params.get("rework_load_factor", 0.5),
            "max_rework_cycles": params.get("max_rework_cycles", 5),
            "decay": params.get("decay", 0.7)
        })

        rework_policy = ReworkPolicy(
            rework_load_factor=float(rework_policy_config.get("rework_load_factor", 0.5)),
            weight_dist_func=partial(lambda r: r.beta(2.0, 5.0), rng),
            max_rework_cycles=int(rework_policy_config.get("max_rework_cycles", 5)),
            decay=float(rework_policy_config.get("decay", 0.7))
        )

        # 統合ポイント: WorkGateのカスタム処理
        class Plan7WorkGate(WorkGate):
            def __init__(self, sim_instance, phase, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sim = sim_instance
                self.phase = phase
            
            def process(self, now):
                if self.busy_servers >= self.n_servers or not self.queue:
                    return
                
                job = self.queue.pop(0)
                self.busy_servers += 1
                self.busy_jobs.append(job)
                
                # フェーズに応じた技術要素の選択
                if job.target_tech is None:
                    if self.phase == 'research':
                        # 基礎研究：不確実性が高いものを優先
                        job.target_tech = max(self.sim.tech_items, key=lambda x: x.uncertainty)
                    elif self.phase == 'prototype':
                        # 試作：成熟度が低く、かつ一定程度不確実性が下がっているものを優先
                        candidates = [t for t in self.sim.tech_items if t.uncertainty < 0.6]
                        if candidates:
                            job.target_tech = min(candidates, key=lambda x: x.maturity)
                        else:
                            job.target_tech = self.sim.engine.rng.choice(self.sim.tech_items)
                    else:
                        # 量産準備：全体的に底上げ
                        job.target_tech = min(self.sim.tech_items, key=lambda x: x.maturity)
                
                # チームの決定
                team_mapping = {
                    'research': 'Research',
                    'prototype': 'Prototype',
                    'mass_production': 'MassProduction'
                }
                executor_team = self.sim.teams.get(team_mapping.get(self.phase, "Research"))
                
                # 通信劣化の模擬 (Plan 1 + Plan 5)
                protocol_adherence = 0.8 + 0.2 * executor_team.skill
                
                # 前工程からの引き継ぎ劣化 + 部門ハンドオフ
                handoff_delay = 0.0
                if hasattr(job, 'last_team_name') and job.last_team_name != executor_team.name:
                    source_team = self.sim.teams.get(job.last_team_name)
                    if source_team:
                        msg = {'job_id': job.job_id, 'tech_name': job.target_tech.name}
                        source_team.send_message(msg, executor_team, now)
                        
                        # 暗黙知度が高いほど劣化の影響が深刻化
                        tacitness = job.target_tech.tacitness
                        if msg.get('lost'):
                            protocol_adherence *= (0.6 - 0.3 * tacitness)
                        elif msg.get('distorted'):
                            protocol_adherence *= (0.9 - 0.4 * tacitness)

                        # 部門遷移時の追加効果（オプション）
                        if self.sim.consider_handoffs or self.sim.consider_departments:
                            from_dept = self.sim.team_department.get(getattr(source_team, 'name', ''), getattr(source_team, 'department', None))
                            to_dept = self.sim.team_department.get(getattr(executor_team, 'name', ''), getattr(executor_team, 'department', None))
                            if from_dept and to_dept and from_dept != to_dept:
                                ho = self.sim.handoff_map.get((from_dept, to_dept))
                                if ho:
                                    sampler = ho.sampler(self.sim.engine.rng)
                                    handoff_delay = float(sampler())
                                    # 品質/情報損失の劣化
                                    protocol_adherence *= max(0.1, float(ho.q_if or 1.0))
                                    protocol_adherence *= max(0.5, 1.0 - float(ho.info_loss_lambda or 0.0) * 0.5)
                                    self.sim.kpis['handoff_events'] += 1
                                    job.add_history(self.node_id, 'HANDOFF', now, from_dept=from_dept, to_dept=to_dept, delay=handoff_delay,
                                                    q_if=float(ho.q_if or 1.0), lambda_=float(ho.info_loss_lambda or 0.0))
                
                job.last_team_name = executor_team.name
                
                exp = Experiment([job.target_tech], executor_team, [], all_tech_items=self.sim.tech_items)
                result = exp.run(protocol_adherence=protocol_adherence)
                
                # 学習 (Plan 1)
                for agent in executor_team.agents:
                    agent.learn()
                
                self.sim.kpis['total_experiments'] += 1
                self.sim.kpis['total_gain'] += result['effective_gain']
                if not result['success']:
                    if result['failure_type'] == 0: self.sim.kpis['technical_failures'] += 1
                    else: self.sim.kpis['operational_failures'] += 1
                
                # 待機時間（enqueueからの差）
                wait_time = now - getattr(job, 'temp_enqueue_time', now)
                # SLA違反チェック（部門SLA: max_wait）
                if self.sim.consider_departments:
                    dept_id = self.sim.team_department.get(executor_team.name, getattr(executor_team, 'department', None))
                    if dept_id and dept_id in self.sim.dept_map:
                        max_wait = float((self.sim.dept_map[dept_id].sla or {}).get('max_wait', 0.0) or 0.0)
                        if max_wait > 0 and wait_time > max_wait:
                            self.sim.kpis['sla_violations'] += 1
                            job.add_history(self.node_id, 'SLA_VIOLATION', now, dept=dept_id, wait_time=wait_time, max_wait=max_wait)

                duration = self.duration_dist() + float(handoff_delay or 0.0)
                finish_time = now + duration
                # 部門コスト（簡易）：処理時間×コスト係数
                if self.sim.consider_departments:
                    dept_id2 = self.sim.team_department.get(executor_team.name, getattr(executor_team, 'department', None))
                    if dept_id2 and dept_id2 in self.sim.dept_map:
                        cf = float(self.sim.dept_map[dept_id2].cost_factor or 1.0)
                        self.sim.kpis['dept_cost_time'] += (duration * cf)
                self.engine.schedule_event(finish_time, "WORK_COMPLETE", {"node_id": self.node_id, "job": job})

        class Plan7MeetingGate(MeetingGate):
            def __init__(self, sim_instance, thresholds, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sim = sim_instance
                self.thresholds = thresholds
            
            def process(self, now):
                # 基本的な延期ロジック
                if not self.queue and self.postpone_options:
                    postpone_days = self.engine.rng.choice(self.postpone_options, p=self.postpone_probs)
                    self.next_meeting_time = now + postpone_days
                    self.postponed_count += 1
                    self.engine.schedule_event(self.next_meeting_time, "MEETING_START", {"node_id": self.node_id})
                    return

                count = 0
                while self.queue and count < self.capacity:
                    job = self.queue.pop(0)
                    count += 1
                    wait_time = now - job.temp_enqueue_time
                    self.total_wait_time += wait_time
                    self.processed_count += 1
                    self.total_cost += self.cost_per_review
                    
                    # 技術状態の評価
                    avg_uncert = np.mean([t.uncertainty for t in self.sim.tech_items])
                    avg_matur = np.mean([t.maturity for t in self.sim.tech_items])
                    
                    # 閾値チェック
                    tech_ok = True
                    if avg_uncert > self.thresholds.get('uncertainty', 1.0): tech_ok = False
                    if avg_matur < self.thresholds.get('maturity', 0.0): tech_ok = False
                    
                    # 判定 (技術ベースの補正)
                    q = self.quality
                    if not tech_ok:
                        q *= 0.5 # 技術が未熟だとパスする確率が激減
                    
                    rand = self.engine.rng.random()
                    decision_time = now + self.decision_latency_days
                    
                    if rand < q: # GO
                        outcome = "GO"
                        target = self.next_node_id
                    elif rand < q + (1.0 - q) * self.conditional_prob_ratio: # CONDITIONAL
                        outcome = "CONDITIONAL"
                        target = self.rework_node_id
                        job.rework_count += 1
                        # 簡易的な再投入ロジック（本来はrework_policyを使うが、ここでは直接ARRIVAL）
                    else: # NO_GO
                        outcome = "NO_GO"
                        target = self.nogo_node_id or self.rework_node_id # 指定がなければreworkへ
                    
                    job.add_history(self.node_id, "DECISION", decision_time, outcome=outcome, 
                                    avg_uncert=avg_uncert, avg_matur=avg_matur)
                    
                    if target:
                        self.engine.schedule_event(decision_time, "ARRIVAL", {"job": job, "target_node": target}, priority=5)
                    else:
                        # 次のノードがない＝完了
                        self.engine.results["completed_jobs"].append(job)

                # 次の会議予約
                self.next_meeting_time = now + self.period_days
                self.engine.schedule_event(self.next_meeting_time, "MEETING_START", {"node_id": self.node_id})

        # ゲートのセットアップ (多段フロー)
        def _create_approvers(app_configs):
            apps = []
            for ac in app_configs:
                if isinstance(ac, str) and ac in APPROVER_TYPES:
                    at = APPROVER_TYPES[ac]
                    apps.append(Approver(f"{ac}_{len(apps)}", ac, at['capacity'], at['quality']))
                elif isinstance(ac, dict):
                    # dictの場合は、typeキーがあればAPPROVER_TYPESからデフォルト値を引き、それ以外は個別指定とする
                    at_type = ac.get('type', 'Senior')
                    base_at = APPROVER_TYPES.get(at_type, {"capacity": 5, "quality": 0.8})
                    
                    apps.append(Approver(ac.get('id', f"{at_type}_{len(apps)}"), 
                                         at_type, 
                                         ac.get('capacity', base_at['capacity']), 
                                         ac.get('quality', base_at['quality'])))
            return apps

        # デフォルトのフロー定義
        default_flow = [
            {
                "id": "RESEARCH_EXP", "type": "work", "phase": "research", 
                "n_servers": params.get('res_n_servers', 5),
                "duration_dist": params.get('res_duration_dist', {"type": "exponential", "params": {"scale": 5}}),
                "next_node_id": "DR1", "task_type": TaskType.SMALL_EXP
            },
            {
                "id": "DR1", "type": "meeting", "period_days": params.get('dr1_period', 14),
                "thresholds": {"uncertainty": 0.4},
                "approvers": params.get('dr1_approvers', ["Senior"]),
                "next_node_id": "PROTO_EXP", "rework_node_id": "RESEARCH_EXP",
                "cost_per_review": params.get('cost_per_review', 100.0)
            },
            {
                "id": "PROTO_EXP", "type": "work", "phase": "prototype",
                "n_servers": params.get('proto_n_servers', 3),
                "duration_dist": params.get('proto_duration_dist', {"type": "exponential", "params": {"scale": 10}}),
                "next_node_id": "DR2", "task_type": TaskType.PROTOTYPE
            },
            {
                "id": "DR2", "type": "meeting", "period_days": params.get('dr2_period', params.get('dr1_period', 14) * 2),
                "thresholds": {"uncertainty": 0.2, "maturity": 0.5},
                "approvers": params.get('dr2_approvers', [{"type": "Director", "capacity": 3, "quality": 0.9}]),
                "next_node_id": "MASS_PROD_EXP", "rework_node_id": "PROTO_EXP",
                "cost_per_review": params.get('cost_per_review', 100.0) * 2
            },
            {
                "id": "MASS_PROD_EXP", "type": "work", "phase": "mass_production",
                "n_servers": params.get('mass_n_servers', 2),
                "duration_dist": params.get('mass_duration_dist', {"type": "exponential", "params": {"scale": 20}}),
                "next_node_id": "DR3", "task_type": TaskType.MASS_PROD
            },
            {
                "id": "DR3", "type": "meeting", "period_days": params.get('dr3_period', params.get('dr1_period', 14) * 4),
                "thresholds": {"maturity": 0.8},
                "approvers": params.get('dr3_approvers', ["Senior"]),
                "next_node_id": None, "rework_node_id": "MASS_PROD_EXP",
                "cost_per_review": params.get('cost_per_review', 100.0) * 5
            }
        ]
        
        flow_configs = params.get('flow', default_flow)
        
        for nc in flow_configs:
            # task_typeの変換 (文字列からTaskType Enumへ)
            tt = nc.get('task_type')
            if isinstance(tt, str):
                try:
                    tt = TaskType[tt]
                except KeyError:
                    tt = None
            
            if nc['type'] == 'work':
                node = Plan7WorkGate(self, nc['phase'], nc['id'], self.engine,
                                     n_servers=nc['n_servers'],
                                     duration_dist=_get_dist(nc['duration_dist'], rng),
                                     next_node_id=nc['next_node_id'],
                                     task_type=tt)
            elif nc['type'] == 'meeting':
                node = Plan7MeetingGate(self, nc['thresholds'], nc['id'], self.engine,
                                        period_days=nc['period_days'],
                                        approvers=_create_approvers(nc['approvers']),
                                        next_node_id=nc['next_node_id'],
                                        rework_node_id=nc['rework_node_id'],
                                        rework_policy=rework_policy,
                                        cost_per_review=nc.get('cost_per_review', 0.0))
            self.engine.add_node(node)
        
        # 初回ジョブの投入
        job_arrival_dist = _get_dist(params.get('job_arrival_dist', {"type": "uniform", "params": {"low": 0, "high": 5}}), rng)
        num_initial_jobs = params.get('num_initial_jobs', 3)
        start_node = params.get('start_node_id', flow_configs[0]['id'] if flow_configs else "RESEARCH_EXP")
        
        for i in range(num_initial_jobs):
            job = Job(f"project_{i}", created_at=0.0)
            job.latent = LatentRisk()
            self.engine.schedule_event(job_arrival_dist(), "ARRIVAL", {"job": job, "target_node": start_node})

    def calibrate_from_logs(self, teams, log_path=None):
        if not log_path or not os.path.exists(log_path):
            log_path = 'configs/past_logs.csv'
        if os.path.exists(log_path):
            df_log = pd.read_csv(log_path)
            for team in teams:
                team_data = df_log[df_log['team_name'] == team.name]
                if not team_data.empty:
                    team.skill = 1.0 - team_data['error_rate'].iloc[0]
                    for agent in team.agents:
                        agent.skill = max(0.1, team.skill * np.random.uniform(0.8, 1.2))

    def run(self, steps=100):
        # DESエンジンを回す
        self.engine.run(steps)
        
        # 結果の回収
        self.kpis['completed_jobs'] = len(self.engine.results["completed_jobs"])
        # MeetingGateのコストなどを集計
        for node in self.engine.nodes.values():
            if isinstance(node, MeetingGate):
                self.kpis['dr_cost'] += node.total_cost
        
        return self.kpis

    def get_tech_status(self):
        return {t.name: {'evidence': t.evidence, 'uncertainty': t.uncertainty, 'maturity': t.maturity} for t in self.tech_items}

    def _record_snapshot(self, at_time):
        current_dr_cost = 0.0
        for node in self.engine.nodes.values():
            if hasattr(node, "total_cost"):
                current_dr_cost += node.total_cost
        
        snapshot = {
            "time": at_time,
            "tech_items": {t.name: {"maturity": t.maturity, "uncertainty": t.uncertainty} for t in self.tech_items},
            "cumulative_cost": current_dr_cost
        }
        self.tech_history.append(snapshot)
        if self.gui_callback:
            self.gui_callback(at_time)

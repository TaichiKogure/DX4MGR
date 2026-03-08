import numpy as np
import pandas as pd
import os
from .technology import TechnologyItem
from .team import Team
from .experiment import Experiment, Resource
from .engine import SimulationEngine
from .entities import Job, TaskType
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
            'completed_jobs': 0
        }

    def setup_with_params(self, params):
        from .entities import Approver, TaskType
        self.params = params
        self.dr_threshold = params.get('dr_threshold', 5.0)
        self.strategy = params.get('strategy', 'strategic')
        seed = params.get('seed', 42)
        rng = np.random.default_rng(seed)
        
        self.engine = SimulationEngine(rng=rng)
        
        # 1. 技術要素の作成 (Plan5x)
        self.tech_items = [
            TechnologyItem("Material Mix", tacitness=0.7),
            TechnologyItem("Process Conditions", tacitness=0.4),
            TechnologyItem("Evaluation Criteria", tacitness=0.3)
        ]
        self.tech_items[0].add_trade_off("Process Conditions", intensity=0.2)
        self.tech_items[1].add_trade_off("Evaluation Criteria", intensity=0.1)

        # 2. チームの作成 (Plan5x)
        research = Team("Research", wip_limit=2)
        prototype = Team("Prototype", wip_limit=1)
        analysis = Team("Analysis", wip_limit=2)
        mass_production = Team("MassProduction", wip_limit=5)
        
        if params.get('use_digital_twin'):
            self.calibrate_from_logs([research, prototype, analysis, mass_production], params.get('past_logs_file'))

        research.add_edge("Prototype", 
                          delay=params.get('delay_res_proto', 2), 
                          loss_prob=params.get('loss_res_proto', 0.1), 
                          distortion_prob=params.get('dist_res_proto', 0.1))
        prototype.add_edge("Analysis", 
                           delay=params.get('delay_proto_ana', 1), 
                           loss_prob=params.get('loss_proto_ana', 0.05), 
                           distortion_prob=params.get('dist_proto_ana', 0.05))
        analysis.add_edge("Research", 
                          delay=params.get('delay_ana_res', 2), 
                          loss_prob=params.get('loss_ana_res', 0.1), 
                          distortion_prob=params.get('dist_ana_res', 0.2))
        prototype.add_edge("MassProduction", delay=5, loss_prob=0.2, distortion_prob=0.2)
        
        self.teams = {t.name: t for t in [research, prototype, analysis, mass_production]}

        # 3. Ver14 Flowの構築 (DES)
        # 簡易化のため、engineにノードを追加していく
        from functools import partial
        def _exp_dist(r, scale): return r.exponential(scale)
        
        rework_policy = ReworkPolicy(
            rework_load_factor=float(params.get("rework_load_factor", 0.5)),
            weight_dist_func=partial(lambda r: r.beta(2.0, 5.0), rng),
            max_rework_cycles=int(params.get("max_rework_cycles", 5)),
            decay=float(params.get("decay", 0.7))
        )

        # 統合ポイント: WorkGateのカスタム処理
        class Plan6WorkGate(WorkGate):
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
                
                # 前工程からの引き継ぎ劣化
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
                
                duration = self.duration_dist()
                finish_time = now + duration
                self.engine.schedule_event(finish_time, "WORK_COMPLETE", {"node_id": self.node_id, "job": job})

        class Plan6MeetingGate(MeetingGate):
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
        dr1_period = params.get('dr1_period', 14)
        dr_cost = params.get('cost_per_review', 100.0)
        
        # 1. Research Phase
        res_exp = Plan6WorkGate(self, "research", "RESEARCH_EXP", self.engine, n_servers=5, 
                                duration_dist=lambda: rng.exponential(5), 
                                next_node_id="DR1", task_type=TaskType.SMALL_EXP)
        dr1 = Plan6MeetingGate(self, {'uncertainty': 0.4}, "DR1", self.engine, period_days=dr1_period, 
                               approvers=[Approver("research_mgr", "Manager", 5, 0.8)],
                               next_node_id="PROTO_EXP", rework_node_id="RESEARCH_EXP", rework_policy=rework_policy,
                               cost_per_review=dr_cost)
        
        # 2. Prototype Phase
        proto_exp = Plan6WorkGate(self, "prototype", "PROTO_EXP", self.engine, n_servers=3,
                                  duration_dist=lambda: rng.exponential(10),
                                  next_node_id="DR2", task_type=TaskType.PROTOTYPE)
        dr2 = Plan6MeetingGate(self, {'uncertainty': 0.2, 'maturity': 0.5}, "DR2", self.engine, period_days=dr1_period * 2,
                               approvers=[Approver("tech_dir", "Director", 3, 0.9)],
                               next_node_id="MASS_PROD_EXP", rework_node_id="PROTO_EXP", rework_policy=rework_policy,
                               cost_per_review=dr_cost * 2)
        
        # 3. Mass Production Phase
        mass_exp = Plan6WorkGate(self, "mass_production", "MASS_PROD_EXP", self.engine, n_servers=2,
                                 duration_dist=lambda: rng.exponential(20),
                                 next_node_id="DR3", task_type=TaskType.MASS_PROD)
        dr3 = Plan6MeetingGate(self, {'maturity': 0.8}, "DR3", self.engine, period_days=dr1_period * 4,
                               approvers=[Approver("factory_mgr", "Manager", 2, 0.85)],
                               next_node_id=None, rework_node_id="MASS_PROD_EXP", rework_policy=rework_policy,
                               cost_per_review=dr_cost * 5)
        
        for node in [res_exp, dr1, proto_exp, dr2, mass_exp, dr3]:
            self.engine.add_node(node)
        
        # 初回ジョブの投入
        for i in range(3):
            job = Job(f"project_{i}", created_at=0.0)
            job.latent = LatentRisk()
            self.engine.schedule_event(rng.uniform(0, 5), "ARRIVAL", {"job": job, "target_node": "RESEARCH_EXP"})

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

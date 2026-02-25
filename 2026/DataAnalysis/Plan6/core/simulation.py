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
            def __init__(self, sim_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sim = sim_instance
            
            def process(self, now):
                # 元の処理でサーバーを確保
                if self.busy_servers >= self.n_servers or not self.queue:
                    return
                
                job = self.queue.pop(0)
                self.busy_servers += 1
                self.busy_jobs.append(job)
                
                # Plan5x: 実験の実行
                # Jobに紐付いた技術要素、または戦略的に選択
                if job.target_tech is None:
                    if self.sim.strategy == 'strategic':
                        job.target_tech = max(self.sim.tech_items, key=lambda x: x.uncertainty)
                    else:
                        job.target_tech = self.sim.engine.rng.choice(self.sim.tech_items)
                
                # チームの決定 (ノード名から推測)
                team_name = "Prototype" if "EXP" in self.node_id else "Research"
                executor_team = self.sim.teams.get(team_name, self.sim.teams["Prototype"])
                
                # 通信劣化の模擬（簡易版：Jobの履歴から直前の通信状況を見る等が可能だが、ここではベースラインを使用）
                protocol_adherence = 0.8 + 0.2 * executor_team.skill
                
                exp = Experiment([job.target_tech], executor_team, [], all_tech_items=self.sim.tech_items)
                result = exp.run(protocol_adherence=protocol_adherence)
                
                self.sim.kpis['total_experiments'] += 1
                self.sim.kpis['total_gain'] += result['effective_gain']
                if not result['success']:
                    if result['failure_type'] == 0: self.sim.kpis['technical_failures'] += 1
                    else: self.sim.kpis['operational_failures'] += 1
                
                # Ver14: 処理時間の決定
                duration = self.duration_dist()
                finish_time = now + duration
                
                self.engine.schedule_event(finish_time, "WORK_COMPLETE", {"node_id": self.node_id, "job": job})

        # ゲートのセットアップ
        small_exp = Plan6WorkGate(self, "SMALL_EXP", self.engine, n_servers=5, 
                                  duration_dist=lambda: rng.exponential(5), 
                                  next_node_id="DR1", task_type=TaskType.SMALL_EXP)
        
        # DR1 (MeetingGate)
        from .entities import APPROVER_TYPES, Approver
        approvers = [Approver("boss", "Senior", 10, 0.9)]
        dr1 = MeetingGate("DR1", self.engine, period_days=30, approvers=approvers,
                          next_node_id=None, rework_node_id="SMALL_EXP", rework_policy=rework_policy)
        
        self.engine.add_node(small_exp)
        self.engine.add_node(dr1)
        
        # 初回ジョブの投入
        for i in range(5):
            job = Job(f"job_{i}", created_at=0.0)
            job.latent = LatentRisk()
            self.engine.schedule_event(rng.uniform(0, 10), "ARRIVAL", {"job": job, "target_node": "SMALL_EXP"})

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

import numpy as np
import pandas as pd
import os
from .technology import TechnologyItem
from .team import Team
from .experiment import Experiment, Resource

class Simulation:
    def __init__(self, config=None):
        self.current_time = 0
        self.tech_items = []
        self.teams = {}
        self.resources = {}
        self.history = []
        
        self.dr_threshold = 5.0 # DRに進むための累積証拠量しきい値
        self.dr_passed = False
        self.production_started = False
        self.strategy = 'strategic' # 'random' or 'strategic' (Plan 3)
        
        # 統計用KPI
        self.kpis = {
            'total_experiments': 0,
            'technical_failures': 0,
            'operational_failures': 0,
            'total_gain': 0,
            'rework_count': 0,
            'integration_days': 0,
            'market_complaints': 0
        }

    def setup_with_params(self, params):
        self.dr_threshold = params.get('dr_threshold', 5.0)
        self.strategy = params.get('strategy', 'strategic')
        
        # 技術要素の作成
        self.tech_items = [
            TechnologyItem("Material Mix", tacitness=0.7),
            TechnologyItem("Process Conditions", tacitness=0.4),
            TechnologyItem("Evaluation Criteria", tacitness=0.3)
        ]
        
        # トレードオフの設定 (Plan 2)
        self.tech_items[0].add_trade_off("Process Conditions", intensity=0.2)
        self.tech_items[1].add_trade_off("Evaluation Criteria", intensity=0.1)

        # リソースの作成 (Plan 2)
        self.resources = {
            'Mixer': Resource('Mixer', failure_rate=0.02),
            'Tester': Resource('Tester', failure_rate=0.01)
        }

        # チームの作成 (Plan 1, 4)
        research = Team("Research", wip_limit=2)
        prototype = Team("Prototype", wip_limit=1)
        analysis = Team("Analysis", wip_limit=2)
        mass_production = Team("MassProduction", wip_limit=5)
        
        # 実測ログからのキャリブレーション (Plan 5)
        if params.get('use_digital_twin'):
            self.calibrate_from_logs([research, prototype, analysis, mass_production], params.get('past_logs_file'))

        # 組織ネットワークの設定 (Plan 1, 4)
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

        # 量産チームへのエッジ (Plan 4)
        prototype.add_edge("MassProduction", delay=5, loss_prob=0.2, distortion_prob=0.2)
        
        self.teams = {t.name: t for t in [research, prototype, analysis, mass_production]}

    def calibrate_from_logs(self, teams, log_path=None):
        """
        実過去データ（ELN/Jira等）からパラメータを推定 (Plan 5)
        """
        if not log_path or not os.path.exists(log_path):
            log_path = 'configs/past_logs.csv'
            
        if os.path.exists(log_path):
            df_log = pd.read_csv(log_path)
            for team in teams:
                team_data = df_log[df_log['team_name'] == team.name]
                if not team_data.empty:
                    # ログに基づいたスキル値の上書き
                    team.skill = 1.0 - team_data['error_rate'].iloc[0]
                    # エージェントのスキルも連動させる
                    for agent in team.agents:
                        agent.skill = max(0.1, team.skill * np.random.uniform(0.8, 1.2))

    def step(self):
        self.current_time += 1
        
        # 0. リソース状態の更新と技術の減衰 (Plan 2)
        for res in self.resources.values():
            res.update_status(self.current_time)
        for t in self.tech_items:
            t.apply_decay(decay_rate=0.0005)

        # 1. 計画: Researchが技術要素を選んでPrototypeに送る (Plan 1, 3)
        total_requests = len(self.teams['Prototype'].inbox) + len(self.teams['Analysis'].inbox) + len(self.teams['Research'].inbox)
        if total_requests == 0 and not self.dr_passed:
            if self.strategy == 'strategic':
                # 最も不確実性が高い要素を優先 (Plan 3)
                target_tech = max(self.tech_items, key=lambda x: x.uncertainty)
            else:
                target_tech = np.random.choice(self.tech_items)
                
            msg = {
                'type': 'experiment_request',
                'tech_items': [target_tech],
                'params': {'target_temp': 100, 'pressure': 50},
                'metadata': {'reason': 'Initial screening'}
            }
            self.teams['Research'].send_message(msg, self.teams['Prototype'], self.current_time)

        # 2. 試作・実験: Prototypeが届いた依頼を処理 (Plan 1, 2, 5)
        proto_team = self.teams['Prototype']
        for msg in list(proto_team.inbox):
            if msg['arrival_time'] <= self.current_time and msg.get('type') == 'experiment_request':
                proto_team.inbox.remove(msg)
                
                # 担当エージェントの決定と学習 (Plan 1)
                agent = np.random.choice(proto_team.agents)
                agent.learn()
                
                # コミュニケーションロスの影響。チームスキルとエージェントスキルが影響 (Plan 5)
                avg_skill = (proto_team.skill + agent.skill) / 2.0
                protocol_adherence = 0.7 + 0.3 * avg_skill
                if msg.get('distorted'): protocol_adherence *= 0.8
                if msg.get('lost'): protocol_adherence *= 0.7
                
                # 実験実行
                exp = Experiment(msg['tech_items'], proto_team, list(self.resources.values()), all_tech_items=self.tech_items)
                result = exp.run(protocol_adherence=protocol_adherence)
                
                # KPI更新
                self.kpis['total_experiments'] += 1
                if result.get('failure_type') != 2: # 設備故障以外はGain加算
                    self.kpis['total_gain'] += result['effective_gain']
                
                if not result['success']:
                    if result['failure_type'] == 0:
                        self.kpis['technical_failures'] += 1
                    elif result['failure_type'] == 1:
                        self.kpis['operational_failures'] += 1
                
                # 結果をAnalysisに送る
                result_msg = {
                    'type': 'experiment_result',
                    'tech_items': msg['tech_items'],
                    'result': result,
                    'original_params': msg['params']
                }
                proto_team.send_message(result_msg, self.teams['Analysis'], self.current_time)

        # 3. 解析: Analysisが結果を解釈してResearchに戻す (Plan 1)
        analysis_team = self.teams['Analysis']
        for msg in list(analysis_team.inbox):
            if msg['arrival_time'] <= self.current_time and msg.get('type') == 'experiment_result':
                analysis_team.inbox.remove(msg)
                
                agent = np.random.choice(analysis_team.agents)
                agent.learn()
                
                # 解析における誤解 (Distortion)
                if msg.get('distorted'):
                    for t in msg['tech_items']:
                        t.uncertainty = min(1.0, t.uncertainty + 0.05)
                        self.kpis['rework_count'] += 1
                
                # 完了報告をResearchに
                back_msg = {'type': 'analysis_complete', 'tech_items': msg['tech_items']}
                analysis_team.send_message(back_msg, self.teams['Research'], self.current_time)

        # 3.5 Researchが完了報告を受け取る
        research_team = self.teams['Research']
        for msg in list(research_team.inbox):
            if msg['arrival_time'] <= self.current_time and msg.get('type') == 'analysis_complete':
                research_team.inbox.remove(msg)
                agent = np.random.choice(research_team.agents)
                agent.learn()

        # 3.6 知識流出 (Plan 1)
        for team in self.teams.values():
            team.handle_turnover(prob=0.001)

        # 4. 統合フェーズとDR判定 (Plan 1, 3, 4)
        total_evidence = sum(t.evidence for t in self.tech_items)
        if total_evidence >= self.dr_threshold and not self.dr_passed:
            avg_doc = np.mean([t.doc_level for t in self.tech_items])
            avg_data = np.mean([t.data_quality for t in self.tech_items])
            
            # 統合の難易度: 証拠不足 (Plan 3) と手戻りの蓄積 (Plan 1)
            evidence_shortage = max(0, 10.0 - total_evidence)
            integration_difficulty = 1.0 + self.kpis['rework_count'] * 0.1 + evidence_shortage * 0.2
            integration_time = 10 * integration_difficulty / (avg_doc * 0.5 + avg_data * 0.5 + 0.1)
            
            self.kpis['integration_days'] = integration_time
            self.dr_passed = True
            
            # 量産移行（TRL Gapの表現） (Plan 4)
            for t in self.tech_items:
                t.uncertainty = min(1.0, t.uncertainty + 0.2)
                t.evidence *= 0.7 
            
            self.production_started = True

        # 5. 市場フィードバック (Plan 4)
        if self.production_started and np.random.random() < 0.05:
            target = np.random.choice(self.tech_items)
            if target.uncertainty > 0.2:
                self.kpis['market_complaints'] += 1
                target.uncertainty = min(1.0, target.uncertainty + 0.3)

    def run(self, steps=100):
        for _ in range(steps):
            self.step()
        return self.kpis

    def get_tech_status(self):
        return {t.name: {'evidence': t.evidence, 'uncertainty': t.uncertainty, 'maturity': t.maturity} for t in self.tech_items}

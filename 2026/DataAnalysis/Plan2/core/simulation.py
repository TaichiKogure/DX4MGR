import numpy as np
from .technology import TechnologyItem
from .team import Team
from .experiment import Experiment

class Simulation:
    def __init__(self, config=None):
        self.current_time = 0
        self.tech_items = []
        self.teams = {}
        self.resources = {}
        self.history = []
        
        self.dr_threshold = 5.0 # DRに進むための累積証拠量しきい値
        self.dr_passed = False
        
        # 統計用KPI
        self.kpis = {
            'total_experiments': 0,
            'technical_failures': 0,
            'operational_failures': 0,
            'total_gain': 0,
            'rework_count': 0,
            'integration_days': 0
        }

    def setup_minimal_model(self):
        # デフォルト設定
        params = {
            'dr_threshold': 5.0,
            'delay_res_proto': 2, 'loss_res_proto': 0.1, 'dist_res_proto': 0.1,
            'delay_proto_ana': 1, 'loss_proto_ana': 0.05, 'dist_proto_ana': 0.05,
            'delay_ana_res': 2, 'loss_ana_res': 0.1, 'dist_ana_res': 0.2
        }
        self.setup_with_params(params)

    def setup_with_params(self, params):
        self.dr_threshold = params.get('dr_threshold', 5.0)
        
        # 技術要素の作成
        self.tech_items = [
            TechnologyItem("Material Mix", tacitness=0.7),
            TechnologyItem("Process Conditions", tacitness=0.4),
            TechnologyItem("Evaluation Criteria", tacitness=0.3)
        ]
        
        # トレードオフの設定 (Plan 2 特有)
        # Mixを詰めるとProcessが難しくなる等の関係
        self.tech_items[0].add_trade_off("Process Conditions", intensity=0.2)
        self.tech_items[1].add_trade_off("Evaluation Criteria", intensity=0.1)

        # リソースの作成
        from .experiment import Resource
        self.resources = {
            'Mixer': Resource('Mixer', failure_rate=0.02),
            'Tester': Resource('Tester', failure_rate=0.01)
        }

        # チームの作成
        research = Team("Research", wip_limit=2)
        prototype = Team("Prototype", wip_limit=1)
        analysis = Team("Analysis", wip_limit=2)
        
        # 組織ネットワークの設定 (パラメータから読み込み)
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
        
        self.teams = {t.name: t for t in [research, prototype, analysis]}

    def step(self):
        self.current_time += 1
        
        # リソース状態の更新と不確実性の再燃(減衰)
        for res in self.resources.values():
            res.update_status(self.current_time)
        for t in self.tech_items:
            t.apply_decay(decay_rate=0.0005)

        # 1. 計画: Researchが技術要素を選んでPrototypeに送る
        # 実行中（inboxやwipにある）の依頼がない場合のみ、新しい依頼を出す
        total_requests = len(self.teams['Prototype'].inbox) + len(self.teams['Analysis'].inbox) + len(self.teams['Research'].inbox)
        if total_requests == 0 and not self.dr_passed:
            target_tech = np.random.choice(self.tech_items)
            msg = {
                'type': 'experiment_request',
                'tech_items': [target_tech],
                'params': {'target_temp': 100, 'pressure': 50},
                'metadata': {'reason': 'Initial screening'}
            }
            self.teams['Research'].send_message(msg, self.teams['Prototype'], self.current_time)

        # 2. 試作・実験: Prototypeが届いた依頼を処理
        proto_team = self.teams['Prototype']
        for msg in list(proto_team.inbox):
            if msg['arrival_time'] <= self.current_time and msg.get('type') == 'experiment_request':
                proto_team.inbox.remove(msg)
                
                # コミュニケーションロスの影響。スキルも影響
                protocol_adherence = 0.8 + 0.2 * proto_team.skill
                if msg.get('distorted'): protocol_adherence *= 0.8
                if msg.get('lost'): protocol_adherence *= 0.7
                
                # 実験実行
                exp = Experiment(msg['tech_items'], proto_team, list(self.resources.values()), all_tech_items=self.tech_items)
                result = exp.run(protocol_adherence=protocol_adherence)
                
                # KPI更新
                self.kpis['total_experiments'] += 1
                if result.get('failure_type') == 2:
                    # 設備故障は回数に含めるが、gainは0
                    pass
                else:
                    self.kpis['total_gain'] += result['effective_gain']
                
                if not result['success']:
                    if result['failure_type'] == 0:
                        self.kpis['technical_failures'] += 1
                    elif result['failure_type'] == 1:
                        self.kpis['operational_failures'] += 1
                    else:
                        # 設備故障
                        pass
                
                # 結果をAnalysisに送る
                result_msg = {
                    'type': 'experiment_result',
                    'tech_items': msg['tech_items'],
                    'result': result,
                    'original_params': msg['params']
                }
                proto_team.send_message(result_msg, self.teams['Analysis'], self.current_time)

        # 3. 解析: Analysisが結果を解釈してResearchに戻す
        analysis_team = self.teams['Analysis']
        for msg in list(analysis_team.inbox):
            if msg['arrival_time'] <= self.current_time and msg.get('type') == 'experiment_result':
                analysis_team.inbox.remove(msg)
                
                # 解析における誤解 (Distortion)
                if msg.get('distorted'):
                    # 誤った解釈による「逆効果」や「手戻り」を表現
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

        # 4. 統合フェーズとDR判定
        total_evidence = sum(t.evidence for t in self.tech_items)
        if total_evidence >= self.dr_threshold and not self.dr_passed:
            # 統合資料作成タスクのシミュレーション
            # 分散組織の影響（doc_level, data_qualityの平均）
            avg_doc = np.mean([t.doc_level for t in self.tech_items])
            avg_data = np.mean([t.data_quality for t in self.tech_items])
            
            # 統合の難易度: 分散による誤解(distortion)の蓄積を反映
            integration_difficulty = 1.0 + self.kpis['rework_count'] * 0.1
            integration_time = 10 * integration_difficulty / (avg_doc * 0.5 + avg_data * 0.5 + 0.1)
            
            self.kpis['integration_days'] = integration_time
            self.dr_passed = True
            # print(f"[{self.current_time}] DR Passed! Integration took {integration_time:.1f} days.")

    def run(self, steps=100):
        for _ in range(steps):
            self.step()
        return self.kpis

    def get_tech_status(self):
        return {t.name: {'evidence': t.evidence, 'uncertainty': t.uncertainty, 'maturity': t.maturity} for t in self.tech_items}

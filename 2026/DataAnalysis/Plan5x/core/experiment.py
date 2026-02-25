import numpy as np

class Resource:
    def __init__(self, name, capacity=1, failure_rate=0.01):
        self.name = name
        self.capacity = capacity
        self.failure_rate = failure_rate
        self.down_until = 0

    def is_down(self):
        return self.down_until > 0

    def update_status(self, current_time):
        if self.down_until <= current_time:
            self.down_until = 0
            # 故障判定 (Plan 2)
            if np.random.random() < self.failure_rate:
                repair_time = np.random.randint(1, 5)
                self.down_until = current_time + repair_time

class Experiment:
    def __init__(self, tech_items, executor_team, resources, all_tech_items=None):
        self.tech_items = tech_items  # 対象となる技術要素のリスト
        self.executor_team = executor_team
        self.resources = resources
        self.all_tech_items = all_tech_items
        
    def run(self, protocol_adherence=1.0, noise_level=0.1):
        """
        実験を実行し、アウトカムを返す
        """
        # (A) 設備故障のチェック (Plan 2)
        resource_failure = False
        for res in self.resources:
            if res.is_down():
                resource_failure = True
                break
        
        if resource_failure:
            return {
                'success': False,
                'failure_type': 2, # 設備故障
                'effective_gain': 0,
                'info_gain_base': 0
            }

        # (B) 現象として成功/失敗
        # チームの平均エージェントスキルの計算
        avg_agent_skill = np.mean([a.skill for a in self.executor_team.agents])
        combined_skill = (self.executor_team.skill + avg_agent_skill) / 2.0

        # 成熟度が高いほど成功しやすい。チームスキルも影響 (Plan 5)
        avg_maturity = np.mean([t.maturity for t in self.tech_items])
        success_prob = 0.3 + 0.4 * avg_maturity + 0.2 * combined_skill
        success = np.random.random() < success_prob
        
        # 失敗理由の分類
        # 0: 技術的失敗 (学び大), 1: 運用的失敗 (学び小)
        failure_type = 0
        if not success:
            # プロトコル遵守度が低いか、スキルが低いと運用的失敗になりやすい
            op_failure_prob = 0.5 * (1 - protocol_adherence) + 0.5 * (1 - combined_skill)
            if np.random.random() < op_failure_prob:
                failure_type = 1 # 運用的失敗
        
        # (C) 情報として有効/無効
        # 1回のアウトカムから得られるベース情報量
        info_gain_base = np.random.beta(2, 5)
        
        # 失敗時の情報量調整
        if not success:
            if failure_type == 0:
                info_gain_base *= 0.8 # 技術的失敗は学びがある
            else:
                info_gain_base *= 0.2 # 運用的失敗は学びが少ない
                noise_level += 0.3    # ノイズも増える
                
        # 実効情報量 (Effective Gain)
        effective_gain = info_gain_base * (1 - noise_level) * protocol_adherence
        effective_gain = max(0, effective_gain)
        
        # 各技術要素の状態を更新 (トレードオフも考慮 - Plan 2)
        for t in self.tech_items:
            t.update_state(effective_gain, success, all_items=self.all_tech_items)
            
        return {
            'success': success,
            'failure_type': failure_type if not success else None,
            'effective_gain': effective_gain,
            'info_gain_base': info_gain_base
        }

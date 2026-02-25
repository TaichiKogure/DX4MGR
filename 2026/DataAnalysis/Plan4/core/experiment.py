import numpy as np

class Experiment:
    def __init__(self, tech_items, executor_team, resources):
        self.tech_items = tech_items  # 対象となる技術要素のリスト
        self.executor_team = executor_team
        self.resources = resources
        
    def run(self, protocol_adherence=1.0, noise_level=0.1):
        """
        実験を実行し、アウトカムを返す
        """
        # (A) 現象として成功/失敗
        # 成熟度が高いほど成功しやすい。チームスキルも影響
        avg_maturity = np.mean([t.maturity for t in self.tech_items])
        success_prob = 0.3 + 0.5 * avg_maturity + 0.1 * self.executor_team.skill
        success = np.random.random() < success_prob
        
        # 失敗理由の分類
        # 0: 技術的失敗 (学び大), 1: 運用的失敗 (学び小)
        failure_type = 0
        if not success:
            op_failure_prob = 0.5 * (1 - protocol_adherence) + 0.5 * (1 - self.executor_team.skill)
            if np.random.random() < op_failure_prob:
                failure_type = 1 # 運用的失敗
        
        # (B) 情報として有効/無効
        # 1回のアウトカムから得られるベース情報量
        # Beta分布 (α=2, β=5) で0に近く、たまに大きい値
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
        
        # 各技術要素の状態を更新
        for t in self.tech_items:
            t.update_state(effective_gain, success)
            
        return {
            'success': success,
            'failure_type': failure_type if not success else None,
            'effective_gain': effective_gain,
            'info_gain_base': info_gain_base
        }

class Resource:
    def __init__(self, name, capacity=1):
        self.name = name
        self.capacity = capacity
        self.queue = []
        self.current_users = []

    def request(self, user, duration):
        # 簡易的なDES的リソース管理
        pass

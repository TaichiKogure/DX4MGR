import numpy as np

class TechnologyItem:
    def __init__(self, name, uncertainty=1.0, maturity=0.0, tacitness=0.5, data_quality=0.5, doc_level=0.5):
        self.name = name
        # 状態変数
        self.uncertainty = uncertainty      # 不確実性 [0, 1]
        self.maturity = maturity            # 再現性/成熟度 [0, 1]
        self.tacitness = tacitness          # 暗黙知度 [0, 1]
        self.data_quality = data_quality    # データ整備度 [0, 1]
        self.doc_level = doc_level          # 文書化度 [0, 1]
        
        # 証拠量
        self.evidence = 0.0
        
        # トレードオフ関係 (他要素の名称: 影響係数)
        # ある要素の確からしさを上げると、別の要素の不確実性が増す
        self.trade_offs = {}

    def add_trade_off(self, other_name, intensity=0.1):
        self.trade_offs[other_name] = intensity

    def update_state(self, effective_gain, success, all_items=None):
        """
        実験結果に基づいて状態を更新する
        """
        self.evidence += effective_gain
        # 不確実性は証拠量に応じて減少（指数関数的な減衰）
        old_uncert = self.uncertainty
        self.uncertainty = max(0.0, self.uncertainty * np.exp(-effective_gain * 0.5))
        
        # トレードオフの適用 (不確実性の再燃)
        if all_items and effective_gain > 0:
            improvement = old_uncert - self.uncertainty
            for other_name, intensity in self.trade_offs.items():
                target = next((item for item in all_items if item.name == other_name), None)
                if target:
                    target.uncertainty = min(1.0, target.uncertainty + improvement * intensity)
        
        if success:
            # 成功した場合、成熟度が向上
            self.maturity = min(1.0, self.maturity + effective_gain * 0.1)
        
    def improve_quality(self, gain):
        self.doc_level = min(1.0, self.doc_level + gain)
        self.data_quality = min(1.0, self.data_quality + gain * 0.8)
        self.uncertainty = max(0.0, self.uncertainty - gain * 0.1) # 整理されると不確実性も減る

    def apply_decay(self, decay_rate=0.001):
        """
        時間経過とともに「確からしさ」が低下する
        """
        # 放置すると不確実性が微増する
        self.uncertainty = min(1.0, self.uncertainty + decay_rate)

    def __repr__(self):
        return f"Tech({self.name}, Evid={self.evidence:.2f}, Uncert={self.uncertainty:.2f}, Mat={self.maturity:.2f})"

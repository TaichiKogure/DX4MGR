import numpy as np
import json
import os

class QualityGate:
    def __init__(self, name, threshold, operator=">"):
        self.name = name
        self.threshold = threshold
        self.operator = operator
        self.status = "PENDING"
        self.value = None

    def check(self, value):
        self.value = value
        if self.operator == ">":
            self.status = "PASS" if value > self.threshold else "FAIL"
        elif self.operator == "<":
            self.status = "PASS" if value < self.threshold else "FAIL"
        elif self.operator == ">=":
            self.status = "PASS" if value >= self.threshold else "FAIL"
        elif self.operator == "<=":
            self.status = "PASS" if value <= self.threshold else "FAIL"
        return self.status

class Analyzer:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def calculate_confidence_interval(self, data, confidence=0.95):
        """
        信頼区間 (Confidence Interval) の計算
        """
        a = 1.0 * np.array(data)
        n = len(a)
        if n < 2: return np.mean(a), np.mean(a), np.mean(a)
        m, se = np.mean(a), np.std(a, ddof=1) / np.sqrt(n)
        # z-score (簡略化のため正規分布近似)
        z = 1.96 if confidence == 0.95 else 2.58
        h = se * z
        return m, m-h, m+h

    def run_quality_gates(self, summary_list, criteria):
        """
        検証ゲート (Quality Gates) の実行
        criteria: { "avg_wait": {"max": 10}, "throughput": {"min": 4.0}, "max_reworks": 2.0 }
        """
        avg_wait = np.mean([s["avg_wait"] for s in summary_list])
        avg_tp = np.mean([s["throughput"] for s in summary_list])
        avg_wip = np.mean([s["avg_wip"] for s in summary_list])
        avg_reworks = np.mean([s["avg_reworks"] for s in summary_list])
        
        gates = []
        
        # Gate 1: 基本的ステータス判定 (スループットと待ち時間)
        g1_tp = QualityGate("Gate1: スループット基準", criteria.get("min_throughput", 0), ">")
        g1_tp.check(avg_tp)
        gates.append(g1_tp)
        
        g1_wait = QualityGate("Gate1: 平均待ち時間基準", criteria.get("max_wait", 100), "<")
        g1_wait.check(avg_wait)
        gates.append(g1_wait)

        # Gate 2: パラメータ変異耐性 (ばらつき/CV判定)
        cv_wait = np.std([s["avg_wait"] for s in summary_list]) / (avg_wait + 1e-9)
        g2_cv = QualityGate("Gate2: 安定性 (CV)", criteria.get("max_cv", 0.5), "<")
        g2_cv.check(cv_wait)
        gates.append(g2_cv)

        # Gate 3: 統計的有意性 (95%信頼区間の幅)
        m, low, high = self.calculate_confidence_interval([s["throughput"] for s in summary_list])
        ci_width_ratio = (high - low) / (m + 1e-9)
        g3_ci = QualityGate("Gate3: 信頼区間精度", criteria.get("max_ci_width", 0.2), "<")
        g3_ci.check(ci_width_ratio)
        gates.append(g3_ci)

        # Gate 4: 差し戻し負荷判定 (Ver6新規)
        g4_rework = QualityGate("Gate4: 平均差し戻し回数", criteria.get("max_reworks", 5.0), "<")
        g4_rework.check(avg_reworks)
        gates.append(g4_rework)

        results = {
            "overall_status": "PASS" if all(g.status == "PASS" for g in gates) else "FAIL",
            "gates": [
                {
                    "name": g.name,
                    "status": g.status,
                    "value": round(float(g.value), 4),
                    "threshold": g.threshold
                } for g in gates
            ],
            "metrics": {
                "avg_wait": avg_wait,
                "avg_throughput": avg_tp,
                "avg_reworks": avg_reworks
            }
        }
        return results

    def compare_scenarios(self, base_summary_list, target_summary_list):
        """
        仮説検定 (Hypothesis Tests) による改善効果の判定
        """
        base_tp = [s["throughput"] for s in base_summary_list]
        target_tp = [s["throughput"] for s in target_summary_list]
        
        m_base, l_base, h_base = self.calculate_confidence_interval(base_tp)
        m_target, l_target, h_target = self.calculate_confidence_interval(target_tp)
        
        # 簡易的な有意差判定 (信頼区間が重ならないか)
        significant = h_base < l_target or h_target < l_base
        
        improvement = (m_target - m_base) / (m_base + 1e-9) * 100
        
        return {
            "base_mean": round(m_base, 3),
            "target_mean": round(m_target, 3),
            "improvement_pct": round(improvement, 2),
            "statistically_significant": significant,
            "base_ci": [round(l_base, 3), round(h_base, 3)],
            "target_ci": [round(l_target, 3), round(h_target, 3)]
        }

    def save_analysis_report(self, results, filename="analysis_report.json"):
        with open(os.path.join(self.out_dir, filename), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

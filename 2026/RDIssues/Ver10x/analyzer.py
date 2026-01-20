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
        Statistical Quality (max_ci_width, completed >= N) と P90/P95 Wait Time を重視
        """
        # 統計品質の基本量
        completed_counts = [s.get("completed_count", 0) for s in summary_list]
        avg_completed = np.mean(completed_counts)
        
        # P90リードタイムの平均
        p90_lts = [s.get("p90_wait") if "p90_wait" in s else s.get("lead_time_p90", 0) for s in summary_list]
        avg_p90_lt = np.mean(p90_lts)
        
        avg_tp = np.mean([s["throughput"] for s in summary_list])
        avg_wip = np.mean([s["avg_wip"] for s in summary_list])
        avg_reworks = np.mean([s["avg_reworks"] for s in summary_list])
        
        gates = []
        
        # Gate 1: 統計的十分性 (完了ジョブ数下限)
        g1_comp = QualityGate("Gate1: 完了ジョブ数下限", criteria.get("min_completed", 10), ">=")
        g1_comp.check(avg_completed)
        gates.append(g1_comp)
        
        # Gate 2: 統計的精度 (CI幅 / TP)
        m_tp, low_tp, high_tp = self.calculate_confidence_interval([s["throughput"] for s in summary_list])
        ci_width_tp = (high_tp - low_tp) / (m_tp + 1e-9)
        g2_ci = QualityGate("Gate2: 統計品質(CI幅)", criteria.get("max_ci_width", 0.3), "<")
        g2_ci.check(ci_width_tp)
        gates.append(g2_ci)

        # Gate 3: 待ち時間 (P90)
        g3_p90 = QualityGate("Gate3: P90待ち時間基準", criteria.get("max_wait_p90", 150), "<")
        g3_p90.check(avg_p90_lt)
        gates.append(g3_p90)

        # Gate 4: 差し戻し負荷判定
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
                "avg_completed": avg_completed,
                "avg_tp": avg_tp,
                "avg_p90_lt": avg_p90_lt,
                "avg_reworks": avg_reworks,
                "ci_width_tp": ci_width_tp
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

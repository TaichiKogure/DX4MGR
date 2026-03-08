import pandas as pd
import numpy as np

def analyze_doe_results(summary_path):
    df = pd.DataFrame(pd.read_csv(summary_path))
    
    # 目的変数 (KPI)
    targets = ['total_gain', 'completed_jobs', 'dr_cost']
    # 説明変数 (パラメータ)
    features = ['dr1_period', 'dr_threshold', 'cost_per_review', 'rework_load_factor', 'decay', 'uncertainty_threshold', 'maturity_threshold']
    
    # 相関係数の算出
    corr_matrix = df[features + targets].corr()
    
    report = []
    report.append("# DOE感度分析サマリーレポート")
    report.append("\n## 1. 分析概要")
    report.append("- 試行回数: 100回")
    report.append("- 目的: 各種パラメータがプロジェクトの成果（Gain）、完了数、コストに与える影響を特定する。")
    
    report.append("\n## 2. パラメータ感度ランキング (相関係数ベース)")
    
    for target in targets:
        report.append(f"\n### ターゲット指標: {target}")
        corrs = corr_matrix[target][features].abs().sort_values(ascending=False)
        for param, val in corrs.items():
            influence = "高" if val > 0.5 else "中" if val > 0.2 else "低"
            direction = "正" if corr_matrix[target][param] > 0 else "負"
            report.append(f"- **{param}**: 感度={influence} ({val:.3f}), 方向={direction}")

    report.append("\n## 3. 結論")
    report.append("\n### 感度が高いパラメータ (有効なレバー)")
    # 相関が0.3以上のものを抽出
    high_impact = []
    for target in targets:
        high_impact.extend(corr_matrix[target][features][corr_matrix[target][features].abs() > 0.3].index.tolist())
    high_impact = list(set(high_impact))
    report.append("- " + ", ".join(high_impact))
    report.append("  - これらのパラメータを調整することで、プロジェクトの成功率や成果を大きくコントロール可能です。")
    
    report.append("\n### 感度が低いパラメータ (あまり影響しないもの)")
    low_impact = []
    for target in targets:
        low_impact.extend(corr_matrix[target][features][corr_matrix[target][features].abs() < 0.1].index.tolist())
    # 全てのターゲットに対して感度が低いもの
    very_low = [p for p in features if all(abs(corr_matrix[t][p]) < 0.2 for t in targets)]
    report.append("- " + (", ".join(very_low) if very_low else "特になし"))
    report.append("  - これらのパラメータは現状のシミュレーション範囲内では結果への影響が限定的です。")

    with open('reports/doe_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print("Sensitivity analysis report generated: reports/doe_analysis_report.md")

if __name__ == "__main__":
    analyze_doe_results('reports/doe_summary.csv')

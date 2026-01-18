import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# 自身のディレクトリをパスに追加（インポートエラー回避のため）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator_v5 import run_monte_carlo_v5, latin_hypercube_sampling, simulate_flow_v5
from analyzer_v5 import AnalyzerV5
import visualizer_v5 as viz

# 出力ディレクトリ
OUT_DIR = "2026/RDIssues/Ver5/output"
os.makedirs(OUT_DIR, exist_ok=True)

def run_v5_pipeline():
    print("=== DX4MGR Ver5: 実験主導型シミュレーション・検証ゲートフロー ===")
    
    # 1. シナリオ読み込み
    csv_path = "2026/RDIssues/Ver5/scenarios.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df_scenarios = pd.read_csv(csv_path)
    analyzer = AnalyzerV5(OUT_DIR)
    
    # 2. DOE (実験計画法) による探索
    # R&D工程の不確実性を捉えるため、パラメータ空間を体系的に探索します
    print("\n[Step 1: DOE (実験計画法) によるパラメータ探索]")
    doe_ranges = {
        "skill_factor": (0.8, 1.5),
        "dependency_level": (0.1, 0.6),
        "p_rework": (0.05, 0.3)
    }
    # 代表的な条件で探索
    fixed_params = {"days": 90, "arrival_vp_per_day": 5.0, "service_vp_per_day": 6.0, "approvers": 1}
    doe_samples = latin_hypercube_sampling(n_samples=30, param_ranges=doe_ranges)
    
    print(f"  {len(doe_samples)} パターンのパラメータ組み合わせを検証中...")
    doe_results = []
    for sample in doe_samples:
        params = {**fixed_params, **sample}
        res = simulate_flow_v5(**params, seed=42)
        doe_results.append({**sample, "throughput": res["summary"]["throughput"]})
    
    doe_df = pd.DataFrame(doe_results)
    viz.plot_doe_analysis(doe_df)
    plt.savefig(f"{OUT_DIR}/v5_step1_doe_analysis.png")
    plt.close()

    # 3. メインシミュレーション (並列実行)
    print("\n[Step 2: 各シナリオの並列モンテカルロシミュレーション]")
    all_summaries = {}
    all_waits = {}
    gate_reports = {}
    
    # 検証ゲートの基準 (Quality Gates)
    criteria = {
        "min_throughput": 4.5,
        "max_wait": 12.0,
        "max_cv": 0.4,       # 安定性指標 (CV < 0.4)
        "max_ci_width": 0.15 # 統計精度指標 (95%CI幅 < 15%)
    }

    for _, row in df_scenarios.iterrows():
        name = row['scenario_name']
        print(f"  シナリオ実行中: {name} ...")
        
        params = row.to_dict()
        n_trials = int(params.pop('n_trials'))
        params.pop('scenario_name')
        
        # 型変換
        for k in ['days', 'approvers']:
            if k in params: params[k] = int(params[k])
        
        # モンテカルロ実行 (並列処理で高速化)
        trials = run_monte_carlo_v5(n_trials=n_trials, use_parallel=True, **params)
        
        summaries = [t["summary"] for t in trials]
        all_summaries[name] = summaries
        all_waits[name] = [w for t in trials for w in t["logs"]["wait_times"]]
        
        # 検証ゲート判定 (Quality Gates)
        gate_res = analyzer.run_quality_gates(summaries, criteria)
        gate_reports[name] = gate_res
        
        # ゲートステータスの視覚化
        viz.plot_quality_gate_status(gate_res, title=f"Quality Gate: {name}")
        plt.savefig(f"{OUT_DIR}/v5_step2_gate_{name}.png")
        plt.close()

    # 4. 統計解析と比較 (仮説検定・信頼区間)
    print("\n[Step 3: 統計的解析と比較]")
    baseline_name = df_scenarios.iloc[0]['scenario_name']
    comparison_summary = {}
    
    print(f"  基準(Baseline): {baseline_name}")
    for name in all_summaries.keys():
        m, low, high = analyzer.calculate_confidence_interval([s["throughput"] for s in all_summaries[name]])
        comparison_summary[name] = {"mean": m, "ci": [low, high]}
        
        if name != baseline_name:
            comp = analyzer.compare_scenarios(all_summaries[baseline_name], all_summaries[name])
            sig_str = "【有意差あり】" if comp['statistically_significant'] else "【有意差なし】"
            print(f"  - {name:20}: 改善率 {comp['improvement_pct']:+6.1f}% {sig_str}")

    # 5. 全体可視化の生成
    print("\n[Step 4: 可視化レポートの生成]")
    viz.plot_comparison_with_ci(comparison_summary, title="全シナリオ比較: スループットと95%信頼区間")
    plt.savefig(f"{OUT_DIR}/v5_step3_comparison_ci.png")
    plt.close()
    
    viz.plot_wait_time_distribution(all_waits, title="Ver5 待ち時間分布比較 (P90/P95強調)")
    plt.savefig(f"{OUT_DIR}/v5_step3_compare_violin.png")
    plt.close()
    
    # CCDFの生成
    viz.plot_ccdf(all_waits, title="納期超過確率 (CCDF) 比較")
    plt.savefig(f"{OUT_DIR}/v5_step3_compare_ccdf.png")
    plt.close()

    # 6. 分析レポートの保存
    analyzer.save_analysis_report({
        "criteria": criteria,
        "gate_reports": gate_reports,
        "comparison_summary": comparison_summary
    }, "v5_final_analysis_report.json")

    print(f"\n=== 全工程完了 ===")
    print(f"画像および分析レポートは以下に保存されました:\n{os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    run_v5_pipeline()

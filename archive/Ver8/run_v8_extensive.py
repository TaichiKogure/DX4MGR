import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# 実行スクリプトのディレクトリを基準にする（絶対パス解決）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator_v8 import run_monte_carlo_v8, latin_hypercube_sampling
from analyzer_v8 import AnalyzerV8
import visualizer_v8 as viz

# 出力ディレクトリ
OUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def run_v8_extensive_pipeline():
    print("=== DX4MGR Ver8: 大規模パラメータ探索・ドラスティック検証 ===")

    # 1. パラメータの定義と動機
    # -------------------------------------------------------------------------
    # arrival_rate: 案件投入頻度。過負荷による「組織の詰まり」の限界点を特定するため。
    # rework_load_factor: 手戻り時の負荷倍率。修正がどれほど破壊的か（0.5=軽微, 2.0=泥沼）を検証。
    # dr_period: レビュー周期。フィードバック頻度が遅延に与える影響を把握。
    # bundle_size: 承認単位。バッチサイズによるリードタイムと効率のトレードオフを検証。
    # approvers: 承認者数。リソース投入によるスケーラビリティの確認。
    # proto_duration: 試作期間。前工程の「溜め」が後工程の安定に寄与するかを確認。
    # -------------------------------------------------------------------------

    # 2. DOE (実験計画法) による大規模探索
    print("\n[Step 1: 50パターンによる大規模DOE感度分析]")
    doe_ranges = {
        "arrival_rate": (0.2, 0.8),         # 投入頻度の変動
        "rework_load_factor": (0.5, 2.0),   # 手戻りの重さ
        "dr_period": (14, 120),             # フィードバック周期
        "bundle_size": (1, 8),              # バッチサイズ
        "approvers": (1, 4),                # リソース
        "proto_duration": (5, 40),          # 試作期間
    }
    fixed_params = {
        "days": 180,
        "small_exp_duration": 5,
        "max_rework_cycles": 5,
        "decay": 0.7,
        "sampling_interval": 5.0
    }

    # 50サンプルに増量
    doe_samples = latin_hypercube_sampling(n_samples=50, param_ranges=doe_ranges)

    print(f"  {len(doe_samples)} パターンのパラメータ組み合わせを検証中...")
    doe_results = []
    for sample in doe_samples:
        params = {**fixed_params, **sample}
        # 型変換（整数が必要なもの）
        for k in ["bundle_size", "approvers"]:
            params[k] = int(round(params[k]))
            
        res = run_monte_carlo_v8(n_trials=1, use_parallel=False, **params)[0]
        doe_results.append({**sample, "throughput": res["summary"]["throughput"]})

    doe_df = pd.DataFrame(doe_results)
    viz.plot_doe_analysis(doe_df, title="Ver8 大規模DOE感度分析 (50 Samples)")
    plt.savefig(os.path.join(OUT_DIR, "v8_extensive_step1_doe.png"))
    plt.close()

    # 3. ドラスティックなシナリオの実行
    print("\n[Step 2: ドラスティックな特定シナリオの検証]")
    
    # 直接シナリオを定義（CSVに頼らず、より過激な設定にする）
    drastic_scenarios = [
        {
            "scenario_name": "01_Baseline",
            "arrival_rate": 0.5, "rework_load_factor": 1.0, "dr_period": 60, 
            "bundle_size": 3, "approvers": 2, "proto_duration": 20, "n_trials": 50
        },
        {
            "scenario_name": "02_Collapse_Risk", # 組織崩壊リスク：高流入・高手戻り・低リソース
            "arrival_rate": 0.8, "rework_load_factor": 2.0, "dr_period": 90, 
            "bundle_size": 5, "approvers": 1, "proto_duration": 10, "n_trials": 50
        },
        {
            "scenario_name": "03_Agile_Extreme", # 極端なアジャイル：超頻回フィードバック・小バッチ
            "arrival_rate": 0.5, "rework_load_factor": 0.5, "dr_period": 14, 
            "bundle_size": 1, "approvers": 3, "proto_duration": 5, "n_trials": 50
        },
        {
            "scenario_name": "04_Waterfall_Rigid", # 硬直したWF：巨大バッチ・低頻度DR
            "arrival_rate": 0.5, "rework_load_factor": 1.5, "dr_period": 180, 
            "bundle_size": 10, "approvers": 2, "proto_duration": 40, "n_trials": 50
        },
        {
            "scenario_name": "05_Lean_Ideal", # 理想的なリーン：手戻り抑制・適正バッチ
            "arrival_rate": 0.5, "rework_load_factor": 0.3, "dr_period": 30, 
            "bundle_size": 2, "approvers": 3, "proto_duration": 15, "n_trials": 50
        }
    ]

    all_summaries = {}
    all_waits = {}
    all_reworks = {}
    gate_reports = {}
    gate_stats_by_scenario = {}
    avg_wip_by_node_by_scenario = {}

    criteria = {
        "min_throughput": 0.01,
        "max_wait": 250.0,
        "max_cv": 0.8,
        "max_ci_width": 0.6,
        "max_reworks": 6.0
    }

    analyzer = AnalyzerV8(OUT_DIR)

    for sc in drastic_scenarios:
        name = sc['scenario_name']
        print(f"  シナリオ実行中: {name} ...")

        params = sc.copy()
        n_trials = params.pop('n_trials')
        params.pop('scenario_name')
        
        # 固定パラメタ追加
        params.update(fixed_params)

        trials = run_monte_carlo_v8(n_trials=n_trials, use_parallel=True, **params)

        summaries = [t["summary"] for t in trials]
        all_summaries[name] = summaries
        all_waits[name] = [lt for t in trials for lt in t["logs"]["lead_times"]]
        all_reworks[name] = [rc for t in trials for rc in t["logs"]["rework_counts"]]

        # ゲート別統計
        gate_rows = []
        for t in trials:
            m = t.get("metrics", {})
            for s in (m.get("gate_stats", []) or []):
                gate_rows.append({"node_id": s.get("node_id"), "avg_wait_time": s.get("avg_wait_time", 0.0)})
        if gate_rows:
            df_gate = pd.DataFrame(gate_rows).groupby("node_id", as_index=False)["avg_wait_time"].mean()
            gate_stats_by_scenario[name] = df_gate.to_dict("records")

        # WIP統計
        wip_rows = []
        for t in trials:
            m = t.get("metrics", {})
            w = (m.get("wip", {}) or {}).get("avg_by_node", {}) or {}
            for node_id, v in w.items():
                wip_rows.append({"node_id": node_id, "avg_wip": float(v)})
        if wip_rows:
            df_wip = pd.DataFrame(wip_rows).groupby("node_id", as_index=False)["avg_wip"].mean()
            avg_wip_by_node_by_scenario[name] = dict(zip(df_wip["node_id"], df_wip["avg_wip"]))

        gate_res = analyzer.run_quality_gates(summaries, criteria)
        gate_reports[name] = gate_res

        viz.plot_quality_gate_status(gate_res, title=f"Quality Gate: {name}")
        plt.savefig(os.path.join(OUT_DIR, f"v8_extensive_gate_{name}.png"))
        plt.close()

    # 4. 解析と比較
    print("\n[Step 3: 統計的比較と考察]")
    baseline_name = drastic_scenarios[0]['scenario_name']
    comparison_summary = {}

    for name in all_summaries.keys():
        m, low, high = analyzer.calculate_confidence_interval([s["throughput"] for s in all_summaries[name]])
        comparison_summary[name] = {"mean": m, "ci": [low, high]}

        if name != baseline_name:
            comp = analyzer.compare_scenarios(all_summaries[baseline_name], all_summaries[name])
            sig_str = "【有意差あり】" if comp['statistically_significant'] else "【有意差なし】"
            print(f"  - {name:20}: 改善率 {comp['improvement_pct']:+6.1f}% {sig_str}")

    # 可視化
    viz.plot_comparison_with_ci(comparison_summary, title="ドラスティック・シナリオ比較: スループット")
    plt.savefig(os.path.join(OUT_DIR, "v8_extensive_step3_ci.png"))
    plt.close()

    viz.plot_wait_time_distribution(all_waits, title="リードタイム分布比較")
    plt.savefig(os.path.join(OUT_DIR, "v8_extensive_step3_violin.png"))
    plt.close()

    viz.plot_gate_wait_heatmap(gate_stats_by_scenario, title="ノード別平均待ち時間")
    plt.savefig(os.path.join(OUT_DIR, "v8_extensive_step4_wait_heatmap.png"))
    plt.close()

    # レポート
    analyzer.save_analysis_report({
        "criteria": criteria,
        "gate_reports": gate_reports,
        "comparison_summary": comparison_summary,
        "notes": "この検証では、50サンプルのDOEと5つの極端なシナリオを通じて、組織の限界性能を調査しました。"
    }, "v8_extensive_report.json")

    print(f"\n=== 全工程完了 ===")
    print(f"結果は {OUT_DIR} に保存されました。")

if __name__ == "__main__":
    run_v8_extensive_pipeline()

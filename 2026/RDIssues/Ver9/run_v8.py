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

def run_v8_pipeline():
    print("=== DX4MGR Ver8: 標準フロー・WIP可視化強化モデル ===")

    # 1. シナリオ読み込み
    csv_path = os.path.join(CURRENT_DIR, "scenarios.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_scenarios = pd.read_csv(csv_path)
    analyzer = AnalyzerV8(OUT_DIR)

    # 2. DOE (実験計画法) による探索
    print("\n[Step 1: DOE (実験計画法) によるパラメータ探索]")
    doe_ranges = {
        "rework_load_factor": (0.2, 1.5),
        "dr_period": (30, 180),
        "proto_duration": (10, 40),
    }
    fixed_params = {
        "days": 180,
        "arrival_rate": 0.5,
        "small_exp_duration": 5,
        "bundle_size": 3,
        "approvers": 2,
        "max_rework_cycles": 5,
        "decay": 0.7,
        "sampling_interval": 5.0
    }

    doe_samples = latin_hypercube_sampling(n_samples=30, param_ranges=doe_ranges)

    print(f"  {len(doe_samples)} パターンのパラメータ組み合わせを検証中...")
    doe_results = []
    for sample in doe_samples:
        params = {**fixed_params, **sample}
        # DOEは軽く1トライアル評価
        res = run_monte_carlo_v8(n_trials=1, use_parallel=False, **params)[0]
        doe_results.append({**sample, "throughput": res["summary"]["throughput"]})

    doe_df = pd.DataFrame(doe_results)
    viz.plot_doe_analysis(doe_df, title="Ver8 DOE 感度分析（標準フロー）")
    plt.savefig(os.path.join(OUT_DIR, "v8_step1_doe_analysis.png"))
    plt.close()

    # 3. メインシミュレーション (並列実行)
    print("\n[Step 2: 各シナリオの並列モンテカルロシミュレーション]")
    all_summaries = {}
    all_waits = {}
    all_reworks = {}
    gate_reports = {}

    gate_stats_by_scenario = {}
    avg_wip_by_node_by_scenario = {}

    criteria = {
        "min_throughput": 0.01,
        "max_wait": 200.0,
        "max_cv": 0.8,
        "max_ci_width": 0.5,
        "max_reworks": 5.0
    }

    for _, row in df_scenarios.iterrows():
        name = row['scenario_name']
        print(f"  シナリオ実行中: {name} ...")

        params = row.to_dict()
        n_trials = int(params.pop('n_trials'))
        params.pop('scenario_name')

        # 型変換
        for k in ['days', 'approvers', 'bundle_size']:
            if k in params and pd.notna(params[k]):
                params[k] = int(params[k])

        trials = run_monte_carlo_v8(n_trials=n_trials, use_parallel=True, **params)

        summaries = [t["summary"] for t in trials]
        all_summaries[name] = summaries

        all_waits[name] = [lt for t in trials for lt in t["logs"]["lead_times"]]
        all_reworks[name] = [rc for t in trials for rc in t["logs"]["rework_counts"]]

        # ゲート別stats（各trialの metrics.gate_stats を平均化）
        gate_rows = []
        for t in trials:
            m = t.get("metrics", {})
            for s in (m.get("gate_stats", []) or []):
                gate_rows.append({"node_id": s.get("node_id"), "avg_wait_time": s.get("avg_wait_time", 0.0)})

        if gate_rows:
            df_gate = pd.DataFrame(gate_rows).groupby("node_id", as_index=False)["avg_wait_time"].mean()
            gate_stats_by_scenario[name] = df_gate.to_dict("records")
        else:
            gate_stats_by_scenario[name] = []

        # WIP（ノード別平均WIP）を trial 全体から平均
        wip_rows = []
        for t in trials:
            m = t.get("metrics", {})
            w = (m.get("wip", {}) or {}).get("avg_by_node", {}) or {}
            for node_id, v in w.items():
                wip_rows.append({"node_id": node_id, "avg_wip": float(v)})

        if wip_rows:
            df_wip = pd.DataFrame(wip_rows).groupby("node_id", as_index=False)["avg_wip"].mean()
            avg_wip_by_node_by_scenario[name] = dict(zip(df_wip["node_id"], df_wip["avg_wip"]))
        else:
            avg_wip_by_node_by_scenario[name] = {}

        gate_res = analyzer.run_quality_gates(summaries, criteria)
        gate_reports[name] = gate_res

        viz.plot_quality_gate_status(gate_res, title=f"Quality Gate: {name}")
        plt.savefig(os.path.join(OUT_DIR, f"v8_step2_gate_{name}.png"))
        plt.close()

    # 4. 統計解析と比較
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

    # 5. 可視化
    print("\n[Step 4: 可視化レポートの生成]")
    viz.plot_comparison_with_ci(comparison_summary, title="全シナリオ比較: スループット(Ver8標準フロー)")
    plt.savefig(os.path.join(OUT_DIR, "v8_step3_comparison_ci.png"))
    plt.close()

    viz.plot_wait_time_distribution(all_waits, title="Ver8 待ち時間(Lead Time)分布比較")
    plt.savefig(os.path.join(OUT_DIR, "v8_step3_compare_violin.png"))
    plt.close()

    viz.plot_rework_distribution(all_reworks, title="Ver8 差し戻し回数分布比較")
    plt.savefig(os.path.join(OUT_DIR, "v8_step3_compare_reworks.png"))
    plt.close()

    viz.plot_gate_wait_heatmap(gate_stats_by_scenario, title="Ver8 ゲート別 平均待ち時間（詰まり）")
    plt.savefig(os.path.join(OUT_DIR, "v8_step4_gate_wait_heatmap.png"))
    plt.close()

    viz.plot_gate_wip_heatmap(avg_wip_by_node_by_scenario, title="Ver8 ゲート別 平均WIP（滞留）")
    plt.savefig(os.path.join(OUT_DIR, "v8_step4_gate_wip_heatmap.png"))
    plt.close()

    # 6. レポート保存
    analyzer.save_analysis_report({
        "criteria": criteria,
        "gate_reports": gate_reports,
        "comparison_summary": comparison_summary
    }, "v8_final_analysis_report.json")

    print(f"\n=== 全工程完了 ===")
    print(f"画像および分析レポートは以下に保存されました:\n{os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    run_v8_pipeline()

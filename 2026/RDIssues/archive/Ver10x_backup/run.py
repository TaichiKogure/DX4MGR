import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# 実行スクリプトのディレクトリを基準にする（絶対パス解決）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator import run_monte_carlo, latin_hypercube_sampling
from analyzer import Analyzer
import visualizer as viz

# 出力ディレクトリ
OUT_DIR = os.path.join(CURRENT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def run_pipeline():
    print("=== DX4MGR Ver10: 並列実験プラットフォームモデル ===")

    # 1. シナリオ読み込み (Step 8.1: Scenario sweep)
    csv_path = os.path.join(CURRENT_DIR, "scenarios.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_scenarios = pd.read_csv(csv_path)
    analyzer = Analyzer(OUT_DIR)
    
    # 共通パラメータ
    base_seed = 42

    # 2. DOE (実験計画法) による探索 (Step 8.3: LHS Latin Hypercube)
    print("\n[Step 1: DOE (実験計画法) によるパラメータ探索]")
    doe_ranges = {
        "rework_load_factor": (0.2, 1.5),
        "arrival_rate": (0.3, 0.8),
        "dr_period": (30, 180),
    }
    fixed_params = {
        "days": 180,
        "small_exp_duration": 5,
        "proto_duration": 20,
        "bundle_size": 3,
        "n_senior": 1,
        "n_coordinator": 1,
        "n_new": 0,
        "max_rework_cycles": 5,
        "decay": 0.7,
        "sampling_interval": 5.0,
        "rework_beta_a": 2.0,
        "rework_beta_b": 5.0,
        "rework_task_type_mix": 1.0,
        "conditional_prob_ratio": 0.8,
        "decision_latency_days": 2.0
    }

    doe_samples = latin_hypercube_sampling(n_samples=30, param_ranges=doe_ranges)

    print(f"  {len(doe_samples)} パターンのパラメータ組み合わせを検証中...")
    doe_results = []
    for sample in doe_samples:
        params = {**fixed_params, **sample}
        # DOEは軽く1トライアル評価 (base_seed固定)
        res = run_monte_carlo(n_trials=1, use_parallel=False, base_seed=base_seed, **params)[0]
        doe_results.append({**sample, "throughput": res["summary"]["throughput"]})

    doe_df = pd.DataFrame(doe_results)
    viz.plot_doe_analysis(doe_df, title="Ver10 DOE 感度分析（並列実験プラットフォーム）")
    plt.savefig(os.path.join(OUT_DIR, "step1_doe_analysis.png"))
    plt.close()

    # 3. メインシミュレーション (並列実行) (Step 8.1, 8.2)
    print("\n[Step 2: 各シナリオの並列モンテカルロシミュレーション]")
    all_summaries = {}
    all_metrics = {} # Step 7: 全メトリクスを保持
    all_waits = {}
    all_reworks = {}
    all_rework_weights = {}
    all_proliferated = {}
    all_wip_histories = {}
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
        for k in ['days', 'n_senior', 'n_coordinator', 'n_new', 'bundle_size']:
            if k in params and pd.notna(params[k]):
                params[k] = int(params[k])

        # Step 8.2: モンテカルロ実行 (並列)
        trials = run_monte_carlo(n_trials=n_trials, use_parallel=True, base_seed=base_seed, **params)

        summaries = [t["summary"] for t in trials]
        all_summaries[name] = summaries
        all_metrics[name] = [t["metrics"] for t in trials] # Step 7

        all_waits[name] = [lt for t in trials for lt in t["logs"]["lead_times"]]
        all_reworks[name] = [rc for t in trials for rc in t["logs"]["rework_counts"]]
        all_rework_weights[name] = [rw for t in trials for rw in t["logs"].get("rework_weights", [])]
        all_proliferated[name] = [pt for t in trials for pt in t["logs"].get("proliferated_tasks", [])]
        
        # WIP時系列 (Step 7: trial[0]の履歴を代表として使用)
        all_wip_histories[name] = trials[0]["logs"]["wip_history"]

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
        plt.savefig(os.path.join(OUT_DIR, f"gate_status_{name}.png"))
        plt.close()

    # 4. 統計解析と比較
    print("\n[Step 3: 統計的解析と比較]")
    baseline_name = df_scenarios.iloc[0]['scenario_name']
    comparison_summary = {}

    print(f"  基準(Baseline): {baseline_name}")
    
    # Step 7: 全指標の表示（標準出力）
    print("\n  --- 詳細指標 (Ver10 可観測性レポート) ---")
    header = f"{'Scenario':20} | {'TP':5} | {'P50':5} | {'P90':5} | {'AvgRwk':6} | {'AvgProl':6} | {'AvgWIP':6}"
    print(header)
    print("-" * len(header))

    for name in all_summaries.keys():
        # スループット信頼区間
        m, low, high = analyzer.calculate_confidence_interval([s["throughput"] for s in all_summaries[name]])
        comparison_summary[name] = {"mean": m, "ci": [low, high]}
        
        # 各種サマリ値（trialの平均）
        metrics_list = [m["summary"] for m in all_metrics[name]]
        tp = np.mean([s["throughput"] for s in metrics_list])
        p50 = np.mean([s["lead_time_p50"] for s in metrics_list])
        p90 = np.mean([s["lead_time_p90"] for s in metrics_list])
        rwk = np.mean([s["avg_reworks"] for s in metrics_list])
        prol = np.mean([s.get("avg_proliferated_tasks", 0) for s in metrics_list])
        wip = np.mean([s["avg_wip"] for s in metrics_list])
        
        print(f"{name:20} | {tp:5.3f} | {p50:5.1f} | {p90:5.1f} | {rwk:6.2f} | {prol:6.1f} | {wip:6.1f}")

    print("\n  --- 比較レポート (対Baseline) ---")
    for name in all_summaries.keys():
        if name != baseline_name:
            comp = analyzer.compare_scenarios(all_summaries[baseline_name], all_summaries[name])
            sig_str = "【有意差あり】" if comp['statistically_significant'] else "【有意差なし】"
            print(f"  - {name:20}: 改善率 {comp['improvement_pct']:+6.1f}% {sig_str}")

    # 5. 可視化
    print("\n[Step 4: 可視化レポートの生成]")
    viz.plot_comparison_with_ci(comparison_summary, title="全シナリオ比較: スループット(Ver10)")
    plt.savefig(os.path.join(OUT_DIR, "comparison_throughput.png"))
    plt.close()

    viz.plot_wait_time_distribution(all_waits, title="待ち時間分布比較")
    plt.savefig(os.path.join(OUT_DIR, "compare_violin.png"))
    plt.close()

    viz.plot_rework_distribution(all_reworks, title="差し戻し回数分布比較")
    plt.savefig(os.path.join(OUT_DIR, "compare_reworks.png"))
    plt.close()

    # Step 7: 新規可視化
    viz.plot_ccdf(all_waits, title="超過確率カーブ (CCDF)")
    plt.savefig(os.path.join(OUT_DIR, "ccdf_analysis.png"))
    plt.close()

    viz.plot_wip_time_series(all_wip_histories, title="WIP時系列推移")
    plt.savefig(os.path.join(OUT_DIR, "wip_time_series.png"))
    plt.close()

    viz.plot_rework_weight_distribution(all_rework_weights, title="差し戻し重み分布")
    plt.savefig(os.path.join(OUT_DIR, "rework_weight_dist.png"))
    plt.close()

    viz.plot_proliferated_tasks_distribution(all_proliferated, title="増殖タスク数分布")
    plt.savefig(os.path.join(OUT_DIR, "proliferated_dist.png"))
    plt.close()

    viz.plot_gate_wait_heatmap(gate_stats_by_scenario, title="ゲート別 平均待ち時間")
    plt.savefig(os.path.join(OUT_DIR, "gate_wait_heatmap.png"))
    plt.close()

    viz.plot_gate_wip_heatmap(avg_wip_by_node_by_scenario, title="ゲート別 平均WIP")
    plt.savefig(os.path.join(OUT_DIR, "gate_wip_heatmap.png"))
    plt.close()

    # 6. レポート保存
    analyzer.save_analysis_report({
        "criteria": criteria,
        "gate_reports": gate_reports,
        "comparison_summary": comparison_summary
    }, "final_analysis_report.json")

    print(f"\n=== 全工程完了 ===")
    print(f"画像および分析レポートは以下に保存されました:\n{os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    run_pipeline()

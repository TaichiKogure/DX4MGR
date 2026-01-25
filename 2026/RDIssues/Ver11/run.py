import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# 実行スクリプトのディレクトリを基準にする（絶対パス解決）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator import run_monte_carlo
from analyzer import Analyzer
import visualizer as viz

# 出力ディレクトリ (デフォルトは "output")
DEFAULT_OUT_DIR = os.path.join(CURRENT_DIR, "output")
DEFAULT_SCENARIOS = "scenarios.csv"

def _safe_mean(values, default=np.nan):
    """空リストでも落ちずに平均を返す（長期運用向けの安全策）。"""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float(default)

def _prompt_scenarios_input() -> str:
    print("シナリオCSVのパスを入力してください。")
    print("  - 空入力: デフォルトの scenarios.csv を使用")
    print("  - ファイル: 指定ファイルを実行")
    print("  - ディレクトリ: 内部のCSVを全て実行")
    return input("Path: ").strip()

def _resolve_csv_paths(raw_input: str):
    if not raw_input:
        default_path = os.path.join(CURRENT_DIR, DEFAULT_SCENARIOS)
        return [default_path] if os.path.exists(default_path) else []

    candidates = [raw_input]
    if not os.path.isabs(raw_input):
        candidates.append(os.path.join(CURRENT_DIR, raw_input))

    for path in candidates:
        if os.path.isfile(path):
            return [os.path.abspath(path)]
        if os.path.isdir(path):
            csvs = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.lower().endswith(".csv")
            ]
            return sorted(os.path.abspath(p) for p in csvs)

    return []

def run_pipeline(scenarios_path="scenarios.csv", out_dir=None):
    print("=== DX4MGR Ver11: 並列実験プラットフォームモデル ===")
    
    OUT_DIR = out_dir or DEFAULT_OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")

    # 1. シナリオ読み込み (Step 8.1: Scenario sweep)
    csv_path = scenarios_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(CURRENT_DIR, csv_path)
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df_scenarios = pd.read_csv(csv_path)
    analyzer = Analyzer(OUT_DIR)
    
    # 共通パラメータ
    base_seed = 42


    # 3. メインシミュレーション (並列実行) (Step 8.1, 8.2)
    print("\n[Step 1: 各シナリオの並列モンテカルロシミュレーション]")
    all_summaries = {}
    all_metrics = {} # Step 7: 全メトリクスを保持
    all_waits = {}
    all_reworks = {}
    all_rework_weights = {}
    all_proliferated = {}
    all_wip_histories = {}
    gate_reports = {}
    all_metrics_for_scorecard = {}

    gate_stats_by_scenario = {}
    avg_wip_by_node_by_scenario = {}

    # 判定基準の更新 (Ver11カスタム)
    # 統計品質 (完了数下限, CI幅) と P90待ち時間を重視
    criteria = {
        "min_completed": 5,        # 完了ジョブ数下限
        "max_ci_width": 0.4,       # CI幅 / TP
        "max_wait_p90": 250.0,     # P90待ち時間 (以前の平均基準より緩和しつつP90で縛る)
        "max_reworks": 5.0
    }

    for _, row in df_scenarios.iterrows():
        name = row['scenario_name']
        print(f"  シナリオ実行中: {name} ...")

        params = row.to_dict()
        n_trials = int(params.pop('n_trials'))
        params.pop('scenario_name')

        # 型変換
        for k in ['days', 'n_senior', 'n_coordinator', 'n_new', 'bundle_size', 'n_servers_proto']:
            if k in params and pd.notna(params[k]):
                params[k] = int(params[k])
        for k in ['arrival_rate', 'small_exp_duration', 'proto_duration', 'rework_load_factor', 'decay', 'friction_alpha', 'decision_latency_days']:
            if k in params and pd.notna(params[k]):
                params[k] = float(params[k])

        # Step 8.2: モンテカルロ実行 (並列)
        trials = run_monte_carlo(n_trials=n_trials, use_parallel=True, base_seed=base_seed, **params)

        summaries = [t.get("summary", {}) for t in trials if t.get("summary")]
        if not summaries:
            # 全てのトライアルでジョブが完了しなかった場合
            summaries = [{"completed_count": 0, "throughput": 0.0, "lead_time_p90": 9999.0, "avg_wip": 0.0, "avg_reworks": 0.0}]

        all_summaries[name] = summaries
        all_metrics[name] = [(t.get("metrics") or {}) for t in trials]  # ← None/欠損に強くする

        all_waits[name] = [lt for t in trials for lt in t["logs"]["lead_times"]]
        all_reworks[name] = [rc for t in trials for rc in t["logs"]["rework_counts"]]
        all_rework_weights[name] = [rw for t in trials for rw in t["logs"].get("rework_weights", [])]
        all_proliferated[name] = [pt for t in trials for pt in t["logs"].get("proliferated_tasks", [])]

        # WIP時系列 (Step 7: trial[0]の履歴を代表として使用)
        all_wip_histories[name] = trials[0]["logs"]["wip_history"]

        # ゲート別stats（各trialの metrics.gate_stats を平均化）
        gate_rows = []
        for t in trials:
            m = t.get("metrics", {}) or {}
            for s in (m.get("gate_stats", []) or []):
                gate_rows.append({"node_id": s.get("node_id"), "avg_wait_time": s.get("avg_wait_time", 0.0)})

        if gate_rows:
            df_gate = pd.DataFrame(gate_rows).groupby("node_id", as_index=False)["avg_wait_time"].mean()
            gate_stats_by_scenario[name] = df_gate.to_dict("records")
        else:
            gate_stats_by_scenario[name] = []

        # ジョブログ（摩擦メトリクス等）の保存
        job_log_all = []
        for i, t in enumerate(trials):
            for entry in t["logs"].get("job_logs", []):
                entry["trial_id"] = i
                job_log_all.append(entry)
        
        if job_log_all:
            df_job_log = pd.DataFrame(job_log_all)
            df_job_log.to_csv(os.path.join(OUT_DIR, f"job_details_{name}.csv"), index=False)

            # Job detail visualizations (sample Gantt / time heatmap / wait distribution)
            viz.plot_job_gantt(df_job_log, title=f"ジョブ進行（サンプル）: {name}", max_jobs=30)
            plt.savefig(os.path.join(OUT_DIR, f"job_gantt_{name}.png"))
            plt.close()

            viz.plot_gate_wait_heatmap_by_time(df_job_log, bin_days=30, title=f"ゲート別待ち時間（時間帯）: {name}")
            plt.savefig(os.path.join(OUT_DIR, f"job_wait_heatmap_{name}.png"))
            plt.close()

            viz.plot_gate_wait_distribution(df_job_log, title=f"ゲート別待ち時間分布: {name}")
            plt.savefig(os.path.join(OUT_DIR, f"job_wait_dist_{name}.png"))
            plt.close()

        # WIP（ノード別平均WIP）を trial 全体から平均
        wip_rows = []
        for t in trials:
            m = t.get("metrics", {}) or {}
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

        # スコアカード用データの蓄積
        all_metrics_for_scorecard[name] = {
            "p90": _safe_mean([s.get("p90_wait") if "p90_wait" in s else s.get("lead_time_p90") for s in summaries]),
            "tp": _safe_mean([s.get("throughput") for s in summaries]),
            "wip": _safe_mean([s.get("avg_wip") for s in summaries]),
            "rework": _safe_mean([s.get("avg_reworks") for s in summaries])
        }

        # Per-scenario gate charts are replaced by a consolidated CSV summary.

    # Consolidated quality gate summary (one CSV for all scenarios)
    gate_rows = []
    for scenario, report in gate_reports.items():
        row = {
            "scenario": scenario,
            "overall_status": report.get("overall_status", "")
        }
        for idx, gate in enumerate(report.get("gates", []), start=1):
            prefix = f"gate{idx}"
            row[f"{prefix}_name"] = gate.get("name")
            row[f"{prefix}_status"] = gate.get("status")
            row[f"{prefix}_value"] = gate.get("value")
            row[f"{prefix}_threshold"] = gate.get("threshold")
        metrics = report.get("metrics", {}) or {}
        row["avg_completed"] = metrics.get("avg_completed")
        row["avg_tp"] = metrics.get("avg_tp")
        row["avg_p90_lt"] = metrics.get("avg_p90_lt")
        row["avg_reworks"] = metrics.get("avg_reworks")
        row["ci_width_tp"] = metrics.get("ci_width_tp")
        gate_rows.append(row)

    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(
            os.path.join(OUT_DIR, "quality_gate_summary.csv"),
            index=False
        )

    # 4. 統計解析と比較
    print("\n[Step 3: 統計的解析と比較]")
    baseline_name = df_scenarios.iloc[0]['scenario_name']
    comparison_summary = {}

    print(f"  基準(Baseline): {baseline_name}")
    
    # Step 7: 全指標の表示（標準出力）
    print("\n  --- 詳細指標 (Ver11 可観測性レポート) ---")
    header = f"{'Scenario':20} | {'TP':5} | {'P50':5} | {'P90':5} | {'AvgRwk':6} | {'AvgProl':6} | {'AvgWIP':6}"
    print(header)
    print("-" * len(header))

    for name in all_summaries.keys():
        # スループット信頼区間
        m, low, high = analyzer.calculate_confidence_interval([s["throughput"] for s in all_summaries[name]])
        comparison_summary[name] = {"mean": m, "ci": [low, high]}

        # 各種サマリ値（trialの平均）
        # "summary" が無い trial（例: completed_jobs=0 → metrics={"error":...}）を除外して集計
        summaries_in_metrics = []
        missing = 0
        for mm in all_metrics[name]:
            s = (mm or {}).get("summary")
            if isinstance(s, dict):
                summaries_in_metrics.append(s)
            else:
                missing += 1

        if missing:
            print(f"  [warn] {name}: metrics.summary が無い trial が {missing} 件あります（完了ジョブ0等の可能性）。集計から除外します。")

        tp = _safe_mean([s.get("throughput") for s in summaries_in_metrics])
        p50 = _safe_mean([s.get("lead_time_p50") for s in summaries_in_metrics])
        p90 = _safe_mean([s.get("lead_time_p90") for s in summaries_in_metrics])
        rwk = _safe_mean([s.get("avg_reworks") for s in summaries_in_metrics])
        prol = _safe_mean([s.get("avg_proliferated_tasks", 0) for s in summaries_in_metrics], default=0.0)
        wip = _safe_mean([s.get("avg_wip") for s in summaries_in_metrics])

        print(f"{name:20} | {tp:5.3f} | {p50:5.1f} | {p90:5.1f} | {rwk:6.2f} | {prol:6.1f} | {wip:6.1f}")

    print("\n  --- 比較レポート (対Baseline) ---")
    for name in all_summaries.keys():
        if name != baseline_name:
            comp = analyzer.compare_scenarios(all_summaries[baseline_name], all_summaries[name])
            sig_str = "【有意差あり】" if comp['statistically_significant'] else "【有意差なし】"
            print(f"  - {name:20}: 改善率 {comp['improvement_pct']:+6.1f}% {sig_str}")

    # 5. 可視化
    print("\n[Step 4: 可視化レポートの生成]")

    # スコアカード生成 (Ver11カスタム)
    print("  スコアカード生成中...")
    viz.plot_scorecard(all_metrics_for_scorecard, baseline_name, title="Ver11 シナリオ性能スコアカード")
    plt.savefig(os.path.join(OUT_DIR, "scenario_scorecard.png"))
    plt.close()

    viz.plot_comparison_with_ci(comparison_summary, title="全シナリオ比較: スループット(Ver11)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=None, help='output directory')
    args = parser.parse_args()

    raw_input = _prompt_scenarios_input()
    csv_paths = _resolve_csv_paths(raw_input)
    if not csv_paths:
        print("Error: 指定されたパスが見つかりませんでした。")
        sys.exit(1)

    if len(csv_paths) == 1:
        run_pipeline(scenarios_path=csv_paths[0], out_dir=args.out)
    else:
        base_out = args.out or DEFAULT_OUT_DIR
        for csv_path in csv_paths:
            stem = os.path.splitext(os.path.basename(csv_path))[0]
            out_dir = os.path.join(base_out, stem)
            run_pipeline(scenarios_path=csv_path, out_dir=out_dir)

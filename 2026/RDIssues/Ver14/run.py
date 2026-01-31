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

def _safe_mean(values, default=np.nan):
    """空リストでも落ちずに平均を返す（長期運用向けの安全策）。"""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float(default)

def _safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

def run_pipeline(scenarios_path=None, scenarios_dir=None, scenarios_file="scenarios.csv", out_dir=None):
    print("=== DX4MGR Ver14: Field Calibration Model ===")
    print("Enabled features: DRCalendar, WorkPackage, LatentRisk, Scheduler, AdoptionGate, DR Postponement")

    # 1. シナリオファイルのリスト作成
    csv_paths = []
    if scenarios_path:
        csv_paths.append(os.path.abspath(scenarios_path) if not os.path.isabs(scenarios_path) else scenarios_path)
    elif scenarios_dir:
        abs_scenarios_dir = os.path.abspath(scenarios_dir) if not os.path.isabs(scenarios_dir) else scenarios_dir
        if os.path.isdir(abs_scenarios_dir):
            for f in sorted(os.listdir(abs_scenarios_dir)):
                if f.endswith(".csv"):
                    csv_paths.append(os.path.join(abs_scenarios_dir, f))
        else:
            print(f"Error: Directory {abs_scenarios_dir} not found.")
            return
    else:
        path = os.path.join(CURRENT_DIR, scenarios_file or "scenarios.csv")
        csv_paths.append(path)

    if not csv_paths:
        print("No scenario CSV files found.")
        return

    BASE_OUT_DIR = out_dir or DEFAULT_OUT_DIR
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    print(f"Base Output directory: {BASE_OUT_DIR}")

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping.")
            continue
        
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n>>> Processing Scenarios from: {csv_path}")

        if len(csv_paths) > 1 or scenarios_dir:
            csv_out_dir = os.path.join(BASE_OUT_DIR, csv_name)
        else:
            csv_out_dir = BASE_OUT_DIR
        
        os.makedirs(csv_out_dir, exist_ok=True)
        
        try:
            df_scenarios = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        analyzer = Analyzer(csv_out_dir)
        base_seed = 42

        all_summaries = {}
        all_metrics = {} 
        all_waits = {}
        all_reworks = {}
        all_rework_weights = {}
        all_proliferated = {}
        all_wip_histories = {}
        gate_reports = {}
        all_metrics_for_scorecard = {}

        gate_stats_by_scenario = {}
        avg_wip_by_node_by_scenario = {}
        flow_rows = []
        cost_summary_rows = []
        loss_rows = []
        rework_summary_rows = []
        dr_gate_rows = []
        
        flow_node_order = ["SMALL_EXP", "BUNDLE_SMALL", "DR1", "MID_EXP", "BUNDLE_MID", "DR2", "FIN_EXP", "BUNDLE_FIN", "DR3"]
        criteria = {"min_completed": 5, "max_ci_width": 0.5, "max_wait_p90": 300.0, "max_reworks": 5.0}

        for _, row in df_scenarios.iterrows():
            name = row['scenario_name']
            print(f"  シナリオ実行中: {name} ...")

            SCENARIO_OUT_DIR = os.path.join(csv_out_dir, f"Output-{name}")
            os.makedirs(SCENARIO_OUT_DIR, exist_ok=True)

            params = row.to_dict()
            n_trials = int(params.pop('n_trials'))
            params.pop('scenario_name')

            # 型変換
            for k in ['days', 'n_senior', 'n_coordinator', 'n_new', 'bundle_size_small', 'bundle_size_mid', 'bundle_size_fin', 'n_servers_mid', 'n_servers_fin', 'dr1_period', 'dr2_period', 'dr3_period', 'dr_capacity', 'dr1_capacity', 'dr2_capacity', 'dr3_capacity', 'engineer_pool_size', 'n_servers_small', 'dr1_t', 'dr2_t', 'dr3_t']:
                if k in params and pd.notna(params[k]): params[k] = int(params[k])
            for k in ['arrival_rate', 'small_exp_duration', 'mid_exp_duration', 'fin_exp_duration', 'rework_load_factor', 'decay', 'friction_alpha', 'decision_latency_days', 'dr1_cost_per_review', 'dr2_cost_per_review', 'dr3_cost_per_review', 'dr2_rework_multiplier', 'hours_per_day_per_engineer', 'dr_quality', 'dr_quality_speed_alpha', 'rework_reinject_ratio', 'adoption_rate']:
                if k in params and pd.notna(params[k]): params[k] = float(params[k])

            trials = run_monte_carlo(n_trials=n_trials, use_parallel=True, base_seed=base_seed, **params)
            summaries = [t.get("summary", {}) for t in trials if t.get("summary")]
            if not summaries:
                summaries = [{"completed_count": 0, "throughput": 0.0, "lead_time_p90": 9999.0, "avg_wip": 0.0, "avg_reworks": 0.0}]

            all_summaries[name] = summaries
            all_metrics[name] = [(t.get("metrics") or {}) for t in trials]
            all_waits[name] = [lt for t in trials for lt in t["logs"]["lead_times"]]
            all_reworks[name] = [rc for t in trials for rc in t["logs"]["rework_counts"]]
            all_rework_weights[name] = [rw for t in trials for rw in t["logs"].get("rework_weights", [])]
            all_proliferated[name] = [pt for t in trials for pt in t["logs"].get("proliferated_tasks", [])]
            all_wip_histories[name] = trials[0]["logs"]["wip_history"]

            # 個別ログ保存
            current_job_log_all = []
            for i, t in enumerate(trials):
                for entry in t["logs"].get("job_logs", []):
                    entry["trial_id"] = i
                    current_job_log_all.append(entry)
            
            if current_job_log_all:
                df_job_log = pd.DataFrame(current_job_log_all)
                df_job_log.to_csv(os.path.join(SCENARIO_OUT_DIR, f"job_details_{name}.csv"), index=False)
                viz.plot_job_gantt(df_job_log, title=f"ジョブ進行（サンプル）: {name}", max_jobs=30)
                plt.savefig(os.path.join(SCENARIO_OUT_DIR, f"job_gantt_{name}.png"))
                plt.close()
                viz.plot_gate_wait_heatmap_by_time(df_job_log, bin_days=30, title=f"ゲート別待ち時間（時間帯）: {name}")
                plt.savefig(os.path.join(SCENARIO_OUT_DIR, f"job_wait_heatmap_{name}.png"))
                plt.close()
                viz.plot_gate_wait_distribution(df_job_log, title=f"ゲート別待ち時間分布: {name}")
                plt.savefig(os.path.join(SCENARIO_OUT_DIR, f"job_wait_dist_{name}.png"))
                plt.close()

            # 集計処理
            decision_latency = float(params.get("decision_latency_days", 0.0) or 0.0)
            
            gate_rows_scenario = []
            for t in trials:
                m = t.get("metrics", {}) or {}
                for s in (m.get("gate_stats", []) or []):
                    gate_rows_scenario.append({"node_id": s.get("node_id"), "avg_wait_time": s.get("avg_wait_time", 0.0)})
            
            gate_wait_map = {}
            if gate_rows_scenario:
                df_gate = pd.DataFrame(gate_rows_scenario).groupby("node_id", as_index=False)["avg_wait_time"].mean()
                gate_wait_map = dict(zip(df_gate["node_id"], df_gate["avg_wait_time"]))
                gate_stats_by_scenario[name] = df_gate.to_dict("records")
            
            work_stats = {}
            if current_job_log_all:
                df_work = pd.DataFrame(current_job_log_all)
                if all(col in df_work.columns for col in ["node_id", "wait_time", "effective_duration"]):
                    df_work = df_work.dropna(subset=["node_id", "wait_time", "effective_duration"])
                    if not df_work.empty:
                        df_work = df_work.groupby("node_id", as_index=False).agg(avg_wait=("wait_time", "mean"), avg_work=("effective_duration", "mean"))
                        work_stats = {row["node_id"]: row for row in df_work.to_dict("records")}

            for node_id in flow_node_order:
                avg_wait = float(gate_wait_map.get(node_id, 0.0))
                avg_work = 0.0
                if node_id in work_stats:
                    avg_wait = float(work_stats[node_id].get("avg_wait", avg_wait))
                    avg_work = float(work_stats[node_id].get("avg_work", 0.0))
                if node_id.startswith("DR"): avg_work = decision_latency
                flow_rows.append({"Scenario": name, "Node": node_id, "AvgTime": avg_wait + avg_work, "AvgWait": avg_wait, "AvgWork": avg_work})

            wip_rows_scenario = []
            for t in trials:
                m = t.get("metrics", {}) or {}
                w = (m.get("wip", {}) or {}).get("avg_by_node", {}) or {}
                for nid, v in w.items(): wip_rows_scenario.append({"node_id": nid, "avg_wip": float(v)})
            if wip_rows_scenario:
                df_wip = pd.DataFrame(wip_rows_scenario).groupby("node_id", as_index=False)["avg_wip"].mean()
                avg_wip_by_node_by_scenario[name] = dict(zip(df_wip["node_id"], df_wip["avg_wip"]))

            dr_cost_acc = {"DR1": [], "DR2": [], "DR3": []}
            total_cost_acc = []
            cost_per_completed_acc = []
            for t in trials:
                m = t.get("metrics", {}) or {}
                stats = m.get("gate_stats", []) or []
                cost_map = {"DR1": 0.0, "DR2": 0.0, "DR3": 0.0}
                for s in stats:
                    node_id = s.get("node_id")
                    if node_id in cost_map: cost_map[node_id] = float(s.get("total_cost", 0.0) or 0.0)
                total_cost = sum(cost_map.values())
                total_cost_acc.append(total_cost)
                for nid, val in cost_map.items(): dr_cost_acc[nid].append(val)
                completed = (m.get("summary") or {}).get("completed_count", 0)
                if completed > 0: cost_per_completed_acc.append(total_cost / completed)

            cost_summary_rows.append({
                "scenario": name,
                "dr1_cost_avg": _safe_mean(dr_cost_acc["DR1"], default=0.0),
                "dr2_cost_avg": _safe_mean(dr_cost_acc["DR2"], default=0.0),
                "dr3_cost_avg": _safe_mean(dr_cost_acc["DR3"], default=0.0),
                "total_review_cost_avg": _safe_mean(total_cost_acc, default=0.0),
                "cost_per_completed_job_avg": _safe_mean(cost_per_completed_acc, default=0.0)
            })

            gate_res = analyzer.run_quality_gates(summaries, criteria)
            gate_reports[name] = gate_res

            cur_loss = [m.get("loss", {}) for m in all_metrics[name] if isinstance(m, dict) and m.get("loss")]
            cur_dr = [m.get("dr_gate", {}) for m in all_metrics[name] if isinstance(m, dict) and m.get("dr_gate")]
            loss_p_avg = _safe_mean([_safe_get(l, ["time", "loss_per_primary"]) for l in cur_loss], default=0.0)
            rw_cost_p_avg = _safe_mean([_safe_get(l, ["cost", "rework_review_per_primary"]) for l in cur_loss], default=0.0)
            
            loss_rows.append({
                "scenario": name,
                "primary_completed_avg": _safe_mean([_safe_get(l, ["time", "primary", "count"]) for l in cur_loss], default=0.0),
                "time_loss_per_primary_avg_days": loss_p_avg,
                "rework_review_cost_per_primary_avg": rw_cost_p_avg
            })

            dr_gate_sum = {}
            for gid in ["DR1", "DR2", "DR3"]:
                cp90 = _safe_mean([_safe_get(d, ["primary", gid, "cycle_p90"]) for d in cur_dr], default=0.0)
                dr_gate_rows.append({"scenario": name, "gate": gid, "cycle_p90": cp90, "pass_rate": _safe_mean([_safe_get(d, ["primary", gid, "pass_rate"]) for d in cur_dr], default=0.0)})
                dr_gate_sum[gid] = {"cycle_p90": cp90}

            all_metrics_for_scorecard[name] = {
                "p90": _safe_mean([s.get("p90_wait") if "p90_wait" in s else s.get("lead_time_p90") for s in summaries]),
                "tp": _safe_mean([s.get("throughput") for s in summaries]),
                "wip": _safe_mean([s.get("avg_wip") for s in summaries]),
                "rework": _safe_mean([s.get("avg_reworks") for s in summaries]),
                "time_loss": loss_p_avg, "cost_loss": rw_cost_p_avg,
                "dr1_p90": dr_gate_sum.get("DR1", {}).get("cycle_p90", 0.0),
                "dr2_p90": dr_gate_sum.get("DR2", {}).get("cycle_p90", 0.0),
                "dr3_p90": dr_gate_sum.get("DR3", {}).get("cycle_p90", 0.0)
            }

            ms = [mm.get("summary") for mm in all_metrics[name] if isinstance(mm.get("summary"), dict)]
            rework_summary_rows.append({
                "scenario": name,
                "rework_completed_avg": _safe_mean([s.get("rework_completed_count") for s in ms], default=0.0),
                "rework_throughput_avg": _safe_mean([s.get("rework_throughput") for s in ms], default=0.0),
                "rework_lead_time_p90_avg": _safe_mean([s.get("rework_lead_time_p90") for s in ms], default=0.0),
            })

        # --- CSVごとのサマリー解析と比較 ---
        scenario_names_list = list(all_summaries.keys())
        if not scenario_names_list: continue

        baseline_name = scenario_names_list[0]
        for n in scenario_names_list:
            if "Baseline" in n:
                baseline_name = n
                break
        
        comparison_summary = {}
        print(f"  基準(Baseline): {baseline_name}")
        print("\n  --- 詳細指標 (Ver14 可観測性レポート) ---")
        header = f"{'Scenario':20} | {'TP':5} | {'P50':5} | {'P90':5} | {'AvgRwk':6} | {'AvgProl':6} | {'AvgWIP':6}"
        print(header)
        print("-" * len(header))

        for name in all_summaries.keys():
            m, low, high = analyzer.calculate_confidence_interval([s["throughput"] for s in all_summaries[name]])
            comparison_summary[name] = {"mean": m, "ci": [low, high]}
            sm = [mm.get("summary") for mm in all_metrics[name] if isinstance(mm.get("summary"), dict)]
            tp = _safe_mean([s.get("throughput") for s in sm])
            p50 = _safe_mean([s.get("lead_time_p50") for s in sm])
            p90 = _safe_mean([s.get("lead_time_p90") for s in sm])
            rwk = _safe_mean([s.get("avg_reworks") for s in sm])
            prol = _safe_mean([s.get("avg_proliferated_tasks", 0) for s in sm], default=0.0)
            wip = _safe_mean([s.get("avg_wip") for s in sm])
            print(f"{name:20} | {tp:5.3f} | {p50:5.1f} | {p90:5.1f} | {rwk:6.2f} | {prol:6.1f} | {wip:6.1f}")

        print("\n  --- 比較レポート (対Baseline) ---")
        for name in all_summaries.keys():
            if name != baseline_name:
                comp = analyzer.compare_scenarios(all_summaries[baseline_name], all_summaries[name])
                sig_str = "【有意差あり】" if comp['statistically_significant'] else "【有意差なし】"
                print(f"  - {name:20}: 改善率 {comp['improvement_pct']:+6.1f}% {sig_str}")

        print("\n[Step 4: 可視化レポートの生成]")
        viz.plot_scorecard(all_metrics_for_scorecard, baseline_name, title="Ver14 シナリオ性能スコアカード")
        plt.savefig(os.path.join(csv_out_dir, "scenario_scorecard.png"))
        plt.close()
        viz.plot_comparison_with_ci(comparison_summary, title="全シナリオ比較: スループット(Ver14)")
        plt.savefig(os.path.join(csv_out_dir, "comparison_throughput.png"))
        plt.close()
        viz.plot_wait_time_distribution(all_waits, title="待ち時間分布比較")
        plt.savefig(os.path.join(csv_out_dir, "compare_violin.png"))
        plt.close()
        viz.plot_rework_distribution(all_reworks, title="差し戻し回数分布比較")
        plt.savefig(os.path.join(csv_out_dir, "compare_reworks.png"))
        plt.close()
        viz.plot_ccdf(all_waits, title="超過確率カーブ (CCDF)")
        plt.savefig(os.path.join(csv_out_dir, "ccdf_analysis.png"))
        plt.close()
        viz.plot_wip_time_series(all_wip_histories, title="WIP時系列推移")
        plt.savefig(os.path.join(csv_out_dir, "wip_time_series.png"))
        plt.close()
        viz.plot_gate_wait_heatmap(gate_stats_by_scenario, title="ゲート別 平均待ち時間")
        plt.savefig(os.path.join(csv_out_dir, "gate_wait_heatmap.png"))
        plt.close()
        viz.plot_gate_wip_heatmap(avg_wip_by_node_by_scenario, title="ゲート別 平均WIP")
        plt.savefig(os.path.join(csv_out_dir, "gate_wip_heatmap.png"))
        plt.close()

        if flow_rows:
            df_f = pd.DataFrame(flow_rows); df_f.to_csv(os.path.join(csv_out_dir, "flow_time_breakdown.csv"), index=False)
            viz.plot_flow_time_breakdown(df_f, node_order=flow_node_order, title="Flow time breakdown (avg days)")
            plt.savefig(os.path.join(csv_out_dir, "flow_time_breakdown.png")); plt.close()

        if cost_summary_rows: pd.DataFrame(cost_summary_rows).to_csv(os.path.join(csv_out_dir, "dr_cost_summary.csv"), index=False)
        if loss_rows: pd.DataFrame(loss_rows).to_csv(os.path.join(csv_out_dir, "loss_breakdown.csv"), index=False)
        if dr_gate_rows: pd.DataFrame(dr_gate_rows).to_csv(os.path.join(csv_out_dir, "dr_gate_cycle_times.csv"), index=False)

        analyzer.save_analysis_report({"criteria": criteria, "gate_reports": gate_reports, "comparison_summary": comparison_summary}, "final_analysis_report.json")
        print(f"\n>>> 完了: {csv_path}\n出力先: {os.path.abspath(csv_out_dir)}")

    print("\n=== 全工程完了 ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', default=None, help='scenarios csv path')
    parser.add_argument('--scenarios-dir', default=None, help='directory for scenarios csv')
    parser.add_argument('--scenarios-file', default='scenarios.csv', help='scenarios csv filename')
    parser.add_argument('--out', default=None, help='output directory')
    args = parser.parse_args()
    run_pipeline(scenarios_path=args.scenarios, scenarios_dir=args.scenarios_dir, scenarios_file=args.scenarios_file, out_dir=args.out)

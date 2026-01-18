import argparse
import os
import sys
try:
    import yaml
except ImportError:
    yaml = None
import json
from runner.adapters import setup_standard_flow
from runner.experiment import run_monte_carlo
from analysis.metrics import calculate_metrics
from analysis.viz import plot_all_results

def load_config(path):
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            if yaml is None:
                raise ImportError("PyYAML is not installed. Please use .json config or install PyYAML.")
            return yaml.safe_load(f)
        else:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="DX4MGR Sim Ver6 Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--seed", type=int, default=None, help="Base seed")
    parser.add_argument("--trials", type=int, default=None, help="Number of trials")
    parser.add_argument("--output", type=str, default="output/final_result.png", help="Output plot path")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 引数による上書き
    if args.seed is not None:
        config["base_seed"] = args.seed
    if args.trials is not None:
        config["n_trials"] = args.trials
    
    base_seed = config.pop("base_seed", 42)
    n_trials = config.pop("n_trials", 100)
    
    print(f"Starting Monte Carlo simulation: {n_trials} trials, seed={base_seed}")
    
    results = run_monte_carlo(
        setup_standard_flow, 
        n_trials=n_trials, 
        base_seed=base_seed, 
        use_parallel=True, 
        **config
    )
    
    # 全試行の平均的な指標を出すための集計
    all_completed_jobs = []
    all_nodes_stats = []
    
    for engine in results:
        all_completed_jobs.extend(engine.results["completed_jobs"])
        # ノード統計は最後の試行のものを代表とするか、集計するか
        # ここでは簡易的に全試行の completed_jobs から全体指標を出す
    
    # calculate_metrics は単一試行用なので、少し調整が必要だが、
    # 簡易的に最初の1件のノード構成を使って全体指標を出す
    sample_nodes_stats = [node.stats() for node in results[0].nodes.values()]
    
    # 期間を全試行分に合算
    total_days = config.get("days", 365) * n_trials
    metrics = calculate_metrics(all_completed_jobs, sample_nodes_stats, total_days)
    
    print("\n--- Aggregated Results ---")
    print(f"Total Completed Jobs: {metrics['summary']['completed_count']}")
    print(f"Throughput: {metrics['summary']['throughput']:.4f} jobs/day")
    print(f"Lead Time P50: {metrics['summary']['lead_time_p50']:.2f} days")
    print(f"Lead Time P90: {metrics['summary']['lead_time_p90']:.2f} days")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_all_results(metrics, output_path=args.output)
    
    # JSONレポートも出す
    report_path = args.output.replace('.png', '.json')
    with open(report_path, 'w') as f:
        # CCDFなどの巨大なデータは除く
        report_summary = {k: v for k, v in metrics.items() if k != 'ccdf' and not k.startswith('raw_')}
        json.dump(report_summary, f, indent=4)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    # パス設定（自パッケージを認識させる）
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()

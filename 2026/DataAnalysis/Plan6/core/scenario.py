import json
try:
    import yaml
except ImportError:
    yaml = None
import os
import pandas as pd
from .simulation import Simulation

class ScenarioManager:
    def __init__(self, base_config_path=None):
        self.base_params = {}
        if base_config_path:
            self.base_params = self.load_config(base_config_path)
    
    def load_config(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            if path.endswith('.json'):
                return json.load(f)
            # YAML fallback to JSON for now if yaml is not available
            # or use a simple parser if needed
            content = f.read()
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                # Simple YAML to Dict parser for basic structures (optional)
                if path.endswith('.yaml') or path.endswith('.yml'):
                   print("Warning: PyYAML not installed. Using simple JSON parser for YAML file.")
                return json.loads(content)
        return {}

    def run_scenario(self, scenario_params):
        params = self.base_params.copy()
        params.update(scenario_params)
        
        sim = Simulation()
        sim.setup_with_params(params)
        steps = params.get('steps', 1000)
        kpis = sim.run(steps=steps)
        
        # 追加情報の付与
        kpis['scenario_id'] = params.get('scenario_id', 'unknown')
        
        # メトリクスの計算 (analysis/metrics.pyを使用)
        try:
            from analysis.metrics import calculate_metrics
            nodes_stats = []
            for node_id, node in sim.engine.nodes.items():
                if hasattr(node, 'get_stats'):
                    nodes_stats.append(node.get_stats())
                else:
                    nodes_stats.append({"node_id": node_id, "avg_wait_time": 0})
            
            metrics = calculate_metrics(
                sim.engine.results["completed_jobs"],
                nodes_stats,
                sim.engine.now,
                sim.engine.results["wip_history"]
            )
            # 技術履歴も追加
            metrics['tech_history'] = sim.tech_history
        except ImportError:
            metrics = {"kpis": kpis}
            
        return kpis, sim, metrics

    def run_from_csv(self, csv_path):
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            return []
        
        df = pd.read_csv(csv_path)
        results = []
        for _, row in df.iterrows():
            scenario_params = row.to_dict()
            kpis, sim, metrics = self.run_scenario(scenario_params)
            results.append({"kpis": kpis, "metrics": metrics})
        
        return results

    def save_results(self, results, output_path):
        # results list of {"kpis": ..., "metrics": ...}
        kpi_list = [r["kpis"] for r in results]
        df = pd.DataFrame(kpi_list)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # 一括比較グラフの生成
        try:
            from analysis.viz import plot_kpi_comparison
            plot_kpi_comparison(kpi_list, output_path.replace(".csv", "_comparison.png"))
        except ImportError:
            pass

    def visualize_single(self, sim, metrics, scenario_id="result"):
        try:
            from analysis.viz import plot_all_results, plot_tech_history, plot_wip_heatmap
            os.makedirs("reports/viz", exist_ok=True)
            
            plot_all_results(metrics, f"reports/viz/{scenario_id}_summary.png")
            plot_tech_history(metrics.get('tech_history', []), f"reports/viz/{scenario_id}_tech.png")
            plot_wip_heatmap(metrics['wip']['history'], f"reports/viz/{scenario_id}_wip.png")
        except Exception as e:
            print(f"Visualization failed: {e}")

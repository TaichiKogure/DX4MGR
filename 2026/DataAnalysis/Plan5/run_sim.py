from core.simulation import Simulation
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("=== Simulation Ver1: CSV-driven Scenario Execution ===")
    
    # パス設定
    config_path = 'configs/simulation_params.csv'
    output_dir = 'reports'
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return

    # パラメータ読み込み
    df_params = pd.read_csv(config_path)
    print(f"Loaded {len(df_params)} scenarios from {config_path}")
    
    all_results = []
    
    for index, row in df_params.iterrows():
        scenario_id = row['scenario_id']
        steps = int(row['steps'])
        print(f"Running scenario: {scenario_id} ...")
        
        # パラメータの辞書化
        params = row.to_dict()
        
        # シミュレーション実行
        sim = Simulation()
        sim.setup_with_params(params)
        
        # 統合組織（Integrated）の場合は初期品質を上げる等の特別処理をここに加えることも可能
        if scenario_id == 'Integrated':
            for t in sim.tech_items:
                t.improve_quality(0.3)
                
        kpis = sim.run(steps=steps)
        kpis['scenario_id'] = scenario_id
        all_results.append(kpis)
        
    # 結果の集計
    df_results = pd.DataFrame(all_results)
    
    # CSV保存
    results_csv = os.path.join(output_dir, 'simulation_results.csv')
    df_results.to_csv(results_csv, index=False)
    print(f"Results saved to {results_csv}")
    
    # 可視化
    plot_kpis(df_results, output_dir)
    
    # レポート生成
    generate_report(df_results, output_dir)

def plot_kpis(df, output_dir):
    metrics = ['total_gain', 'operational_failures', 'rework_count', 'integration_days']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        df.plot(kind='bar', x='scenario_id', y=metric, ax=axes[i], legend=False)
        axes[i].set_title(metric.replace('_', ' ').capitalize())
        axes[i].set_ylabel('Value')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'kpi_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def generate_report(df, output_dir):
    report_path = os.path.join(output_dir, 'simulation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Simulation Execution Report\n\n")
        f.write("## Scenario Comparison Summary\n\n")
        # tabulateがない場合の簡易Markdown表形式
        f.write("| " + " | ".join(df.columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(df.columns)) + " |\n")
        for _, row in df.iterrows():
            f.write("| " + " | ".join([str(val) for val in row]) + " |\n")
        
        f.write("\n\n## Visualization\n\n")
        f.write("![KPI Comparison](kpi_comparison.png)\n")
        
    print(f"Markdown report generated at {report_path}")

if __name__ == "__main__":
    main()

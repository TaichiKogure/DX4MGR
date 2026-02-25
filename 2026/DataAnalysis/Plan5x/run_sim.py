import pandas as pd
import matplotlib.pyplot as plt
import os
from core.simulation import Simulation

def run_scenarios():
    # パラメータの読み込み
    params_df = pd.read_csv('configs/simulation_params.csv')
    all_results = []

    for _, row in params_df.iterrows():
        params = row.to_dict()
        params['past_logs_file'] = 'configs/past_logs.csv'
        
        sim = Simulation()
        sim.setup_with_params(params)
        
        # シミュレーション実行
        kpis = sim.run(steps=int(params['steps']))
        kpis['scenario_id'] = params['scenario_id']
        all_results.append(kpis)
        
        print(f"Scenario {params['scenario_id']} finished.")
        print(f"  Total Gain: {kpis['total_gain']:.2f}")
        print(f"  Integration Days: {kpis['integration_days']:.1f}")
        print(f"  Market Complaints: {kpis['market_complaints']}")

    # 結果の保存
    res_df = pd.DataFrame(all_results)
    res_df.to_csv('reports/simulation_results.csv', index=False)
    
    # 可視化
    plot_kpis(res_df)
    
    # レポート作成
    generate_report(res_df)

def plot_kpis(df):
    metrics = ['total_gain', 'integration_days', 'technical_failures', 'operational_failures', 'market_complaints']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15))
    
    for i, metric in enumerate(metrics):
        df.plot(kind='bar', x='scenario_id', y=metric, ax=axes[i], legend=False)
        axes[i].set_title(metric)
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('reports/kpi_comparison.png')
    plt.close()

def generate_report(df):
    report_path = 'reports/simulation_report.md'
    with open(report_path, 'w') as f:
        f.write("# Plan 5x: Integrated Simulation Report\n\n")
        f.write("## Scenario Comparison Summary\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n## Analysis Notes\n\n")
        f.write("- **Strategic vs Random**: Strategic selection focuses on high-uncertainty items, potentially reducing integration days and market complaints.\n")
        f.write("- **Digital Twin Effect**: Calibrating from logs reflects real-world team performance, showing a more realistic (and often tougher) baseline.\n")
        f.write("- **Market Feedback**: High residual uncertainty after DR leads to market complaints, highlighting the importance of the TRL gap.\n")
        f.write("\n## Visualization\n\n")
        f.write("![KPI Comparison](kpi_comparison.png)\n")

if __name__ == "__main__":
    if not os.path.exists('reports'):
        os.makedirs('reports')
    run_scenarios()

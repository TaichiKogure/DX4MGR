import os
import numpy as np
import matplotlib.pyplot as plt
from simulator_v3 import simulate_flow_v3, run_monte_carlo
import visualizer_v3 as viz

# 出力ディレクトリ作成
OUT_DIR = "2026/RDIssues/Ver3/output"
os.makedirs(OUT_DIR, exist_ok=True)

def step1_capacity_breakdown():
    print("Step 1: 実効処理能力の内訳を可視化中...")
    params = {
        "service_vp_per_day": 10.0,
        "skill_factor": 1.2,
        "skill_at_dependency_resolution": 0.5,
        "dependency_level": 0.4,
        "coordination_penalty": 2.0
    }
    
    # 計算
    base = params["service_vp_per_day"]
    with_skill = base * params["skill_factor"]
    skill_inc = with_skill - base
    
    # 依存によるロス
    adj_dep = params["dependency_level"] / (1.0 + params["skill_at_dependency_resolution"])
    coord_factor = 1.0 / (1.0 + params["coordination_penalty"] * adj_dep)
    final = with_skill * coord_factor
    dep_dec = with_skill - final
    
    viz.plot_waterfall(base, skill_inc, dep_dec, final)
    plt.savefig(f"{OUT_DIR}/step1_waterfall.png")
    plt.close()

def step2_sensitivity_analysis():
    print("Step 2: 改善の優先順位 (感度分析) を可視化中...")
    base_params = {
        "arrival_vp_per_day": 5.0,
        "service_vp_per_day": 6.0,
        "p_rework": 0.2,
        "skill_factor": 1.0,
        "dependency_level": 0.3,
        "coordination_penalty": 1.0
    }
    
    def get_tp(p):
        res = simulate_flow_v3(**p, seed=42)
        return res["summary"]["throughput"]
    
    base_tp = get_tp(base_params)
    
    labels = ["skill_factor", "dependency_level", "p_rework", "service_rate"]
    param_keys = ["skill_factor", "dependency_level", "p_rework", "service_vp_per_day"]
    changes = []
    
    for key in param_keys:
        p_up = base_params.copy()
        if key == "dependency_level" or key == "p_rework":
            p_up[key] *= 0.8 # 改善方向
        else:
            p_up[key] *= 1.2 # 改善方向
            
        tp_up = get_tp(p_up)
        changes.append(tp_up - base_tp)
        
    viz.plot_tornado(labels, changes)
    plt.savefig(f"{OUT_DIR}/step2_tornado.png")
    plt.close()

def step3_fear_of_congestion():
    print("Step 3: 詰まりの怖さ (上層部向け) を可視化中...")
    
    # ケース設定
    cases = {
        "現状 (High Risk)": {
            "arrival_vp_per_day": 5.8, # 稼働率ギリギリ
            "service_vp_per_day": 6.0,
            "p_rework": 0.1
        },
        "改善後 (Low Risk)": {
            "arrival_vp_per_day": 5.8,
            "service_vp_per_day": 8.0, # 能力向上
            "p_rework": 0.05           # 品質向上
        }
    }
    
    wait_times_dict = {}
    wip_logs_dict = {}
    
    for label, params in cases.items():
        # モンテカルロ実行
        trials = run_monte_carlo(n_trials=50, **params)
        
        all_waits = []
        for r in trials:
            all_waits.extend(r["logs"]["wait_times"])
        
        wait_times_dict[label] = all_waits
        
        # WIPは最初の1回分を代表として表示
        wip_logs_dict[label] = trials[0]["logs"]["wip_history"]

    # 1. 待ち時間分布
    viz.plot_wait_time_distribution(wait_times_dict)
    plt.savefig(f"{OUT_DIR}/step3_violin.png")
    plt.close()
    
    # 2. CCDF
    viz.plot_ccdf(wait_times_dict)
    plt.savefig(f"{OUT_DIR}/step3_ccdf.png")
    plt.close()
    
    # 3. WIP推移
    viz.plot_wip_timeseries(wip_logs_dict)
    plt.savefig(f"{OUT_DIR}/step3_wip.png")
    plt.close()

def step4_organization_walls():
    print("Step 4: 組織構造の壁 (依存関係) を可視化中...")
    import networkx as nx
    
    G = nx.DiGraph()
    # 部署間の依存関係をダミーで作成
    edges = [
        ("開発A", "材料評価", 5),
        ("開発B", "材料評価", 3),
        ("材料評価", "共通分析", 8),
        ("共通分析", "承認ゲート", 2),
        ("開発A", "承認ゲート", 1)
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            width=weights, edge_color='gray', arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("組織間依存ネットワーク (エッジの太さ＝依存の重み)")
    plt.savefig(f"{OUT_DIR}/step4_network.png")
    plt.close()

def step5_executive_dashboard():
    print("Step 5: 上層部向けエグゼクティブ・ダッシュボードを作成中...")
    
    # ケース設定 (Step3と同様だが、より極端な比較)
    cases = {
        "現状 (人海戦術)": {
            "arrival_vp_per_day": 5.5,
            "service_vp_per_day": 6.0,
            "p_rework": 0.2,
            "dependency_level": 0.5,
            "coordination_penalty": 2.0
        },
        "改善案 (構造改革)": {
            "arrival_vp_per_day": 5.5,
            "service_vp_per_day": 8.0, # スキル向上
            "p_rework": 0.05,          # 品質向上
            "dependency_level": 0.1,   # 依存削減
            "coordination_penalty": 1.0
        }
    }
    
    wait_times_dict = {}
    for label, params in cases.items():
        trials = run_monte_carlo(n_trials=100, **params)
        all_waits = []
        for r in trials:
            all_waits.extend(r["logs"]["wait_times"])
        wait_times_dict[label] = all_waits

    viz.plot_executive_summary(wait_times_dict, threshold=14)
    plt.savefig(f"{OUT_DIR}/step5_dashboard.png")
    plt.close()

if __name__ == "__main__":
    step1_capacity_breakdown()
    step2_sensitivity_analysis()
    step3_fear_of_congestion()
    step4_organization_walls()
    step5_executive_dashboard()
    print(f"\n全ての可視化が完了しました。出力先: {OUT_DIR}")

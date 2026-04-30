import argparse
import sys
import os
import shutil
import datetime
import yaml
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dataclasses
# Add src directory to sys.path to support direct execution
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from iondynamics.config import load_config
from iondynamics.simulate import run_simulation
from iondynamics.particles import generate_particles, map_concentration_to_particles
from iondynamics.breakdown import compute_breakdown, plot_breakdown_stack
from iondynamics.visualize import plot_voltage_time, plot_ce_time_series, plot_kpi_time_series
from iondynamics.animate import animate_particles
from iondynamics.sweep import run_sweep, plot_sensitivity_heatmap
from iondynamics.postprocess import compute_kpis, save_thickness_data
from iondynamics.microstructure import (
    MicrostructureConfig,
    MicrostructureGenerator,
    MicrostructureAnalyzer,
    save_microstructure_results
)
from iondynamics.solver2d import (
    TransportSolverConfig,
    TransportSolver2D,
    save_transport_results
)
from iondynamics.visualize2d import visualize_transport_results
import copy

def main():
    parser = argparse.ArgumentParser(description="IondynamicsSim CLI")
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)

    # animate
    anim_parser = subparsers.add_parser("animate")
    anim_parser.add_argument("--config", required=True)
    anim_parser.add_argument("--out", default="outputs/animations/demo.mp4")

    # sweep
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--config", required=True)
    sweep_parser.add_argument("--spec", required=True)

    # compare
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--config", required=True)
    compare_parser.add_argument("--axis", choices=["thickness", "porosity", "particle"], required=True)
    compare_parser.add_argument("--values", type=float, nargs='+', required=True)

    # particles
    part_parser = subparsers.add_parser("particles")
    part_parser.add_argument("--mode", default="random")
    part_parser.add_argument("--n", type=int, default=300)
    part_parser.add_argument("--preview", action="store_true")

    # microstructure
    micro_parser = subparsers.add_parser("microstructure")
    micro_subparsers = micro_parser.add_subparsers(dest="subcommand")
    
    gen_parser = micro_subparsers.add_parser("generate")
    gen_parser.add_argument("--config", required=True)
    gen_parser.add_argument("--out", default="outputs/microstructure")
    
    preview_parser = micro_subparsers.add_parser("preview")
    preview_parser.add_argument("--config", required=True)
    
    compare_micro_parser = micro_subparsers.add_parser("compare")
    compare_micro_parser.add_argument("--configs", nargs='+', required=True)
    compare_micro_parser.add_argument("--out", default="outputs/microstructure_compare")
    
    stats_parser = micro_subparsers.add_parser("stats")
    stats_parser.add_argument("--dir", required=True)

    # transport2d
    trans_parser = subparsers.add_parser("transport2d")
    trans_subparsers = trans_parser.add_subparsers(dest="subcommand")
    
    tr_run_parser = trans_subparsers.add_parser("run")
    tr_run_parser.add_argument("--config", required=True, help="SimConfig path (includes microstructure section)")
    tr_run_parser.add_argument("--out", default="outputs/transport2d")
    
    tr_comp_parser = trans_subparsers.add_parser("compare")
    tr_comp_parser.add_argument("--config", required=True)
    tr_comp_parser.add_argument("--axis", choices=["thickness", "porosity", "particle_size"], required=True)
    tr_comp_parser.add_argument("--values", type=float, nargs='+', required=True)
    tr_comp_parser.add_argument("--out", default="outputs/transport2d_compare")

    args = parser.parse_args()

    if args.command == "run":
        run_sim(args.config)
    elif args.command == "animate":
        run_anim(args.config, args.out)
    elif args.command == "sweep":
        run_param_sweep(args.config, args.spec)
    elif args.command == "compare":
        run_comparison(args.config, args.axis, args.values)
    elif args.command == "particles":
        run_part_preview(args.mode, args.n, args.preview)
    elif args.command == "microstructure":
        if args.subcommand == "generate":
            run_micro_generate(args.config, args.out)
        elif args.subcommand == "preview":
            run_micro_preview(args.config)
        elif args.subcommand == "compare":
            run_micro_compare(args.configs, args.out)
        elif args.subcommand == "stats":
            run_micro_stats(args.dir)
    elif args.command == "transport2d":
        if args.subcommand == "run":
            run_transport2d_single(args.config, args.out)
        elif args.subcommand == "compare":
            run_transport2d_compare(args.config, args.axis, args.values, args.out)

def create_run_dir(case_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs/runs", f"{timestamp}_{case_name}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def run_sim(config_path):
    cfg = load_config(config_path)
    run_dir = execute_simulation(cfg)
    # 元のファイルをコピー（互換性のため）
    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
    print(f"Results saved to {run_dir}")

def execute_simulation(cfg):
    run_dir = create_run_dir(cfg.meta.case_name)
    
    # configの保存 (GUI等での変更を反映するため)
    with open(os.path.join(run_dir, "config.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(dataclasses.asdict(cfg), f, allow_unicode=True)

    result = run_simulation(cfg)
    breakdown = compute_breakdown(result)
    kpis = compute_kpis(result)

    # 図の保存
    fig, ax = plt.subplots()
    plot_voltage_time(result, ax=ax)
    fig.savefig(os.path.join(run_dir, "voltage_time.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_ce_time_series(result, ax=ax)
    fig.savefig(os.path.join(run_dir, "thickness_ce_profiles.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_kpi_time_series(result, kpis, ax=ax)
    fig.savefig(os.path.join(run_dir, "kpi_time_series.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_breakdown_stack(breakdown, result.time, ax=ax)
    fig.savefig(os.path.join(run_dir, "breakdown_stack.png"))
    plt.close(fig)

    # CSV
    df = pd.DataFrame({
        "time": result.time,
        "voltage": result.voltage,
        "current": result.current,
        "delta_ce": kpis.delta_ce,
        "max_grad_ce": kpis.max_grad_ce,
        "resistance_index": kpis.resistance_index,
        "ce_ratio": kpis.ce_ratio
    })
    for k, v in breakdown.items():
        df[f"breakdown_{k}"] = v
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    
    # 厚み方向データの詳細保存
    save_thickness_data(result, run_dir)
    
    # サマリCSV (時間平均 + 新規KPI)
    summary = {
        "case_name": cfg.meta.case_name,
        "capacity_Ah": np.trapezoid(result.current, result.time) / 3600,
        "avg_voltage": np.mean(result.voltage)
    }
    for k, v in breakdown.items():
        if k != "total":
            summary[f"avg_{k}_V"] = np.mean(v)
    
    # 新規KPIの追加
    summary.update(kpis.final_values)
    
    pd.DataFrame([summary]).to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    
    return run_dir

def run_anim(config_path, out_path):
    cfg = load_config(config_path)
    # デフォルトのパスが指定されている場合は、runsディレクトリ内に保存するように変更
    if out_path == "outputs/animations/demo.mp4":
        run_dir = create_run_dir(f"{cfg.meta.case_name}_anim")
        out_path = os.path.join(run_dir, "animation.mp4")
    
    execute_animation(cfg, out_path)
    print(f"Animation saved to {os.path.abspath(out_path)}")

def execute_animation(cfg, out_path):
    # 出力先ディレクトリの作成を保証
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    result = run_simulation(cfg)
    
    # 粒子配置
    particles = generate_particles(cfg.particles.mode, cfg.particles.count, 
                                   cfg.particles.bbox_um, cfg.electrode.particle_radius_um,
                                   cfg.particles.seed, cfg.particles.bimodal)
    
    # 粒子表面濃度の補間 (正極内)
    particle_c = map_concentration_to_particles(particles, result.x, result.c_s_surf, cfg.electrode.thickness_um)
    
    animate_particles(result, particles, particle_c, out_path)

def run_param_sweep(config_path, spec_path):
    cfg = load_config(config_path)
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    df = run_sweep(cfg, spec)
    
    run_dir = create_run_dir("sweep")
    df.to_csv(os.path.join(run_dir, "sweep_results.csv"), index=False)
    
    # ヒートマップ出力 (最初の2軸)
    keys = list(spec.keys())
    if len(keys) >= 2:
        plot_sensitivity_heatmap(df, keys[0], keys[1], "capacity_Ah", os.path.join(run_dir, "sensitivity_capacity.png"))
    
    print(f"Sweep results saved to {run_dir}")

def run_comparison(config_path, axis, values):
    base_cfg = load_config(config_path)
    results = []
    case_labels = []
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir = os.path.join("outputs/runs", f"{timestamp}_compare_{axis}")
    os.makedirs(comp_dir, exist_ok=True)
    
    for val in values:
        cfg = copy.deepcopy(base_cfg)
        if axis == "thickness":
            cfg.electrode.thickness_um = val
            label = f"thickness_{int(val)}um"
        elif axis == "porosity":
            cfg.electrode.porosity = val
            label = f"porosity_{val:.2f}"
        elif axis == "particle":
            cfg.electrode.particle_radius_um = val
            label = f"particle_{val:.1f}um"
        
        cfg.meta.case_name = label
        case_labels.append(label)
        
        print(f"Running case: {label}...")
        res = run_simulation(cfg)
        kpis = compute_kpis(res)
        results.append((res, kpis))
        
    # 比較レポートの生成
    generate_comparison_report(results, case_labels, comp_dir, axis)
    print(f"Comparison report saved to {comp_dir}")

def generate_comparison_report(results_list, labels, out_dir, axis):
    # 1. 電圧比較
    fig, ax = plt.subplots()
    for (res, _), label in zip(results_list, labels):
        ax.plot(res.time, res.voltage, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "compare_voltage.png"))
    plt.close(fig)
    
    # 2. 濃度分布比較 (最終時刻)
    fig, ax = plt.subplots()
    for (res, _), label in zip(results_list, labels):
        ax.plot(res.x * 1e6, res.c_e[:, -1], label=label)
    ax.set_xlabel("x [um] (0: separator side)")
    ax.set_ylabel("Electrolyte conc [mol/m3]")
    ax.set_title("Concentration Profile at End of Discharge")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "compare_ce_profile_final.png"))
    plt.close(fig)
    
    # 3. KPI比較 (Δc_e)
    fig, ax = plt.subplots()
    for (_, kpis), label in zip(results_list, labels):
        ax.plot(kpis.delta_ce, label=label)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Δc_e [mol/m3]")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "compare_delta_ce.png"))
    plt.close(fig)
    
    # 4. サマリCSV
    summary_data = []
    for (res, kpis), label in zip(results_list, labels):
        row = {"label": label, "axis_value": label.split('_')[-1]}
        row.update(kpis.final_values)
        summary_data.append(row)
    
    pd.DataFrame(summary_data).to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    
    # 5. Markdownサマリ
    with open(os.path.join(out_dir, "report.md"), 'w', encoding='utf-8') as f:
        f.write(f"# Comparison Report: {axis}\n\n")
        f.write("## KPI Summary\n\n")
        try:
            f.write(pd.DataFrame(summary_data).to_markdown(index=False))
        except ImportError:
            # tabulateがない場合のフォールバック
            f.write(pd.DataFrame(summary_data).to_string(index=False))
        f.write("\n\n## Figures\n\n")
        f.write("![Voltage](./compare_voltage.png)\n")
        f.write("![Concentration Profile](./compare_ce_profile_final.png)\n")
        f.write("![Delta Ce](./compare_delta_ce.png)\n")

def run_part_preview(mode, n, preview):
    bbox = [200, 80, 0]
    radius = 5.0
    particles = generate_particles(mode, n, bbox, radius)
    print(f"Generated {len(particles)} particles in mode {mode}")
    if preview:
        plt.scatter(particles[:, 0], particles[:, 1], s=particles[:, 3]**2)
        plt.gca().set_aspect('equal')
        plt.title(f"Particle Preview: {mode}")
        plt.show()

def run_micro_generate(config_path, out_dir):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    m_cfg = MicrostructureConfig.from_dict(data.get('microstructure', data))
        
    generator = MicrostructureGenerator(m_cfg)
    phase_map = generator.generate()
    
    analyzer = MicrostructureAnalyzer(phase_map)
    transport_map = analyzer.analyze()
    stats = analyzer.get_summary_statistics(transport_map, m_cfg)
    
    save_microstructure_results(out_dir, m_cfg, phase_map, transport_map, stats, generator.particles)
    print(f"Microstructure generated and saved to {out_dir}")

def run_micro_preview(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    m_cfg = MicrostructureConfig.from_dict(data.get('microstructure', data))
        
    generator = MicrostructureGenerator(m_cfg)
    phase_map = generator.generate()
    analyzer = MicrostructureAnalyzer(phase_map)
    transport_map = analyzer.analyze()
    
    out_dir = "outputs/microstructure_preview"
    os.makedirs(out_dir, exist_ok=True)
    preview_path = os.path.join(out_dir, "preview.png")
    from iondynamics.microstructure import visualize_microstructure
    visualize_microstructure(phase_map, transport_map, save_path=preview_path)
    print(f"Preview saved to {preview_path}")

def run_micro_compare(config_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    all_stats = []
    
    for i, path in enumerate(config_paths):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        m_cfg = MicrostructureConfig.from_dict(data.get('microstructure', data))
        if m_cfg.case_name == "default":
            m_cfg.case_name = f"case_{i}"
            
        generator = MicrostructureGenerator(m_cfg)
        phase_map = generator.generate()
        analyzer = MicrostructureAnalyzer(phase_map)
        transport_map = analyzer.analyze()
        stats = analyzer.get_summary_statistics(transport_map, m_cfg)
        stats['case_name'] = m_cfg.case_name
        all_stats.append(stats)
        
        case_dir = os.path.join(out_dir, m_cfg.case_name)
        save_microstructure_results(case_dir, m_cfg, phase_map, transport_map, stats, generator.particles)

    pd.DataFrame(all_stats).to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print(f"Comparison completed. Results in {out_dir}")

def run_micro_stats(dir_path):
    import csv
    stats_path = os.path.join(dir_path, "stats.csv")
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                print("Microstructure Statistics:")
                for k, v in row.items():
                    print(f"  {k}: {v}")
    else:
        print(f"Error: No stats.csv found in {dir_path}")

def run_transport2d_single(config_path, out_dir=None):
    cfg = load_config(config_path)
    
    # MicrostructureConfigの抽出
    m_dict = cfg.microstructure
    if not m_dict:
        # デフォルト値を使用
        m_cfg = MicrostructureConfig(
            width_um=100, 
            thickness_um=cfg.electrode.thickness_um
        )
    else:
        m_cfg = MicrostructureConfig.from_dict(m_dict)
    
    # 1. 構造生成
    gen = MicrostructureGenerator(m_cfg)
    phase_map = gen.generate()
    
    # 2. 輸送ソルバ設定
    # SimConfigから輸送に関わるパラメータを反映させる
    t_cfg = TransportSolverConfig(
        case_name=cfg.meta.case_name,
        d_electrolyte_bulk=cfg.resistances.solid_diffusivity_m2_s, # 1D設定を流用(本来は電解液用が必要)
        k_electrolyte_bulk=cfg.resistances.ionic_conductivity_S_m,
        applied_current_density=cfg.operation.c_rate * 20.0, # 簡易換算
    )
    
    if out_dir is None or out_dir == "outputs/transport2d":
        out_dir = create_run_dir(f"{cfg.meta.case_name}_transport2d")

    # 3. 実行
    solver = TransportSolver2D(t_cfg, phase_map.grid)
    prop_map = solver.prepare_property_map(phase_map)
    
    print("Solving steady potential field...")
    res_steady = solver.solve_steady_potential()
    
    print("Solving transient concentration field...")
    res_transient = solver.solve_transient_concentration()
    
    # 4. 保存と可視化
    save_transport_results(out_dir, t_cfg, res_steady, res_transient, prop_map)
    
    grid_info = {
        "width_um": m_cfg.width_um,
        "thickness_um": m_cfg.thickness_um
    }
    visualize_transport_results(res_steady, res_transient, prop_map, grid_info, out_dir, phase_map=phase_map)
    
    print(f"Transport 2D simulation finished. Results in {out_dir}")
    return out_dir

def run_transport2d_compare(config_path, axis, values, out_dir):
    base_cfg = load_config(config_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # 並列化の準備
    import multiprocessing as mp
    
    tasks = []
    for val in values:
        cfg = copy.deepcopy(base_cfg)
        if axis == "thickness":
            cfg.electrode.thickness_um = val
            if cfg.microstructure: cfg.microstructure["thickness_um"] = val
            label = f"thickness_{int(val)}um"
        elif axis == "porosity":
            cfg.electrode.porosity = val
            if cfg.microstructure: cfg.microstructure["target_porosity"] = val
            label = f"porosity_{val:.2f}"
        elif axis == "particle_size":
            cfg.electrode.particle_radius_um = val
            # MicrostructureConfigにも反映が必要な場合はここで
            label = f"particle_{val:.1f}um"
            
        cfg.meta.case_name = label
        case_dir = os.path.join(out_dir, label)
        
        # 一時的なconfigファイルを作成
        temp_cfg_path = os.path.join(out_dir, f"temp_{label}.yaml")
        with open(temp_cfg_path, 'w', encoding='utf-8') as f:
            import dataclasses
            yaml.dump(dataclasses.asdict(cfg), f)
        
        tasks.append((temp_cfg_path, case_dir))

    print(f"Starting comparison on {axis} with {len(tasks)} cases (using multiprocessing)...")
    
    with mp.Pool(processes=min(len(tasks), mp.cpu_count())) as pool:
        pool.starmap(run_transport2d_single, tasks)
        
    # 後片付け
    for t_path, _ in tasks:
        if os.path.exists(t_path): os.remove(t_path)
        
    print(f"All comparison cases finished. Results in {out_dir}")

if __name__ == "__main__":
    main()

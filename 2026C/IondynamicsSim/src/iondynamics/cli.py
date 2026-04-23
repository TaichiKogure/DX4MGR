import argparse
import os
import shutil
import datetime
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from .config import load_config
from .simulate import run_simulation
from .particles import generate_particles, map_concentration_to_particles
from .breakdown import compute_breakdown, plot_breakdown_stack
from .visualize import plot_voltage_time
from .animate import animate_particles
from .sweep import run_sweep, plot_sensitivity_heatmap

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

    # particles
    part_parser = subparsers.add_parser("particles")
    part_parser.add_argument("--mode", default="random")
    part_parser.add_argument("--n", type=int, default=300)
    part_parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        run_sim(args.config)
    elif args.command == "animate":
        run_anim(args.config, args.out)
    elif args.command == "sweep":
        run_param_sweep(args.config, args.spec)
    elif args.command == "particles":
        run_part_preview(args.mode, args.n, args.preview)

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

    # 図の保存
    fig, ax = plt.subplots()
    plot_voltage_time(result, ax=ax)
    fig.savefig(os.path.join(run_dir, "voltage_time.png"))
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_breakdown_stack(breakdown, result.time, ax=ax)
    fig.savefig(os.path.join(run_dir, "breakdown_stack.png"))
    plt.close(fig)

    # CSV
    df = pd.DataFrame({
        "time": result.time,
        "voltage": result.voltage,
        "current": result.current
    })
    for k, v in breakdown.items():
        df[f"breakdown_{k}"] = v
    df.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    
    # サマリCSV (時間平均)
    summary = {
        "case_name": cfg.meta.case_name,
        "capacity_Ah": np.trapezoid(result.current, result.time) / 3600,
        "avg_voltage": np.mean(result.voltage)
    }
    for k, v in breakdown.items():
        if k != "total":
            summary[f"avg_{k}_V"] = np.mean(v)
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

if __name__ == "__main__":
    main()

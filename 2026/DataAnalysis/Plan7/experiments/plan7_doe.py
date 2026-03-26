"""
Plan7 パラメータ感度 DOE ランナー（バッチ実行 & レポート生成）

目的:
- 新しい Plan7 シミュレーションに対して、主要パラメータの感度をDOE（単純グリッド）で評価します。
- 各試行の中間データ（主要KPI・DR/待ち/リワーク要約など）を収集し、CSV/PNG/Markdown レポートとして保存します。

出力先（既定）:
- 2026/DataAnalysis/Plan7/reports/doe/<YYYYMMDD_HHMMSS>/ 以下に一式を保存
  - runs.csv: 各試行の生データ（1 行=1 試行）
  - summary_main_effects.csv: パラメータ毎の主効果（平均・p50/p90 等）
  - plots/*.png: 効果図（折れ線/箱ひげ/ヒートマップ等の一部）
  - DOE_Report.md: 日本語サマリーレポート

使い方（例）:
  python 2026/DataAnalysis/Plan7/experiments/plan7_doe.py \
      --steps 365 --seeds 3 --outdir 2026/DataAnalysis/Plan7/reports/doe

  # パラメータグリッドをJSONで指定（任意）
  python 2026/DataAnalysis/Plan7/experiments/plan7_doe.py \
      --config 2026/DataAnalysis/Plan7/experiments/sample_doe_config.json

備考:
- 依存: pandas, numpy, matplotlib（標準的な環境で利用可能）
- ffmpeg が無い環境でも実行可能（動画生成は行いませんが、静止画は出力します）
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 数字始まりディレクトリのためパッケージ import が困難。Plan7 直下を sys.path に通す。
import sys
from pathlib import Path as _PathForSys
_PLAN7_DIR = _PathForSys(__file__).resolve().parents[1]
if str(_PLAN7_DIR) not in sys.path:
    sys.path.insert(0, str(_PLAN7_DIR))

from core.simulation import Simulation  # type: ignore
from analysis.metrics import calculate_metrics  # type: ignore
from analysis import viz3d  # type: ignore


# --------------------------- 設定/型 ---------------------------

@dataclass
class DOEConfig:
    steps: int = 365
    seeds: int = 3
    base_params: Dict[str, Any] = None  # None の場合はランナー内のデフォルトで補完
    grid: Dict[str, List[Any]] = None   # None の場合は簡易デフォルトを使用


def _default_base_params() -> Dict[str, Any]:
    return {
        "scenario_id": "DOE_Base",
        "steps": 365,
        "sampling_interval": 1.0,
        # DR 関連（必要に応じて上書き）
        "dr1_period": 14,
        "dr2_period": 28,
        "dr3_period": 56,
        # サーバ数（工程の並列度）
        "res_n_servers": 5,
        "proto_n_servers": 3,
        "mass_n_servers": 2,
    }


def _default_grid() -> Dict[str, List[Any]]:
    return {
        # 代表的な3軸（計 3×2×2=12 点）
        "res_n_servers": [3, 5, 7],
        "proto_n_servers": [2, 3],
        "dr1_period": [7, 14],
    }


def load_config(path: str | None) -> DOEConfig:
    if not path:
        return DOEConfig(steps=365, seeds=3, base_params=_default_base_params(), grid=_default_grid())
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    steps = int(raw.get("steps", 365))
    seeds = int(raw.get("seeds", 3))
    base_params = raw.get("base_params") or _default_base_params()
    grid = raw.get("grid") or _default_grid()
    return DOEConfig(steps=steps, seeds=seeds, base_params=base_params, grid=grid)


# --------------------------- 実行本体 ---------------------------

def _ensure_outdir(root_out: str | None) -> Path:
    base = Path(root_out) if root_out else Path("2026/DataAnalysis/Plan7/reports/doe")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = base / ts
    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    (outdir / "videos").mkdir(parents=True, exist_ok=True)
    return outdir


def _param_product(grid: Dict[str, List[Any]]) -> List[Tuple[Tuple[str, Any], ...]]:
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    combos: List[Tuple[Tuple[str, Any], ...]] = []
    for prod in itertools.product(*values):
        combos.append(tuple(zip(keys, prod)))
    return combos


def _scenario_suffix(param_pairs: Tuple[Tuple[str, Any], ...]) -> str:
    parts = [f"{k}={v}" for k, v in param_pairs]
    return ",".join(parts)


def _collect_metrics(sim: Simulation) -> Dict[str, Any]:
    # nodes_stats
    nodes_stats = []
    for node_id, node in sim.engine.nodes.items():  # type: ignore[attr-defined]
        if hasattr(node, "get_stats"):
            nodes_stats.append(node.get_stats())

    m = calculate_metrics(
        sim.engine.results.get("completed_jobs", []),  # type: ignore[attr-defined]
        nodes_stats,
        sim.engine.now,  # type: ignore[attr-defined]
        sim.engine.results.get("wip_history", [])  # type: ignore[attr-defined]
    )
    return m


def _row_from_results(run_id: str, params: Dict[str, Any], kpis: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    s = (metrics or {}).get("summary", {})
    loss_time = (((metrics or {}).get("loss", {}) or {}).get("time", {}) or {})
    dr_primary = (((metrics or {}).get("dr_gate", {}) or {}).get("primary", {}) or {})

    row = {
        "run_id": run_id,
        "scenario_id": params.get("scenario_id"),
        # 主要出力（お好みで拡張）
        "completed_count": s.get("completed_count", 0),
        "throughput": s.get("throughput", 0.0),
        "lead_time_p50": s.get("lead_time_p50", 0.0),
        "lead_time_p90": s.get("lead_time_p90", 0.0),
        "avg_wip": s.get("avg_wip", 0.0),
        "loss_time_per_primary": loss_time.get("loss_per_primary", 0.0),
        # KPI（積み上げ）
        "total_experiments": kpis.get("total_experiments", 0),
        "technical_failures": kpis.get("technical_failures", 0),
        "operational_failures": kpis.get("operational_failures", 0),
        "rework_count_total": kpis.get("rework_count", 0),
        "dr_cost_total": kpis.get("dr_cost", 0.0),
    }

    # パラメータも列として持たせる
    for k, v in params.items():
        if isinstance(v, (int, float, str)):
            # steps のような制御系は除外
            if k not in ("steps", "seed", "sampling_interval"):
                row[k] = v
    return row


def _save_plots(df: pd.DataFrame, outdir: Path, grid_keys: List[str]) -> List[str]:
    saved: List[str] = []
    # 単因子の主効果図（折れ線）: throughput, lead_time_p50
    targets = ["throughput", "lead_time_p50"]
    for key in grid_keys:
        g = df.groupby(key)[targets].agg(["mean", "median", lambda x: np.percentile(x, 90)])
        g.columns = ["_".join([c for c in col if c]) for col in g.columns.values]
        ax = g.reset_index().plot(x=key, y=[c for c in g.columns], marker="o")
        ax.set_title(f"Main effects by {key}")
        fig = ax.get_figure()
        p = outdir / "plots" / f"main_effects_{key}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))

    # 2因子ヒートマップ（先頭2軸があれば）: throughput
    if len(grid_keys) >= 2:
        kx, ky = grid_keys[0], grid_keys[1]
        pivot = df.pivot_table(index=ky, columns=kx, values="throughput", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.set_title(f"Throughput heatmap: {ky} vs {kx}")
        fig.colorbar(im, ax=ax)
        p = outdir / "plots" / f"heatmap_{ky}_vs_{kx}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(p))
    return saved


def _save_wip_3d_examples(examples: List[Tuple[str, Dict[str, Any]]], outdir: Path) -> List[str]:
    paths: List[str] = []
    for label, res in examples:
        wip_hist = (((res or {}).get("metrics", {}) or {}).get("wip", {}) or {}).get("history", [])
        if not wip_hist:
            continue
        p = outdir / "plots" / f"wip3d_{label}.png"
        try:
            viz3d.save_wip_surface_png(wip_hist, str(p), title=f"WIP 3D Surface - {label}")
            paths.append(str(p))
        except Exception:
            # 3D 出力失敗は致命ではないので無視
            pass
    return paths


def _write_md_report(outdir: Path, cfg: DOEConfig, df: pd.DataFrame, plot_paths: List[str]) -> str:
    md = []
    md.append("# Plan7 パラメータ感度 DOE レポート")
    md.append("")
    md.append(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    md.append("## 実験設定")
    md.append(f"- steps: {cfg.steps}")
    md.append(f"- seeds/point: {cfg.seeds}")
    md.append(f"- 走査パラメータ: {', '.join(sorted((cfg.grid or {}).keys()))}")
    md.append("")
    md.append("## 中間データの要約（平均値）")
    summ = df[[
        "throughput", "lead_time_p50", "lead_time_p90", "avg_wip", "loss_time_per_primary"
    ]].mean().to_dict()
    for k, v in summ.items():
        md.append(f"- {k}: {v:.4f}")
    md.append("")
    md.append("## 代表的な図")
    for p in plot_paths:
        rel = os.path.relpath(p, outdir)
        md.append(f"![plot]({rel})")
    md.append("")
    md.append("## データファイル")
    md.append("- runs.csv: 各試行の生データ")
    md.append("- summary_main_effects.csv: パラメータ毎の主効果")
    path = outdir / "DOE_Report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return str(path)


def run_doe(cfg: DOEConfig, root_out: str | None = None) -> Path:
    outdir = _ensure_outdir(root_out)
    base_params = {**_default_base_params(), **(cfg.base_params or {})}

    combos = _param_product(cfg.grid or _default_grid())
    grid_keys = sorted((cfg.grid or _default_grid()).keys())

    rows: List[Dict[str, Any]] = []
    perrun_results: List[Dict[str, Any]] = []

    for combo in combos:
        # 一点のパラメータ辞書
        p_over = {k: v for k, v in combo}
        # 複数シードを回す
        for s in range(cfg.seeds):
            params = {**base_params, **p_over}
            params["seed"] = int(10_000 + s)
            # シナリオIDにパラメータを焼き込む
            params["scenario_id"] = f"DOE[{_scenario_suffix(combo)}]_seed{params['seed']}"

            sim = Simulation()
            sim.setup_with_params(params)
            kpis = sim.run(steps=cfg.steps)
            metrics = _collect_metrics(sim)

            run_id = f"{params['scenario_id']}"
            row = _row_from_results(run_id, params, kpis, metrics)
            rows.append(row)
            perrun_results.append({
                "run_id": run_id,
                "params": params,
                "kpis": kpis,
                "metrics": metrics,
            })

    # DataFrame 化と保存
    df = pd.DataFrame(rows)
    runs_csv = outdir / "runs.csv"
    df.to_csv(runs_csv, index=False, encoding="utf-8-sig")

    # 主効果の簡易要約
    effects: List[pd.DataFrame] = []
    for k in grid_keys:
        g = df.groupby(k)[["throughput", "lead_time_p50", "lead_time_p90", "avg_wip"]] \
             .agg(["mean", "median", lambda x: np.percentile(x, 90)])
        g.columns = ["_".join([c for c in cols if c]) for cols in g.columns.values]
        g["param"] = k
        effects.append(g.reset_index())
    eff_df = pd.concat(effects, ignore_index=True)
    eff_csv = outdir / "summary_main_effects.csv"
    eff_df.to_csv(eff_csv, index=False, encoding="utf-8-sig")

    # 図をいくつか保存
    plot_paths = _save_plots(df, outdir, grid_keys)

    # 代表例（最良/最悪）でWIP 3Dサーフェスを保存（静止画）
    try:
        best = max(perrun_results, key=lambda r: ((r.get("metrics") or {}).get("summary") or {}).get("throughput", 0))
        worst = min(perrun_results, key=lambda r: ((r.get("metrics") or {}).get("summary") or {}).get("throughput", 0))
        plot_paths += _save_wip_3d_examples([
            ("best_throughput", best), ("worst_throughput", worst)
        ], outdir)
    except Exception:
        pass

    # Markdown レポート
    _write_md_report(outdir, cfg, df, plot_paths)

    return outdir


def main():
    ap = argparse.ArgumentParser(description="Plan7 DOE Runner")
    ap.add_argument("--config", type=str, default=None, help="JSON で base_params/grid/steps/seeds を指定（任意）")
    ap.add_argument("--outdir", type=str, default="2026/DataAnalysis/Plan7/reports/doe", help="出力ベースディレクトリ")
    ap.add_argument("--steps", type=int, default=None, help="1 試行あたりの steps（config 未指定時の上書き）")
    ap.add_argument("--seeds", type=int, default=None, help="各グリッド点でのシード数（config 未指定時の上書き）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.steps = int(args.steps)
    if args.seeds is not None:
        cfg.seeds = int(args.seeds)

    outdir = run_doe(cfg, root_out=args.outdir)
    print(f"DOE 実行完了: {outdir}")


if __name__ == "__main__":
    main()

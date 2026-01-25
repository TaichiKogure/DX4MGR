import os
import sys
import argparse
from typing import Dict, Any, Iterable, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator import run_monte_carlo, latin_hypercube_sampling
import visualizer as viz

INT_PARAMS = {
    "days",
    "n_senior",
    "n_coordinator",
    "n_new",
    "bundle_size_small",
    "bundle_size_mid",
    "bundle_size_fin",
    "n_servers_mid",
    "n_servers_fin",
    "dr1_period",
    "dr2_period",
    "dr3_period",
    "dr_capacity",
    "dr1_capacity",
    "dr2_capacity",
    "dr3_capacity",
}

FLOAT_PARAMS = {
    "arrival_rate",
    "small_exp_duration",
    "mid_exp_duration",
    "fin_exp_duration",
    "rework_load_factor",
    "decay",
    "friction_alpha",
    "decision_latency_days",
    "dr1_cost_per_review",
    "dr2_cost_per_review",
    "dr3_cost_per_review",
    "dr2_rework_multiplier",
    "rework_beta_a",
    "rework_beta_b",
    "rework_task_type_mix",
    "conditional_prob_ratio",
    "sampling_interval",
    "dr_quality",
}

def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float(default)

def coerce_param_types(params: Dict[str, Any]) -> Dict[str, Any]:
    typed: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k in INT_PARAMS:
            typed[k] = int(v)
        elif k in FLOAT_PARAMS:
            typed[k] = float(v)
        else:
            typed[k] = v
    return typed

def load_base_params(scenarios_path: str, use_base_from_csv: bool = True) -> Tuple[Dict[str, Any], pd.DataFrame]:
    df_scenarios = pd.read_csv(scenarios_path)
    base_params: Dict[str, Any] = {}
    if use_base_from_csv and not df_scenarios.empty:
        base_row = df_scenarios.iloc[0].to_dict()
        for k, v in base_row.items():
            if k in ("scenario_name", "n_trials"):
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            base_params[k] = v
        base_params = coerce_param_types(base_params)
    return base_params, df_scenarios

def _normalize_param_ranges(param_ranges: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, info in param_ranges.items():
        if isinstance(info, dict):
            min_v = info.get("min")
            max_v = info.get("max")
            p_type = info.get("type")
        elif isinstance(info, (tuple, list)) and len(info) == 2:
            min_v, max_v = info
            p_type = None
        else:
            continue
        if min_v is None or max_v is None:
            continue
        if min_v > max_v:
            min_v, max_v = max_v, min_v
        if not p_type:
            p_type = "int" if name in INT_PARAMS or name == "dr_period" else "float"
        normalized[name] = {"min": float(min_v), "max": float(max_v), "type": p_type}
    return normalized

def _get_gate_value(gate_stats: List[Dict[str, Any]], node_id: str, key: str, default: float = 0.0) -> float:
    for stat in gate_stats:
        if stat.get("node_id") == node_id:
            return float(stat.get(key, default) or 0.0)
    return float(default)

def summarize_trials(trials: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trials:
        return {
            "throughput": 0.0,
            "p90_wait": 0.0,
            "avg_wip": 0.0,
            "avg_reworks": 0.0,
            "dr1_wait": 0.0,
            "dr2_wait": 0.0,
            "dr3_wait": 0.0,
            "dr1_cost": 0.0,
            "dr2_cost": 0.0,
            "dr3_cost": 0.0,
            "dr1_wip": 0.0,
            "dr2_wip": 0.0,
            "dr3_wip": 0.0,
            "total_review_cost": 0.0,
            "cost_per_completed": 0.0,
            "completed_count": 0.0,
            "dr2_wait_ratio": 0.0,
        }

    tps: List[float] = []
    p90s: List[float] = []
    wips: List[float] = []
    reworks: List[float] = []
    completed: List[float] = []
    dr_wait = {"DR1": [], "DR2": [], "DR3": []}
    dr_cost = {"DR1": [], "DR2": [], "DR3": []}
    dr_wip = {"DR1": [], "DR2": [], "DR3": []}
    total_costs: List[float] = []
    cost_per_completed: List[float] = []

    for t in trials:
        summary = t.get("summary", {}) or {}
        metrics = t.get("metrics", {}) or {}
        tps.append(float(summary.get("throughput", 0.0) or 0.0))
        p90s.append(float(summary.get("p90_wait", summary.get("lead_time_p90", 0.0)) or 0.0))
        wips.append(float(summary.get("avg_wip", 0.0) or 0.0))
        reworks.append(float(summary.get("avg_reworks", 0.0) or 0.0))
        completed_count = float(summary.get("approved_count", summary.get("completed_count", 0.0)) or 0.0)
        completed.append(completed_count)

        gate_stats = metrics.get("gate_stats", []) or []
        trial_costs: Dict[str, float] = {}
        for node in ["DR1", "DR2", "DR3"]:
            wait = _get_gate_value(gate_stats, node, "avg_wait_time", 0.0)
            cost = _get_gate_value(gate_stats, node, "total_cost", 0.0)
            dr_wait[node].append(wait)
            dr_cost[node].append(cost)
            trial_costs[node] = cost

        total_cost = sum(trial_costs.values())
        total_costs.append(total_cost)
        if completed_count > 0:
            cost_per_completed.append(total_cost / completed_count)

        wip_by_node = (metrics.get("wip", {}) or {}).get("avg_by_node", {}) or {}
        for node in ["DR1", "DR2", "DR3"]:
            dr_wip[node].append(float(wip_by_node.get(node, 0.0) or 0.0))

    dr1_wait = _safe_mean(dr_wait["DR1"])
    dr2_wait = _safe_mean(dr_wait["DR2"])
    dr3_wait = _safe_mean(dr_wait["DR3"])
    dr2_wait_ratio = dr2_wait / max(dr1_wait, 1e-9)

    return {
        "throughput": _safe_mean(tps),
        "p90_wait": _safe_mean(p90s),
        "avg_wip": _safe_mean(wips),
        "avg_reworks": _safe_mean(reworks),
        "dr1_wait": dr1_wait,
        "dr2_wait": dr2_wait,
        "dr3_wait": dr3_wait,
        "dr1_cost": _safe_mean(dr_cost["DR1"]),
        "dr2_cost": _safe_mean(dr_cost["DR2"]),
        "dr3_cost": _safe_mean(dr_cost["DR3"]),
        "dr1_wip": _safe_mean(dr_wip["DR1"]),
        "dr2_wip": _safe_mean(dr_wip["DR2"]),
        "dr3_wip": _safe_mean(dr_wip["DR3"]),
        "total_review_cost": _safe_mean(total_costs),
        "cost_per_completed": _safe_mean(cost_per_completed),
        "completed_count": _safe_mean(completed),
        "dr2_wait_ratio": dr2_wait_ratio,
    }


def _minmax(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    denom = max(max_v - min_v, 1e-12)
    return (series - min_v) / denom


def _score_doe(df: pd.DataFrame, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    if not weights:
        tp_n = _minmax(df["throughput"])
        p90_n = _minmax(df["p90_wait"])
        wip_n = _minmax(df["avg_wip"])
        rwk_n = _minmax(df["avg_reworks"])

        # Higher TP is better, lower P90/WIP/rework is better.
        df["score"] = tp_n - p90_n - 0.5 * wip_n - 0.5 * rwk_n
        return df

    score = np.zeros(len(df), dtype=float)
    for metric, weight in weights.items():
        if metric not in df.columns:
            continue
        score += weight * _minmax(df[metric])
    df["score"] = score
    return df


def _build_scenarios_from_doe(
    df_top: pd.DataFrame,
    template_df: pd.DataFrame,
    n_trials: int,
    param_names: List[str],
    scenario_prefix: str = "DOE_TOP",
) -> pd.DataFrame:
    if df_top.empty:
        return pd.DataFrame(columns=template_df.columns)

    base_row = template_df.iloc[0].copy()
    rows = []
    for i, row in enumerate(df_top.itertuples(), start=1):
        r = base_row.copy()
        if "scenario_name" in template_df.columns:
            r["scenario_name"] = f"{scenario_prefix}_{i:02d}"
        if "n_trials" in template_df.columns:
            r["n_trials"] = int(n_trials)
        for param in param_names:
            if not hasattr(row, param):
                continue
            value = getattr(row, param)
            if param == "dr_period":
                if "dr1_period" in template_df.columns:
                    r["dr1_period"] = int(round(value))
                if "dr2_period" in template_df.columns:
                    r["dr2_period"] = int(round(value))
                if "dr3_period" in template_df.columns:
                    r["dr3_period"] = int(round(value))
                continue
            if param in template_df.columns:
                if param in INT_PARAMS:
                    r[param] = int(round(value))
                else:
                    r[param] = float(value)
        rows.append(r)

    return pd.DataFrame(rows, columns=template_df.columns)

def _apply_constraints(df: pd.DataFrame, constraints: Optional[Dict[str, float]]) -> pd.DataFrame:
    if df.empty or not constraints:
        return df
    filtered = df.copy()
    for key, value in constraints.items():
        if key.endswith("_max"):
            metric = key[:-4]
            if metric in filtered.columns:
                filtered = filtered[filtered[metric] <= value]
        elif key.endswith("_min"):
            metric = key[:-4]
            if metric in filtered.columns:
                filtered = filtered[filtered[metric] >= value]
    return filtered

def _write_insights(out_dir: str, df: pd.DataFrame, param_names: List[str], metrics: List[str]):
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        corr = df[param_names].corrwith(df[metric]).dropna()
        for param, value in corr.items():
            rows.append({"metric": metric, "param": param, "corr": float(value)})
    if not rows:
        return
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "doe_param_correlations.csv"), index=False)

    lines = []
    for metric in metrics:
        subset = [r for r in rows if r["metric"] == metric]
        if not subset:
            continue
        subset.sort(key=lambda r: r["corr"], reverse=True)
        lines.append(f"[{metric}]")
        top_pos = subset[:5]
        top_neg = list(reversed(subset[-5:]))
        lines.append("  Top + correlation:")
        for r in top_pos:
            lines.append(f"    {r['param']}: {r['corr']:.3f}")
        lines.append("  Top - correlation:")
        for r in top_neg:
            lines.append(f"    {r['param']}: {r['corr']:.3f}")
        lines.append("")
    if lines:
        with open(os.path.join(out_dir, "doe_insights.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def run_doe_from_spec(
    spec: Dict[str, Any],
    scenarios_path: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    use_base = bool(spec.get("use_base_from_csv", True))
    base_params, df_scenarios = load_base_params(scenarios_path, use_base_from_csv=use_base)

    n_samples = int(spec.get("n_samples", 30))
    n_trials = int(spec.get("n_trials", 1))
    top_k = int(spec.get("top_k", 5))
    scenario_trials = int(spec.get("scenario_trials", 100))
    base_seed = int(spec.get("base_seed", 42))
    target_metric = spec.get("target_metric", "throughput")
    score_weights = spec.get("score_weights")
    constraints = spec.get("constraints")
    scenario_prefix = spec.get("scenario_prefix", "DOE_TOP")
    use_parallel = bool(spec.get("use_parallel", False))
    insight_metrics = spec.get("insight_metrics", [])

    param_ranges = _normalize_param_ranges(spec.get("param_ranges", {}))
    if not param_ranges:
        print("No DOE parameter ranges defined. Abort.")
        return
    param_names = list(param_ranges.keys())

    fixed_params = coerce_param_types(spec.get("fixed_params", {}))

    doe_samples = latin_hypercube_sampling(
        n_samples=n_samples,
        param_ranges={k: (v["min"], v["max"]) for k, v in param_ranges.items()},
    )
    doe_results = []

    for raw in doe_samples:
        sample: Dict[str, Any] = {}
        for name, value in raw.items():
            p_type = param_ranges[name]["type"]
            if p_type == "int":
                value = int(round(value))
            else:
                value = float(value)
            sample[name] = value

        params = dict(base_params)
        params.update(fixed_params)

        explicit_periods = {
            k for k in ["dr1_period", "dr2_period", "dr3_period"]
            if k in fixed_params or k in sample
        }

        params.update({k: v for k, v in sample.items() if k != "dr_period"})

        dr_period = None
        if "dr_period" in params:
            dr_period = params.pop("dr_period")
        if "dr_period" in sample:
            dr_period = sample.get("dr_period")

        if dr_period is not None:
            dr_val = int(round(dr_period))
            for k in ["dr1_period", "dr2_period", "dr3_period"]:
                if k not in explicit_periods:
                    params[k] = dr_val

        params = coerce_param_types(params)

        trials = run_monte_carlo(
            n_trials=n_trials,
            use_parallel=use_parallel,
            base_seed=base_seed,
            **params,
        )

        summary = summarize_trials(trials)
        doe_results.append({**sample, **summary})

    doe_df = pd.DataFrame(doe_results)
    if doe_df.empty:
        return

    plot_metric = target_metric if target_metric in doe_df.columns else "throughput"
    plot_cols = [c for c in param_names if c in doe_df.columns] + [plot_metric]
    viz.plot_doe_analysis(doe_df[plot_cols], target_col=plot_metric, title="Ver12 DOE Sensitivity (exploration)")
    plt.savefig(os.path.join(out_dir, "doe_analysis.png"))
    plt.close()

    doe_df.to_csv(os.path.join(out_dir, "doe_results.csv"), index=False)

    scored = _score_doe(doe_df, weights=score_weights).sort_values("score", ascending=False)
    scored.to_csv(os.path.join(out_dir, "doe_results_scored.csv"), index=False)

    filtered = _apply_constraints(scored, constraints)
    if filtered.empty:
        filtered = scored

    top_k = max(1, min(int(top_k), len(filtered)))
    df_top = filtered.head(top_k).copy()
    df_top.to_csv(os.path.join(out_dir, "doe_top_summary.csv"), index=False)

    df_suggest = _build_scenarios_from_doe(df_top, df_scenarios, scenario_trials, param_names, scenario_prefix)
    df_suggest.to_csv(os.path.join(out_dir, "scenarios_from_doe.csv"), index=False)

    if insight_metrics:
        _write_insights(out_dir, doe_df, param_names, insight_metrics)


def run_doe(
    scenarios_path: str,
    out_dir: str,
    n_samples: int,
    n_trials: int,
    top_k: int,
    scenario_trials: int,
):
    spec = {
        "use_base_from_csv": False,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "top_k": top_k,
        "scenario_trials": scenario_trials,
        "target_metric": "throughput",
        "param_ranges": {
            "rework_load_factor": {"min": 0.2, "max": 1.5, "type": "float"},
            "arrival_rate": {"min": 0.3, "max": 0.8, "type": "float"},
            "dr_period": {"min": 30, "max": 180, "type": "int"},
            "dr2_rework_multiplier": {"min": 1.0, "max": 3.0, "type": "float"},
        },
        "fixed_params": {
            "days": 1095,
            "small_exp_duration": 5,
            "bundle_size_small": 6,
            "mid_exp_duration": 20,
            "fin_exp_duration": 30,
            "bundle_size_mid": 2,
            "bundle_size_fin": 2,
            "dr1_capacity": 10,
            "dr2_capacity": 14,
            "dr3_capacity": 18,
            "dr1_cost_per_review": 1.0,
            "dr2_cost_per_review": 2.0,
            "dr3_cost_per_review": 3.0,
            "n_senior": 1,
            "n_coordinator": 1,
            "n_new": 0,
            "max_rework_cycles": 5,
            "decay": 0.7,
            "sampling_interval": 5.0,
            "rework_beta_a": 2.0,
            "rework_beta_b": 5.0,
            "rework_task_type_mix": 1.0,
            "conditional_prob_ratio": 0.8,
            "decision_latency_days": 2.0,
            "n_servers_mid": 5,
            "n_servers_fin": 3,
        },
    }
    run_doe_from_spec(spec, scenarios_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Run DOE exploration (Ver12).")
    parser.add_argument("--scenarios", default="scenarios.csv", help="Template scenarios.csv path")
    parser.add_argument("--out", default="output_doe", help="DOE output directory")
    parser.add_argument("--samples", type=int, default=30, help="DOE sample count")
    parser.add_argument("--trials", type=int, default=1, help="Trials per DOE sample")
    parser.add_argument("--top-k", type=int, default=5, help="Top K scenarios to export")
    parser.add_argument("--scenario-trials", type=int, default=100, help="n_trials for exported scenarios")
    args = parser.parse_args()

    scenarios_path = args.scenarios
    if not os.path.isabs(scenarios_path):
        scenarios_path = os.path.join(CURRENT_DIR, scenarios_path)

    out_dir = args.out
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(CURRENT_DIR, out_dir)

    run_doe(
        scenarios_path=scenarios_path,
        out_dir=out_dir,
        n_samples=args.samples,
        n_trials=args.trials,
        top_k=args.top_k,
        scenario_trials=args.scenario_trials,
    )

    print(f"DOE complete. Output in: {out_dir}")


if __name__ == "__main__":
    main()

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator import run_monte_carlo, latin_hypercube_sampling
import visualizer as viz


def _minmax(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    denom = max(max_v - min_v, 1e-12)
    return (series - min_v) / denom


def _score_doe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tp_n = _minmax(df["throughput"])
    p90_n = _minmax(df["p90_wait"])
    wip_n = _minmax(df["avg_wip"])
    rwk_n = _minmax(df["avg_reworks"])

    # Higher TP is better, lower P90/WIP/rework is better.
    df["score"] = tp_n - p90_n - 0.5 * wip_n - 0.5 * rwk_n
    return df


def _build_scenarios_from_doe(df_top: pd.DataFrame, template_df: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    if df_top.empty:
        return pd.DataFrame(columns=template_df.columns)

    base_row = template_df.iloc[0].copy()
    rows = []
    for i, row in enumerate(df_top.itertuples(), start=1):
        r = base_row.copy()
        if "scenario_name" in template_df.columns:
            r["scenario_name"] = f"DOE_TOP_{i:02d}"
        if "n_trials" in template_df.columns:
            r["n_trials"] = int(n_trials)
        if "arrival_rate" in template_df.columns:
            r["arrival_rate"] = float(row.arrival_rate)
        if "rework_load_factor" in template_df.columns:
            r["rework_load_factor"] = float(row.rework_load_factor)
        if "dr_period" in template_df.columns:
            r["dr_period"] = float(row.dr_period)
        if "dr_periods" in template_df.columns:
            r["dr_periods"] = str(round(float(row.dr_period), 3)).rstrip("0").rstrip(".")
        rows.append(r)

    return pd.DataFrame(rows, columns=template_df.columns)


def run_doe(
    scenarios_path: str,
    out_dir: str,
    n_samples: int,
    n_trials: int,
    top_k: int,
    scenario_trials: int,
):
    os.makedirs(out_dir, exist_ok=True)

    df_scenarios = pd.read_csv(scenarios_path)
    base_seed = 42

    doe_ranges = {
        "rework_load_factor": (0.2, 1.5),
        "arrival_rate": (0.3, 0.8),
        "dr_period": (30, 180),
    }
    fixed_params = {
        "days": 180,
        "small_exp_duration": 5,
        "proto_duration": 20,
        "bundle_size": 3,
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
        "decision_latency_days": 2.0
    }

    doe_samples = latin_hypercube_sampling(n_samples=n_samples, param_ranges=doe_ranges)
    doe_results = []

    for sample in doe_samples:
        params = {**fixed_params, **sample}
        trials = run_monte_carlo(n_trials=n_trials, use_parallel=False, base_seed=base_seed, **params)
        summaries = [t.get("summary", {}) for t in trials if t.get("summary")]
        if not summaries:
            summaries = [{"throughput": 0.0, "p90_wait": 0.0, "avg_wip": 0.0, "avg_reworks": 0.0}]

        tp = np.mean([s.get("throughput", 0.0) for s in summaries])
        p90 = np.mean([s.get("p90_wait") if "p90_wait" in s else s.get("lead_time_p90", 0.0) for s in summaries])
        wip = np.mean([s.get("avg_wip", 0.0) for s in summaries])
        rework = np.mean([s.get("avg_reworks", 0.0) for s in summaries])

        doe_results.append({
            **sample,
            "throughput": float(tp),
            "p90_wait": float(p90),
            "avg_wip": float(wip),
            "avg_reworks": float(rework),
        })

    doe_df = pd.DataFrame(doe_results)
    if doe_df.empty:
        return

    viz.plot_doe_analysis(doe_df, title="Ver11 DOE Sensitivity (exploration)")
    plt.savefig(os.path.join(out_dir, "doe_analysis.png"))
    plt.close()

    doe_df.to_csv(os.path.join(out_dir, "doe_results.csv"), index=False)

    doe_df = _score_doe(doe_df)
    doe_df = doe_df.sort_values("score", ascending=False)
    doe_df.to_csv(os.path.join(out_dir, "doe_results_scored.csv"), index=False)

    top_k = max(1, min(int(top_k), len(doe_df)))
    df_top = doe_df.head(top_k).copy()
    df_top.to_csv(os.path.join(out_dir, "doe_top_summary.csv"), index=False)

    df_suggest = _build_scenarios_from_doe(df_top, df_scenarios, scenario_trials)
    df_suggest.to_csv(os.path.join(out_dir, "scenarios_from_doe.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Run DOE exploration (Ver11).")
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

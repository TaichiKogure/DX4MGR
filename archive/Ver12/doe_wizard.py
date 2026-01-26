import os
import sys
import json
from typing import Dict, Any, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from simulator import run_monte_carlo
from run_doe import (
    run_doe_from_spec,
    load_base_params,
    summarize_trials,
    INT_PARAMS,
)

DEFAULT_PARAMS_DR2 = [
    "arrival_rate",
    "rework_load_factor",
    "dr2_rework_multiplier",
    "dr2_period",
    "dr2_capacity",
    "bundle_size_mid",
    "n_servers_mid",
]

RANGE_HINTS: Dict[str, Tuple[float, float]] = {
    "arrival_rate": (0.05, 1.5),
    "rework_load_factor": (0.2, 3.0),
    "dr2_rework_multiplier": (1.0, 4.0),
    "dr2_period": (15, 240),
    "dr2_capacity": (1, 60),
    "bundle_size_mid": (1, 12),
    "n_servers_mid": (1, 20),
    "decision_latency_days": (0.0, 10.0),
    "dr1_period": (15, 240),
    "dr3_period": (15, 240),
}

DEFAULT_WEIGHTS_BLOWUP = {
    "dr2_wait": 1.0,
    "dr2_cost": 0.7,
    "cost_per_completed": 0.5,
    "dr2_wait_ratio": 0.5,
    "avg_reworks": 0.3,
    "throughput": -0.6,
}

DEFAULT_WEIGHTS_MITIGATE = {
    "throughput": 1.0,
    "dr2_wait": -1.0,
    "dr2_cost": -0.7,
    "cost_per_completed": -0.5,
    "dr2_wait_ratio": -0.4,
    "avg_reworks": -0.3,
}

def _prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{text}{suffix}: ").strip()
        if raw:
            return raw
        if default is not None:
            return str(default)

def _prompt_bool(text: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{text} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False

def _prompt_int(text: str, default: int) -> int:
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return int(raw)
        except ValueError:
            print("整数を入力してください。")

def _prompt_float(text: str, default: float) -> float:
    while True:
        raw = _prompt(text, default=str(default))
        try:
            return float(raw)
        except ValueError:
            print("数値を入力してください。")

def _prompt_choice(text: str, choices: Dict[str, str], default_key: str) -> str:
    print(text)
    for key, label in choices.items():
        print(f"  {key}) {label}")
    while True:
        raw = _prompt("選択", default=default_key)
        if raw in choices:
            return raw

def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(CURRENT_DIR, path)

def _guess_param_type(param: str) -> str:
    if param in INT_PARAMS or param == "dr_period":
        return "int"
    return "float"

def _suggest_range(param: str, base_value: Optional[float]) -> Tuple[float, float]:
    if base_value is None or base_value <= 0:
        return RANGE_HINTS.get(param, (0.1, 1.0))
    min_v = base_value * 0.5
    max_v = base_value * 2.0
    if param in RANGE_HINTS:
        min_hint, max_hint = RANGE_HINTS[param]
        min_v = max(min_v, min_hint)
        max_v = min(max_v, max_hint)
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    return min_v, max_v

def _parse_range(text: str, param_type: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("min,max を指定してください。")
    min_v = float(parts[0])
    max_v = float(parts[1])
    if min_v > max_v:
        min_v, max_v = max_v, min_v
    if param_type == "int":
        return float(int(round(min_v))), float(int(round(max_v)))
    return min_v, max_v

def _prompt_param_ranges(param_names: List[str], base_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    ranges: Dict[str, Dict[str, Any]] = {}
    for param in param_names:
        base_val = base_params.get(param)
        param_type = _guess_param_type(param)
        min_v, max_v = _suggest_range(param, float(base_val) if base_val is not None else None)
        default = f"{min_v:.4g},{max_v:.4g}"
        while True:
            raw = _prompt(f"{param} の範囲(min,max)", default=default)
            try:
                min_v, max_v = _parse_range(raw, param_type)
                break
            except ValueError as e:
                print(e)
        ranges[param] = {"min": min_v, "max": max_v, "type": param_type}
    return ranges

def _prompt_weights(default_weights: Dict[str, float]) -> Dict[str, float]:
    print("スコア重み (metric=weight, カンマ区切り) を指定できます。")
    print(f"デフォルト: {default_weights}")
    if _prompt_bool("デフォルトを使いますか?", default=True):
        return dict(default_weights)
    raw = _prompt("weights 例: dr2_wait=1,dr2_cost=0.5,throughput=-0.4", default="")
    weights: Dict[str, float] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(val.strip())
        except ValueError:
            pass
    return weights if weights else dict(default_weights)

def main():
    print("=== DX4MGR Ver12 DOE Wizard ===")
    scenarios_path = _resolve_path(_prompt("scenarios.csv のパス", default="scenarios.csv"))
    out_dir = _resolve_path(_prompt("出力ディレクトリ", default="output_doe"))
    os.makedirs(out_dir, exist_ok=True)

    mode = _prompt_choice(
        "探索モード",
        {"1": "DR2悪化条件を抽出", "2": "DR2改善策を探索"},
        default_key="1",
    )
    mode_key = "blowup" if mode == "1" else "mitigate"

    base_params, _ = load_base_params(scenarios_path, use_base_from_csv=True)

    if _prompt_bool("デフォルトの探索パラメータを使いますか?", default=True):
        param_names = list(DEFAULT_PARAMS_DR2)
    else:
        raw = _prompt("探索したいパラメータをカンマ区切りで入力", default=",".join(DEFAULT_PARAMS_DR2))
        param_names = [p.strip() for p in raw.split(",") if p.strip()]

    param_ranges = _prompt_param_ranges(param_names, base_params)

    default_samples = max(20, len(param_names) * 8)
    n_samples = _prompt_int("DOEサンプル数", default_samples)
    n_trials = _prompt_int("DOEサンプルごとの試行回数", 1)
    top_k = _prompt_int("上位シナリオの出力数", 8)
    scenario_trials = _prompt_int("出力シナリオの n_trials", 100)

    constraints: Dict[str, float] = {}
    if _prompt_bool("DR1のガードレールを入れますか?", default=True):
        baseline_trials = _prompt_int("基準の簡易試行回数", 3)
        base_seed = 42
        trials = run_monte_carlo(
            n_trials=baseline_trials,
            use_parallel=False,
            base_seed=base_seed,
            **base_params,
        )
        baseline = summarize_trials(trials)
        print(
            f"Baseline DR1 wait={baseline['dr1_wait']:.2f}, "
            f"DR1 cost={baseline['dr1_cost']:.2f}"
        )
        ratio = _prompt_float("DR1許容倍率 (wait/cost)", 1.2)
        if baseline["dr1_wait"] > 0:
            constraints["dr1_wait_max"] = baseline["dr1_wait"] * ratio
        if baseline["dr1_cost"] > 0:
            constraints["dr1_cost_max"] = baseline["dr1_cost"] * ratio

    if mode_key == "blowup":
        score_weights = _prompt_weights(DEFAULT_WEIGHTS_BLOWUP)
        target_metric = "dr2_wait"
        scenario_prefix = "DOE_BLOWUP"
    else:
        score_weights = _prompt_weights(DEFAULT_WEIGHTS_MITIGATE)
        target_metric = "throughput"
        scenario_prefix = "DOE_MITIGATE"

    spec = {
        "use_base_from_csv": True,
        "n_samples": n_samples,
        "n_trials": n_trials,
        "top_k": top_k,
        "scenario_trials": scenario_trials,
        "target_metric": target_metric,
        "scenario_prefix": scenario_prefix,
        "param_ranges": param_ranges,
        "score_weights": score_weights,
        "constraints": constraints,
        "fixed_params": {},
        "insight_metrics": ["dr2_wait", "dr2_cost", "cost_per_completed", "throughput"],
    }

    spec_path = os.path.join(out_dir, "doe_spec.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    run_doe_from_spec(spec, scenarios_path, out_dir)
    print(f"DOE complete. Output in: {out_dir}")

if __name__ == "__main__":
    main()

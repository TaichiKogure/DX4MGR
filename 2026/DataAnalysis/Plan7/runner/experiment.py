import numpy as np
import concurrent.futures
import multiprocessing
from typing import Dict, Any, List, Callable
from core.engine import SimulationEngine

def run_single_trial(setup_func: Callable, trial_id: int, base_seed: int, **params):
    seed = base_seed + trial_id
    rng = np.random.default_rng(seed)
    
    engine = setup_func(rng, **params)
    engine.run(params.get("days", 365))
    
    return engine

def run_monte_carlo(setup_func: Callable, n_trials: int, base_seed: int, use_parallel: bool = True, **params):
    args = [(setup_func, i, base_seed) for i in range(n_trials)]
    
    results = []
    if use_parallel and n_trials > 1:
        cpu_count = min(multiprocessing.cpu_count(), n_trials)
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            # paramsを固定するためにクロージャっぽく渡す必要があるが、
            # ProcessPoolExecutorではpickle可能である必要があるため、argsに含める
            full_args = [(setup_func, i, base_seed, params) for i in range(n_trials)]
            results = list(executor.map(_run_wrapper, full_args))
    else:
        for i in range(n_trials):
            results.append(_run_wrapper((setup_func, i, base_seed, params)))
            
    return results

def _run_wrapper(args):
    setup_func, trial_id, base_seed, params = args
    return run_single_trial(setup_func, trial_id, base_seed, **params)

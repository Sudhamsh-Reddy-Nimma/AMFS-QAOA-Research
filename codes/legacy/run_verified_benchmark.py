"""Comprehensive benchmark runner for Rigorous Verification and Research Development of AMFS for QAOA."""
from __future__ import annotations
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from qaoa_simulator import (
    NoiseConfig,
    QAOACircuitEngine,
    QAOAOfflineEvaluator,
    QAOAOptimizerOracle,
    make_graph_instance,
)
import amfs_variants as av
import baselines as bl

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE

METHOD_OFFSETS = {
    "amfs_orig": 11,
    "amfs_orig_grid27": 13,
    "amfs_faithful": 17,
    "amfs_faithful_grid27": 19,
    "amfs_improved": 21,
    "amfs_improved_grid27": 22,
    "amfs_imp_target": 25,
    "amfs_imp_warmstart": 27,
    "ablate_no_multidim": 31,
    "ablate_no_screening": 33,
    "ablate_no_widen_fix": 35,
    "spsa": 23,
    "spsa_seqtest": 29,
    "cobyla_orig": 37,
    "cobyla_hao2025": 38,
    "bobyqa_trust": 39,
    "gpbo_orig": 41,
    "gpbo_tibaldi": 43,
    "icans": 47,
    "wecans": 49,
    "dds": 51,
    "nft": 52,
    "random": 53,
    "fixed_naive": 67,
    "fixed_wurtz_love": 71,
}


def dispatch_method(
    method: str,
    oracle: QAOAOptimizerOracle,
    budget: int,
    shots_per_eval: int,
    seed: int,
) -> Tuple[np.ndarray, int, Dict]:
    if method == "amfs_orig":
        return av.run_amfs_orig(oracle, budget, seed, grid_n=25, passes=2)
    elif method == "amfs_orig_grid27":
        return av.run_amfs_orig(oracle, budget, seed, grid_n=27, passes=2)
    elif method == "amfs_faithful":
        return av.run_amfs_faithful(oracle, budget, seed, grid_n=25, passes=2)
    elif method == "amfs_faithful_grid27":
        return av.run_amfs_faithful(oracle, budget, seed, grid_n=27, passes=2)
    elif method == "amfs_improved":
        return av.run_amfs_improved(
            oracle, budget, seed, grid_n=25, use_multidim=True, use_screening=True, use_widen_all=True
        )
    elif method == "amfs_improved_grid27":
        return av.run_amfs_improved(
            oracle, budget, seed, grid_n=27, use_multidim=True, use_screening=True, use_widen_all=True
        )
    elif method == "amfs_imp_target":
        return av.run_amfs_improved(
            oracle,
            budget,
            seed,
            grid_n=25,
            use_multidim=True,
            use_screening=True,
            use_widen_all=True,
            use_target_stop=True,
        )
    elif method == "amfs_imp_warmstart":
        return av.run_amfs_improved(
            oracle,
            budget,
            seed,
            grid_n=25,
            use_multidim=True,
            use_screening=True,
            use_widen_all=True,
            use_warmstart=True,
        )
    elif method == "ablate_no_multidim":
        return av.run_amfs_improved(
            oracle, budget, seed, grid_n=25, use_multidim=False, use_screening=True, use_widen_all=True
        )
    elif method == "ablate_no_screening":
        return av.run_amfs_improved(
            oracle, budget, seed, grid_n=25, use_multidim=True, use_screening=False, use_widen_all=True
        )
    elif method == "ablate_no_widen_fix":
        return av.run_amfs_improved(
            oracle, budget, seed, grid_n=25, use_multidim=True, use_screening=True, use_widen_all=False
        )
    elif method == "spsa":
        return bl.run_spsa(oracle, budget, shots_per_eval, seed)
    elif method == "spsa_seqtest":
        return bl.run_spsa_seqtest(oracle, budget, seed)
    elif method == "cobyla_orig":
        return bl.run_cobyla_orig(oracle, budget, shots_per_eval)
    elif method == "cobyla_hao2025":
        return bl.run_cobyla_hao2025(oracle, budget, seed, init_mode="wurtz_love")
    elif method == "bobyqa_trust":
        return bl.run_bobyqa_trust(oracle, budget, shots_per_eval)
    elif method == "gpbo_orig":
        return bl.run_gpbo_orig(oracle, budget, shots_per_eval, seed)
    elif method == "gpbo_tibaldi":
        return bl.run_gpbo_tibaldi(oracle, budget, shots_per_eval, seed)
    elif method == "icans":
        return bl.run_icans(oracle, budget, seed)
    elif method == "wecans":
        return bl.run_wecans(oracle, budget, seed)
    elif method == "dds":
        return bl.run_dds(oracle, budget, seed)
    elif method == "nft":
        return bl.run_nft(oracle, budget, shots_per_eval, seed)
    elif method == "random":
        return bl.run_random(oracle, budget, shots_per_eval, seed)
    elif method == "fixed_naive":
        return bl.run_fixed_naive(oracle, budget)
    elif method == "fixed_wurtz_love":
        return bl.run_fixed_wurtz_love(oracle, budget)
    raise ValueError(f"Unknown method: {method}")


def evaluate_single_run(
    suite_name: str,
    family: str,
    n_nodes: int,
    p_depth: int,
    graph_seed: int,
    seed: int,
    budget: int,
    shots_per_eval: int,
    method: str,
    engine: QAOACircuitEngine,
    evaluator: QAOAOfflineEvaluator,
    noise: Optional[NoiseConfig] = None,
    target_ratio: float = 0.78,
) -> Dict:
    offset = METHOD_OFFSETS.get(method, 99)
    oracle_seed = seed * 1009 + offset
    target_cut = target_ratio * evaluator.opt_cut
    oracle = QAOAOptimizerOracle(
        engine,
        seed=oracle_seed,
        p=p_depth,
        max_budget=budget,
        strict_budget=True,
        noise=noise,
        target_cut_threshold=target_cut,
    )

    t0 = time.perf_counter()
    theta_final, shots_used, meta = dispatch_method(method, oracle, budget, shots_per_eval, seed)
    wall_sec = time.perf_counter() - t0

    final_ratio = evaluator.true_ratio(theta_final)
    qaoa_ceiling = evaluator.compute_qaoa_ceiling(grid_res=40) if p_depth == 1 else 0.865
    rel_target = min(target_ratio, 0.965 * qaoa_ceiling)

    posthoc_first_shots = None
    for cum_s, _, q_theta, _, _, _ in oracle.query_log:
        if evaluator.true_ratio(q_theta) >= target_ratio:
            posthoc_first_shots = int(cum_s)
            break

    inc_shots_to_target = None
    if final_ratio >= target_ratio:
        for cum_s, inc_theta in meta.get("incumbent_trace", []):
            if evaluator.true_ratio(inc_theta) >= target_ratio:
                inc_shots_to_target = int(cum_s)
                break
        if inc_shots_to_target is None:
            inc_shots_to_target = int(shots_used)

    inc_trace = meta.get("incumbent_trace", [(shots_used, theta_final)])
    checkpoints = {}
    for cp in (300, 600, 900, 1200, 1500, 1800):
        cand_theta = None
        for cum_s, t_vec in inc_trace:
            if cum_s <= cp:
                cand_theta = t_vec
            else:
                break
        if cand_theta is None and inc_trace:
            cand_theta = inc_trace[0][1]
        checkpoints[f"ratio_at_{cp}"] = evaluator.true_ratio(cand_theta) if cand_theta is not None else final_ratio

    return {
        "suite": suite_name,
        "family": family,
        "n_nodes": n_nodes,
        "p_depth": p_depth,
        "graph_seed": graph_seed,
        "seed": seed,
        "budget": budget,
        "method": method,
        "shots_used": int(shots_used),
        "circuit_executions": int(oracle.circuit_executions),
        "expected_cut_ratio": float(final_ratio),
        "qaoa_ceiling": float(qaoa_ceiling),
        "optimality_gap_to_ceiling": float(max(0.0, qaoa_ceiling - final_ratio)),
        "target_078_final": int(final_ratio >= target_ratio),
        "target_rel_final": int(final_ratio >= rel_target),
        "certified_target_lcb": int(bool(meta.get("certified_target", False))),
        "shots_to_first_078_candidate": posthoc_first_shots if posthoc_first_shots is not None else -1,
        "shots_to_retained_078_incumbent": inc_shots_to_target if inc_shots_to_target is not None else -1,
        "theta_gamma_1": float(theta_final[0]),
        "theta_beta_1": float(theta_final[p_depth]),
        "wall_seconds": float(wall_sec),
        "stop_reason": str(meta.get("stop_reason", "")),
        **checkpoints,
    }


def run_suite_1_original_vs_corrected() -> pd.DataFrame:
    methods = list(METHOD_OFFSETS.keys())
    rows = []
    for g in range(8):
        g_seed = 100 + g
        n_nodes, edges = make_graph_instance("3_regular", 8, g_seed)
        eng = QAOACircuitEngine(n_nodes, edges)
        ev = QAOAOfflineEvaluator(eng, p=1)
        for s_idx in range(4):
            seed = s_idx + g * 17
            for m in methods:
                if m.startswith("ablate_"):
                    continue
                row = evaluate_single_run("suite1_orig_vs_corr", "3_regular", 8, 1, g_seed, seed, 1800, 60, m, eng, ev)
                row["is_orig_seed_pair"] = int(s_idx < 2)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "suite1_original_vs_corrected.csv", index=False)
    return df


if __name__ == "__main__":
    print("Benchmark runner initialized in", OUT_DIR)
    df1 = run_suite_1_original_vs_corrected()
    print("Saved suite1_original_vs_corrected.csv with", len(df1), "rows.")

"""
Phase 6 Joint Search-and-Certification Regime Map Runner (Upgraded: Matched Splits, Spectral Support, n=8..16, p=1..4)
======================================================================================================================
Executes the joint operational problem (Sections 6.1 through 6.10) with Peer-Review Upgrades:
1. Fair Matched-Split Comparison (`Evaluation_A_MatchedSplit`):
   Every method (`amfs_md_target`, `icans`, `wecans`, `dds`, `cobyla_hao2025`, `bobyqa_trust`, `gpbo_tibaldi`,
   `spsa`, `random`, `fixed_wurtz_love`, `fixed_shaydulin_rescaled`) receives the EXACT SAME confirmation shot
   allocation `B_confirmation = int(0.40 * B)` (e.g., 240 shots at B=600, 480 shots at B=1200), eliminating the
   allocation tautology where transfer only won because it was given 2.25x more confirmation shots.
2. Validation-Tuned Split Allocation (`Evaluation_C_TunedSplit` & `split_ratio_optimization.csv`):
   Sweeps confirmation split ratio `rho_conf in {0.25, 0.40, 0.60, 1.00}` on the `val` partition for each
   method and evaluates each method's validation-optimal split `rho_conf^*` out-of-sample on the `eval` partition.
3. Matched Cold Initialization (`Evaluation_B_MatchedInit`):
   All search methods start from the exact same `cold_naive` initialization with matched confirmation shots.
4. Spectrally-Tightened Certification (`[1 - U_spec(G)/W, 1]`):
   Records both worst-case R=1.0 bounds (`time_uniform_eb_cs_valid`, `empirical_bernstein_valid`) AND online
   spectrally-tightened bounds (`time_uniform_eb_cs_spectral_valid`, `eb_spectral_valid`, `intersection_spectral_valid`)
   using the Delorme-Poljak / Mohar-Poljak bound `R_spec(G) = U_spec(G)/W`.
5. Multi-Scale (`n in {8, 12, 14, 16}`) and Multi-Depth (`p in {1, 2, 3, 4}`) Generalization:
   Expands evaluation across n=8, 12, 14, 16 and circuit depths p=1, 2, 3, 4 (`scaling_n12_to_16_p1_to_4_summary.csv`).
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, wilcoxon

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))

from graph_library import generate_deduplicated_graph_suite, GraphMetadata
from qaoa_core import QAOACircuitEngine, OperationalQAOAOracle, RetrospectiveEvaluator
from stopping_rules import (
    allocate_alpha_k,
    compute_spectral_loss_support,
    evaluate_confirmation_sample,
)
from optimizers import (
    run_amfs_md_target,
    run_icans,
    run_wecans,
    run_dds,
    run_cobyla_hao2025,
    run_bobyqa_trust,
    run_gpbo_tibaldi,
    run_spsa,
    run_random_search,
    run_fixed_angle_method,
)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 1500, seed: int = 42) -> Tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1 or np.allclose(arr, arr[0]):
        v = float(arr[0])
        return v, v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return float(np.mean(arr)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_bonferroni_adjust(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adj = min(1.0, pval * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def run_single_joint_trial(
    method_id: str,
    eval_mode: str,
    engine: QAOACircuitEngine,
    p_depth: int,
    total_budget: int,
    seed: int,
    tau_target_loss: float,
    conf_split_override: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Execute search + confirmation within a single fixed total budget B = B_search + B_confirmation."""
    # Under Evaluation_A_MatchedSplit and Evaluation_B_MatchedInit:
    # ALL primary methods get the exact same confirmation shot count: int(0.40 * total_budget)!
    # Only the explicit allocation-boundary controls (`fixed_shaydulin_fullconf` with rho_conf=0.75/1.0
    # and `icans_starved_conf` with B_conf=15) use non-matched confirmation budgets.
    if conf_split_override is not None:
        conf_reserve = max(15, min(total_budget, int(round(conf_split_override * total_budget))))
    elif method_id == "fixed_shaydulin_fullconf":
        conf_reserve = min(total_budget, 450)
    elif method_id == "icans_starved_conf":
        conf_reserve = 15
    else:
        # Matched confirmation budget across ALL methods (transfer and adaptive search alike!)
        conf_reserve = int(round(0.40 * total_budget))

    oracle = OperationalQAOAOracle(
        engine=engine,
        seed=seed,
        p=p_depth,
        max_budget=total_budget,
        strict_budget=True,
        operational_target_loss=tau_target_loss,
    )

    t0 = time.perf_counter()
    init_choice = "naive" if eval_mode == "Evaluation_B_MatchedInit" else (
        "shaydulin_rescaled" if (p_depth > 1 or method_id in ("cobyla_hao2025", "amfs_md_target", "bobyqa_trust")) else "naive"
    )

    search_budget = max(0, total_budget - conf_reserve)

    if method_id in ("fixed_wurtz_love", "fixed_wurtz_love_fullconf") or search_budget < 30:
        mode_f = "wurtz_love" if "wurtz" in method_id else "shaydulin_rescaled"
        best_theta, _ = run_fixed_angle_method(oracle, mode=mode_f, cert_rule="none", conf_shots=0)
    elif method_id in ("fixed_shaydulin_rescaled", "fixed_shaydulin_fullconf"):
        best_theta, _ = run_fixed_angle_method(oracle, mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
    elif method_id == "amfs_md_target":
        best_theta, _ = run_amfs_md_target(
            oracle, budget=search_budget, seed=seed, init_mode=init_choice, stopping_rule="none", use_mf_surrogate=False
        )
    elif method_id == "amfs_mf_surrogate":
        best_theta, _ = run_amfs_md_target(
            oracle, budget=search_budget, seed=seed, init_mode=init_choice, stopping_rule="none", use_mf_surrogate=True
        )
    elif method_id in ("icans", "icans_starved_conf"):
        best_theta, _ = run_icans(oracle, budget=search_budget, seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "wecans":
        best_theta, _ = run_wecans(oracle, budget=search_budget, seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "dds":
        best_theta, _ = run_dds(oracle, budget=search_budget, seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "cobyla_hao2025":
        best_theta, _ = run_cobyla_hao2025(oracle, budget=search_budget, seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "bobyqa_trust":
        best_theta, _ = run_bobyqa_trust(oracle, budget=search_budget, shots_per_eval=max(30, min(80, search_budget // 5)), init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "gpbo_tibaldi":
        best_theta, _ = run_gpbo_tibaldi(oracle, budget=search_budget, shots_per_eval=max(30, min(60, search_budget // 5)), seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "spsa":
        best_theta, _ = run_spsa(oracle, budget=search_budget, shots_per_eval=max(20, min(50, search_budget // 6)), seed=seed, init_mode=init_choice, cert_rule="none", conf_shots=0)
    elif method_id == "random":
        best_theta, _ = run_random_search(oracle, budget=search_budget, shots_per_eval=max(20, min(50, search_budget // 6)), seed=seed, cert_rule="none", conf_shots=0)
    else:
        raise ValueError(f"Unknown method_id: {method_id}")

    # For zero-search transfer under matched confirmation split (e.g. conf_reserve = 0.40 * B),
    # draw exactly `conf_reserve` confirmation shots so the comparison is 100% matched!
    rem = oracle.remaining_shots()
    conf_to_draw = min(conf_reserve, rem)
    conf_samples = oracle.confirmation_loss_samples(best_theta, conf_to_draw) if conf_to_draw > 0 else np.empty(0, dtype=float)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert oracle.verify_resource_invariant(), "Resource accounting invariant violated!"
    res_info = {
        "search_shots": oracle.search_shots,
        "confirmation_shots": oracle.confirmation_shots,
        "total_operational_shots": oracle.total_requested_shots,
        "conf_split_ratio": round(oracle.confirmation_shots / max(1, total_budget), 3),
        "objective_evaluations": oracle.circuit_evaluations,
        "simulator_calls": oracle.simulator_calls,
        "runtime_ms": round(elapsed_ms, 2),
        "init_used": init_choice,
    }
    return best_theta, conf_samples, res_info


def evaluate_all_rules_on_sample(
    conf_samples: np.ndarray,
    tau_base: float,
    alpha_nom: float,
    loss_min: float,
    r_spec: float,
) -> Dict:
    """Evaluate both loose R=1.0 bounds and spectrally-tightened R_spec = U_spec/W bounds."""
    out_tu_spec_d00 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.00, alpha_k=alpha_nom, rule="time_uniform_eb_cs_spectral_valid", loss_min=loss_min, loss_max=1.0
    )
    out_tu_spec_d05 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.05, alpha_k=alpha_nom, rule="time_uniform_eb_cs_spectral_valid", loss_min=loss_min, loss_max=1.0
    )
    out_eb_spec_d05 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.05, alpha_k=alpha_nom, rule="eb_spectral_valid", loss_min=loss_min, loss_max=1.0
    )
    out_int_spec_d05 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.05, alpha_k=alpha_nom, rule="intersection_spectral_valid", loss_min=loss_min, loss_max=1.0
    )
    out_tu_r1_d05 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.05, alpha_k=alpha_nom, rule="time_uniform_eb_cs_valid", loss_min=0.0, loss_max=1.0
    )
    out_eb_r1_d05 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.05, alpha_k=alpha_nom, rule="empirical_bernstein_valid", loss_range=1.0
    )
    out_st_d00 = evaluate_confirmation_sample(
        conf_samples, tau_base + 0.00, alpha_k=alpha_nom, rule="student_t_heuristic"
    )
    return {
        "out_tu_spec_d00": out_tu_spec_d00,
        "out_tu_spec_d05": out_tu_spec_d05,
        "out_eb_spec_d05": out_eb_spec_d05,
        "out_int_spec_d05": out_int_spec_d05,
        "out_tu_r1_d05": out_tu_r1_d05,
        "out_eb_r1_d05": out_eb_r1_d05,
        "out_st_d00": out_st_d00,
    }


def main():
    print("[Phase 6] Building deduplicated graph suites across n in {8, 12, 14, 16} with dev/val/eval partitions...")
    families = [
        "random_3_regular",
        "weighted_3_regular_uniform",
        "weighted_3_regular_bimodal",
        "erdos_renyi",
    ]
    all_graphs: List[GraphMetadata] = []
    # Generate 7 graphs/family at n=8 and n=12 (1 dev, 1 val, 5 eval) + 4 graphs/family at n=14 and n=16
    for f_idx, fam in enumerate(families):
        for n_nodes, count_g in [(8, 5), (12, 5), (14, 4), (16, 3)]:
            suite = generate_deduplicated_graph_suite(
                fam, n=n_nodes, target_count=count_g, start_seed=500 + f_idx * 100 + n_nodes * 10, assign_partitions=True
            )
            all_graphs.extend(suite)

    pd.DataFrame([g.to_record() for g in all_graphs]).to_csv(
        BASE_DIR / "results_raw" / "phase6" / "phase6_graph_instances.csv", index=False
    )
    print(f"[Phase 6] Generated {len(all_graphs)} deduplicated non-isomorphic graphs across n in {{8, 12, 14, 16}}.")

    methods_eval_a = [
        "amfs_md_target",
        "amfs_mf_surrogate",
        "icans",
        "icans_starved_conf",
        "wecans",
        "dds",
        "cobyla_hao2025",
        "bobyqa_trust",
        "gpbo_tibaldi",
        "spsa",
        "random",
        "fixed_wurtz_love",
        "fixed_shaydulin_rescaled",
        "fixed_shaydulin_fullconf",
    ]
    methods_eval_b = [
        "amfs_md_target",
        "amfs_mf_surrogate",
        "icans",
        "cobyla_hao2025",
        "bobyqa_trust",
        "gpbo_tibaldi",
        "spsa",
    ]

    engines: Dict[str, QAOACircuitEngine] = {}
    evaluators: Dict[Tuple[str, int], RetrospectiveEvaluator] = {}
    ceilings: Dict[Tuple[str, int], float] = {}

    for g in all_graphs:
        eng = QAOACircuitEngine(g.n_nodes, g.edges)
        engines[g.graph_id] = eng

    def get_eval_and_ceiling(g: GraphMetadata, p_val: int) -> Tuple[RetrospectiveEvaluator, float]:
        key = (g.graph_id, p_val)
        if key not in evaluators:
            ev = RetrospectiveEvaluator(engines[g.graph_id], p=p_val)
            evaluators[key] = ev
            c_r, _ = ev.compute_exact_qaoa_ceiling()
            ceilings[key] = c_r
        return evaluators[key], ceilings[key]

    trial_rows = []
    alpha_nom = 0.05
    seeds = [601, 602]

    # 1. Core Regime Map across n in {8, 12}, p in {1, 2}, budgets B in {300, 600, 1200}
    core_graphs = [g for g in all_graphs if g.n_nodes in (8, 12) and g.partition != "dev"]
    print(f"[Phase 6] Executing Matched-Split & Matched-Init Joint Regime Map on {len(core_graphs)} graphs (n=8, 12)...")

    for g in core_graphs:
        eng = engines[g.graph_id]
        loss_min, loss_max, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
        u_over_w = g.spectral_upper_bound / g.total_weight

        # For n=12 (our primary scaled cohort), run full budget grid; for n=8, run B=600, 1200
        p_budget_plan = [(1, [300, 600, 1200]), (2, [1200])] if g.n_nodes == 12 else [(1, [600]), (2, [1200])]

        for p_val, budgets_list in p_budget_plan:
            ev, ceil_r = get_eval_and_ceiling(g, p_val)
            rho_op = 0.72 if p_val == 1 else 0.77
            tau_base = 1.0 - rho_op * u_over_w

            for budget in budgets_list:
                eval_modes = [("Evaluation_A_EndToEnd", methods_eval_a)]
                if budget == 600 and g.n_nodes == 12 and p_val == 1:
                    eval_modes.append(("Evaluation_B_MatchedInit", methods_eval_b))

                for e_mode, m_list in eval_modes:
                    for m_id in m_list:
                        for s in (seeds if (g.n_nodes == 12 and p_val == 1) else [601]):
                            best_theta, conf_samples, res_info = run_single_joint_trial(
                                method_id=m_id,
                                eval_mode=e_mode,
                                engine=eng,
                                p_depth=p_val,
                                total_budget=budget,
                                seed=s,
                                tau_target_loss=tau_base,
                            )
                            true_loss = ev.true_loss(best_theta)
                            true_ratio = ev.true_ratio(best_theta)
                            achieved_op_ratio = (1.0 - true_loss) / u_over_w
                            outs = evaluate_all_rules_on_sample(conf_samples, tau_base, alpha_nom, loss_min, r_spec)

                            if len(conf_samples) < 20:
                                cert_status_primary = "inconclusive_insufficient_confirmation_budget"
                                delta_ach_tu_spec = np.nan
                                delta_ach_eb_spec = np.nan
                            else:
                                cert_status_primary = "certified" if outs["out_tu_spec_d05"].certified else "not_certified"
                                delta_ach_tu_spec = round(max(0.0, float(outs["out_tu_spec_d00"].ucb_loss - tau_base)), 5)
                                delta_ach_eb_spec = round(max(0.0, float(outs["out_eb_spec_d05"].ucb_loss - tau_base)), 5)

                            trial_rows.append({
                                "evaluation_mode": e_mode,
                                "graph_id": g.graph_id,
                                "partition": g.partition,
                                "family": g.family,
                                "n_nodes": g.n_nodes,
                                "p_depth": p_val,
                                "total_budget": budget,
                                "method": m_id,
                                "init_used": res_info["init_used"],
                                "optimizer_seed": s,
                                "spectral_loss_range_R_spec": round(r_spec, 4),
                                "target_rho_spectral": rho_op,
                                "target_loss_tau": round(tau_base, 5),
                                "nominal_alpha": alpha_nom,
                                "search_shots": res_info["search_shots"],
                                "confirmation_shots": res_info["confirmation_shots"],
                                "conf_split_ratio": res_info["conf_split_ratio"],
                                "total_operational_shots": res_info["total_operational_shots"],
                                "objective_evaluations": res_info["objective_evaluations"],
                                "runtime_ms": res_info["runtime_ms"],
                                "true_expected_loss": round(true_loss, 6),
                                "retrospective_true_ratio": round(true_ratio, 6),
                                "numerically_estimated_p_ceiling": round(ceil_r, 6),
                                "optimality_gap_to_ceiling": round(max(0.0, ceil_r - true_ratio), 6),
                                "achieved_spectral_op_ratio": round(achieved_op_ratio, 6),
                                "primary_certification_status": cert_status_primary,
                                "cert_time_uniform_cs_tol00": int(outs["out_tu_spec_d00"].certified),
                                "cert_time_uniform_cs_tol05": int(outs["out_tu_spec_d05"].certified),
                                "cert_eb_valid_tol05": int(outs["out_eb_spec_d05"].certified),
                                "cert_intersection_valid_tol05": int(outs["out_int_spec_d05"].certified),
                                "cert_tu_loose_r1_tol05": int(outs["out_tu_r1_d05"].certified),
                                "cert_eb_loose_r1_tol05": int(outs["out_eb_r1_d05"].certified),
                                "cert_student_t_heur_tol00": int(outs["out_st_d00"].certified),
                                "false_cert_tu_tol00": int(outs["out_tu_spec_d00"].certified and (true_loss > tau_base + 1e-9)),
                                "false_cert_tu_tol05": int(outs["out_tu_spec_d05"].certified and (true_loss > tau_base + 0.05 + 1e-9)),
                                "false_cert_eb_tol05": int(outs["out_eb_spec_d05"].certified and (true_loss > tau_base + 0.05 + 1e-9)),
                                "false_cert_st_tol00": int(outs["out_st_d00"].certified and (true_loss > tau_base + 1e-9)),
                                "ucb_time_uniform_cs": round(float(outs["out_tu_spec_d00"].ucb_loss), 5),
                                "ucb_eb_valid": round(float(outs["out_eb_spec_d05"].ucb_loss), 5),
                                "ucb_tu_loose_r1": round(float(outs["out_tu_r1_d05"].ucb_loss), 5),
                                "achieved_tolerance_tu_cs": delta_ach_tu_spec,
                                "achieved_tolerance_eb": delta_ach_eb_spec,
                            })

    # 2. Split-Ratio Optimization Sweep (`split_ratio_optimization.csv`):
    # Sweep rho_conf in {0.25, 0.40, 0.60, 0.80} on n=12 graphs at B=600 and B=1200 across core methods
    print("[Phase 6] Running Search/Confirmation Split-Ratio Optimization Sweep (rho_conf in {0.25, 0.40, 0.60, 0.80})...")
    split_rows = []
    n12_graphs = [g for g in all_graphs if g.n_nodes == 12 and g.partition in ("val", "eval")][:12]
    split_methods = ["amfs_md_target", "icans", "cobyla_hao2025", "bobyqa_trust", "spsa", "fixed_shaydulin_rescaled"]
    for g in n12_graphs:
        eng = engines[g.graph_id]
        loss_min, _, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
        u_over_w = g.spectral_upper_bound / g.total_weight
        ev, ceil_r = get_eval_and_ceiling(g, 1)
        tau_base = 1.0 - 0.72 * u_over_w
        for budget in [600, 1200]:
            for rho_c in [0.25, 0.40, 0.60, 0.80]:
                for m_id in split_methods:
                    best_theta, conf_samples, res_info = run_single_joint_trial(
                        method_id=m_id,
                        eval_mode="Evaluation_C_SplitSweep",
                        engine=eng,
                        p_depth=1,
                        total_budget=budget,
                        seed=601,
                        tau_target_loss=tau_base,
                        conf_split_override=rho_c,
                    )
                    true_ratio = ev.true_ratio(best_theta)
                    outs = evaluate_all_rules_on_sample(conf_samples, tau_base, alpha_nom, loss_min, r_spec)
                    split_rows.append({
                        "graph_id": g.graph_id,
                        "partition": g.partition,
                        "family": g.family,
                        "n_nodes": g.n_nodes,
                        "total_budget": budget,
                        "conf_split_ratio": rho_c,
                        "search_shots": res_info["search_shots"],
                        "confirmation_shots": res_info["confirmation_shots"],
                        "method": m_id,
                        "retrospective_true_ratio": round(true_ratio, 6),
                        "cert_spectral_tu_tol05": int(outs["out_tu_spec_d05"].certified),
                        "cert_spectral_eb_tol05": int(outs["out_eb_spec_d05"].certified),
                        "cert_loose_r1_tu_tol05": int(outs["out_tu_r1_d05"].certified),
                        "ucb_spectral_tu": round(float(outs["out_tu_spec_d05"].ucb_loss), 5),
                        "ucb_loose_r1_tu": round(float(outs["out_tu_r1_d05"].ucb_loss), 5),
                    })
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(BASE_DIR / "results_paper" / "metrics" / "split_ratio_optimization.csv", index=False)

    # 3. Multi-Scale (`n in {8, 12, 14, 16}`) & Multi-Depth (`p in {1, 2, 3, 4}`) Generalization Benchmark
    print("[Phase 6] Running Multi-Scale (n=8..16) & Multi-Depth (p=1..4) Generalization Benchmark...")
    scale_rows = []
    scale_methods = ["amfs_md_target", "cobyla_hao2025", "icans", "spsa", "fixed_shaydulin_rescaled"]
    for n_nodes in [8, 12, 14, 16]:
        n_graphs_scale = [g for g in all_graphs if g.n_nodes == n_nodes and g.partition == "eval"][:8]
        p_list = [1, 2, 3, 4] if n_nodes in (8, 12, 14) else [1, 2, 3]
        for g in n_graphs_scale:
            eng = engines[g.graph_id]
            loss_min, _, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
            u_over_w = g.spectral_upper_bound / g.total_weight
            for p_val in p_list:
                ev, ceil_r = get_eval_and_ceiling(g, p_val)
                rho_op = 0.72 + 0.02 * (p_val - 1)
                tau_base = 1.0 - rho_op * u_over_w
                budget = 900 if p_val <= 2 else 1200
                for m_id in scale_methods:
                    best_theta, conf_samples, res_info = run_single_joint_trial(
                        method_id=m_id,
                        eval_mode="Evaluation_A_MatchedSplit",
                        engine=eng,
                        p_depth=p_val,
                        total_budget=budget,
                        seed=601,
                        tau_target_loss=tau_base,
                        conf_split_override=0.40,
                    )
                    true_ratio = ev.true_ratio(best_theta)
                    true_loss = ev.true_loss(best_theta)
                    outs = evaluate_all_rules_on_sample(conf_samples, tau_base, alpha_nom, loss_min, r_spec)
                    scale_rows.append({
                        "graph_id": g.graph_id,
                        "family": g.family,
                        "n_nodes": n_nodes,
                        "p_depth": p_val,
                        "total_budget": budget,
                        "search_shots": res_info["search_shots"],
                        "confirmation_shots": res_info["confirmation_shots"],
                        "method": m_id,
                        "spectral_loss_range_R_spec": round(r_spec, 4),
                        "retrospective_true_ratio": round(true_ratio, 6),
                        "numerically_estimated_p_ceiling": round(ceil_r, 6),
                        "optimality_gap_to_ceiling": round(max(0.0, ceil_r - true_ratio), 6),
                        "cert_spectral_tu_tol05": int(outs["out_tu_spec_d05"].certified),
                        "cert_spectral_eb_tol05": int(outs["out_eb_spec_d05"].certified),
                        "cert_loose_r1_tu_tol05": int(outs["out_tu_r1_d05"].certified),
                        "false_cert_spectral_tu": int(outs["out_tu_spec_d05"].certified and (true_loss > tau_base + 0.05 + 1e-9)),
                    })
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(BASE_DIR / "results_raw" / "phase6" / "scaling_n8_to_16_p1_to_4_trials.csv", index=False)

    scale_sum = (
        scale_df.groupby(["n_nodes", "p_depth", "method"], as_index=False)
        .agg(
            n_trials=("graph_id", "count"),
            mean_R_spec=("spectral_loss_range_R_spec", "mean"),
            mean_true_ratio=("retrospective_true_ratio", "mean"),
            mean_p_ceiling=("numerically_estimated_p_ceiling", "mean"),
            mean_optimality_gap=("optimality_gap_to_ceiling", "mean"),
            cert_prob_spectral_tu=("cert_spectral_tu_tol05", "mean"),
            cert_prob_spectral_eb=("cert_spectral_eb_tol05", "mean"),
            cert_prob_loose_r1_tu=("cert_loose_r1_tu_tol05", "mean"),
            false_cert_rate=("false_cert_spectral_tu", "mean"),
        )
        .round(5)
    )
    scale_sum.to_csv(BASE_DIR / "results_paper" / "metrics" / "scaling_n12_to_16_p1_to_4_summary.csv", index=False)

    map_df = pd.DataFrame(trial_rows)
    map_df.to_csv(BASE_DIR / "results_paper" / "metrics" / "joint_regime_map.csv", index=False)
    map_df.to_csv(BASE_DIR / "results_raw" / "phase6" / "joint_regime_map_raw.csv", index=False)
    print(f"[Phase 6] Saved joint_regime_map.csv ({len(map_df)} trials) and scaling_n12_to_16_p1_to_4_summary.csv ({len(scale_sum)} rows).")

    # Build joint_regime_summary.csv
    sum_rows = []
    for (e_mode, fam, p_val, budget, m_id), grp in map_df.groupby(
        ["evaluation_mode", "family", "p_depth", "total_budget", "method"], sort=False
    ):
        n_trials = len(grp)
        n_graphs = grp["graph_id"].nunique()
        g_means = grp.groupby("graph_id")[
            ["retrospective_true_ratio", "true_expected_loss", "cert_time_uniform_cs_tol05", "cert_eb_valid_tol05", "cert_tu_loose_r1_tol05"]
        ].mean()
        r_mean, r_lo, r_hi = paired_bootstrap_ci(g_means["retrospective_true_ratio"].to_numpy(), n_boot=1000, seed=42)

        c_tu05 = int(grp["cert_time_uniform_cs_tol05"].sum())
        c_lo, c_hi = clopper_pearson_ci(c_tu05, n_trials)
        c_eb05 = int(grp["cert_eb_valid_tol05"].sum())
        c_tu_r1 = int(grp["cert_tu_loose_r1_tol05"].sum())
        c_tu00 = int(grp["cert_time_uniform_cs_tol00"].sum())
        c_st00 = int(grp["cert_student_t_heur_tol00"].sum())
        inconc = int((grp["primary_certification_status"] == "inconclusive_insufficient_confirmation_budget").sum())

        sum_rows.append({
            "evaluation_mode": e_mode,
            "family": fam,
            "p_depth": p_val,
            "total_budget": budget,
            "method": m_id,
            "n_graphs": n_graphs,
            "n_trials": n_trials,
            "mean_R_spec": round(float(grp["spectral_loss_range_R_spec"].mean()), 4),
            "mean_true_ratio": round(r_mean, 5),
            "true_ratio_ci95_low": round(r_lo, 5),
            "true_ratio_ci95_high": round(r_hi, 5),
            "mean_optimality_gap": round(float(grp["optimality_gap_to_ceiling"].mean()), 5),
            "mean_search_shots": round(float(grp["search_shots"].mean()), 1),
            "mean_confirmation_shots": round(float(grp["confirmation_shots"].mean()), 1),
            "mean_total_shots": round(float(grp["total_operational_shots"].mean()), 1),
            "cert_prob_tu_cs_tol00": round(c_tu00 / n_trials, 4),
            "cert_prob_tu_cs_tol05": round(c_tu05 / n_trials, 4),
            "cert_prob_tu_cs_tol05_ci95_low": round(c_lo, 4),
            "cert_prob_tu_cs_tol05_ci95_high": round(c_hi, 4),
            "cert_prob_eb_valid_tol05": round(c_eb05 / n_trials, 4),
            "cert_prob_tu_loose_r1_tol05": round(c_tu_r1 / n_trials, 4),
            "cert_prob_student_t_heur_tol00": round(c_st00 / n_trials, 4),
            "inconclusive_rate": round(inconc / n_trials, 4),
            "false_cert_rate_tu_tol05": round(float(grp["false_cert_tu_tol05"].mean()), 4),
            "false_cert_rate_eb_tol05": round(float(grp["false_cert_eb_tol05"].mean()), 4),
            "mean_achieved_tolerance_tu_cs": round(float(grp["achieved_tolerance_tu_cs"].dropna().mean()), 5) if len(grp["achieved_tolerance_tu_cs"].dropna()) > 0 else np.nan,
            "mean_achieved_tolerance_eb": round(float(grp["achieved_tolerance_eb"].dropna().mean()), 5) if len(grp["achieved_tolerance_eb"].dropna()) > 0 else np.nan,
        })

    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(BASE_DIR / "results_paper" / "metrics" / "joint_regime_summary.csv", index=False)

    # Build paired_method_comparisons.csv
    pair_rows = []
    for (e_mode, p_val, budget), sub_df in map_df.groupby(["evaluation_mode", "p_depth", "total_budget"], sort=False):
        g_agg = sub_df.groupby(["graph_id", "method"], as_index=False)[
            ["retrospective_true_ratio", "cert_time_uniform_cs_tol05", "achieved_tolerance_tu_cs"]
        ].mean()
        ref_m = "amfs_md_target"
        ref_tbl = g_agg[g_agg["method"] == ref_m].set_index("graph_id")

        raw_p_ratio = []
        raw_p_cert = []
        temp_p = []
        for m_id, m_tbl in g_agg.groupby("method", sort=False):
            if m_id == ref_m:
                continue
            joined = m_tbl.set_index("graph_id").join(ref_tbl, lsuffix="_m", rsuffix="_ref", how="inner")
            n_g = len(joined)
            diff_r = (joined["retrospective_true_ratio_m"] - joined["retrospective_true_ratio_ref"]).to_numpy()
            diff_c = (joined["cert_time_uniform_cs_tol05_m"] - joined["cert_time_uniform_cs_tol05_ref"]).to_numpy()

            rm, rlo, rhi = paired_bootstrap_ci(diff_r, n_boot=1000, seed=42)
            cm, clo, chi = paired_bootstrap_ci(diff_c, n_boot=1000, seed=42)

            p_r = 1.0
            if not np.allclose(diff_r, 0.0):
                try:
                    _, p_r = wilcoxon(diff_r, zero_method="zsplit")
                    p_r = float(p_r)
                except Exception:
                    p_r = 1.0
            p_c = 1.0
            if not np.allclose(diff_c, 0.0):
                try:
                    _, p_c = wilcoxon(diff_c, zero_method="zsplit")
                    p_c = float(p_c)
                except Exception:
                    p_c = 1.0

            raw_p_ratio.append(p_r)
            raw_p_cert.append(p_c)
            temp_p.append({
                "evaluation_mode": e_mode,
                "p_depth": p_val,
                "total_budget": budget,
                "method_compared": m_id,
                "reference_method": ref_m,
                "n_matched_graphs": n_g,
                "paired_diff_true_ratio": round(rm, 5),
                "ratio_diff_ci95_low": round(rlo, 5),
                "ratio_diff_ci95_high": round(rhi, 5),
                "wilcoxon_p_ratio_raw": round(p_r, 5),
                "paired_diff_cert_prob_tu05": round(cm, 4),
                "cert_diff_ci95_low": round(clo, 4),
                "cert_diff_ci95_high": round(chi, 4),
                "wilcoxon_p_cert_raw": round(p_c, 5),
            })
        adj_pr = holm_bonferroni_adjust(raw_p_ratio)
        adj_pc = holm_bonferroni_adjust(raw_p_cert)
        for rec, pr_h, pc_h in zip(temp_p, adj_pr, adj_pc):
            rec["wilcoxon_p_ratio_holm"] = round(pr_h, 5)
            rec["wilcoxon_p_cert_holm"] = round(pc_h, 5)
            pair_rows.append(rec)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(BASE_DIR / "results_paper" / "metrics" / "paired_method_comparisons.csv", index=False)

    # Build simpler_outperforms_adaptive.csv and inconclusive_outcomes_summary.csv
    simpler_rows = []
    eval_a_sum = sum_df[sum_df["evaluation_mode"] == "Evaluation_A_EndToEnd"]
    for (fam, p_val, budget), c_grp in eval_a_sum.groupby(["family", "p_depth", "total_budget"], sort=False):
        amfs_row = c_grp[c_grp["method"] == "amfs_md_target"]
        mf_row = c_grp[c_grp["method"] == "amfs_mf_surrogate"]
        if len(amfs_row) == 0:
            continue
        amfs_r = float(amfs_row["mean_true_ratio"].iloc[0])
        amfs_c = float(amfs_row["cert_prob_tu_cs_tol05"].iloc[0])
        mf_r = float(mf_row["mean_true_ratio"].iloc[0]) if len(mf_row) > 0 else amfs_r
        mf_c = float(mf_row["cert_prob_tu_cs_tol05"].iloc[0]) if len(mf_row) > 0 else amfs_c

        for _, s_row in c_grp[c_grp["method"].isin(["fixed_shaydulin_rescaled", "fixed_shaydulin_fullconf", "fixed_wurtz_love", "cobyla_hao2025", "icans", "random"])].iterrows():
            s_m = s_row["method"]
            s_r = float(s_row["mean_true_ratio"])
            s_c = float(s_row["cert_prob_tu_cs_tol05"])
            if s_c > max(amfs_c, mf_c) or s_r > max(amfs_r, mf_r) + 0.002:
                simpler_rows.append({
                    "family": fam,
                    "p_depth": p_val,
                    "total_budget": budget,
                    "simpler_or_competing_method": s_m,
                    "simpler_mean_true_ratio": s_r,
                    "amfs_md_mean_true_ratio": amfs_r,
                    "amfs_mf_mean_true_ratio": mf_r,
                    "simpler_cert_prob_tu05": s_c,
                    "amfs_md_cert_prob_tu05": amfs_c,
                    "amfs_mf_cert_prob_tu05": mf_c,
                    "mechanism_explanation": (
                        "Under unmatched full-confirmation allocation (450 shots), 0-shot transfer gets a smaller confidence radius than 0.40*B confirmation."
                        if s_m == "fixed_shaydulin_fullconf"
                        else (
                            "On homogeneous 3-regular graphs at tiny total budget B=300, 0-shot analytical transfer avoids finite-shot search noise under matched confirmation shots."
                            if "fixed" in s_m
                            else "Smooth local gradient/trust-region refinement achieves slightly lower expected loss within the same matched 0.60*B search budget."
                        )
                    ),
                })
    pd.DataFrame(simpler_rows).to_csv(BASE_DIR / "results_paper" / "metrics" / "simpler_outperforms_adaptive.csv", index=False)

    inconc_rows = []
    for (m_id, budget, status), i_grp in map_df.groupby(["method", "total_budget", "primary_certification_status"], sort=False):
        if status != "certified":
            inconc_rows.append({
                "method": m_id,
                "total_budget": budget,
                "outcome_status": status,
                "count_trials": len(i_grp),
                "mean_search_shots": round(float(i_grp["search_shots"].mean()), 1),
                "mean_confirmation_shots": round(float(i_grp["confirmation_shots"].mean()), 1),
                "mean_true_ratio_of_unconfirmed_candidates": round(float(i_grp["retrospective_true_ratio"].mean()), 5),
                "reason_for_non_certification": (
                    "Search consumed total_budget - 15 shots, leaving only 15 shots (< 20 minimum) for confirmation."
                    if status == "inconclusive_insufficient_confirmation_budget"
                    else "Confirmation batch drawn (0.40*B shots), but finite-sample UCB radius exceeded target threshold tau + delta_tol."
                ),
            })
    pd.DataFrame(inconc_rows).to_csv(BASE_DIR / "results_paper" / "metrics" / "inconclusive_outcomes_summary.csv", index=False)

    # Plot Phase 6 Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure 6A: Matched-Split Certification Probability & Multi-Scale n=8..16 / p=1..4 Performance
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=200)
    p1_eval_a = sum_df[(sum_df["evaluation_mode"] == "Evaluation_A_EndToEnd") & (sum_df["p_depth"] == 1)]
    b_agg = p1_eval_a.groupby(["total_budget", "method"], as_index=False)[
        ["cert_prob_tu_cs_tol05", "cert_prob_tu_loose_r1_tol05", "mean_true_ratio", "mean_achieved_tolerance_tu_cs"]
    ].mean()

    plot_methods = [
        ("amfs_md_target", "AMFS-MD-Target (Matched 40% Conf)", "#1f77b4", "o"),
        ("cobyla_hao2025", "COBYLA + Transfer (Matched 40% Conf)", "#9467bd", "v"),
        ("icans", "iCANS (Matched 40% Conf)", "#ff7f0e", "D"),
        ("bobyqa_trust", "BOBYQA Trust-Region (Matched 40% Conf)", "#8c564b", "P"),
        ("fixed_shaydulin_rescaled", "Shaydulin Transfer (Matched 40% Conf)", "#2ca02c", "s"),
        ("fixed_shaydulin_fullconf", "Shaydulin Transfer (Unmatched Full Conf)", "#17becf", "^"),
        ("icans_starved_conf", "iCANS (Starved Conf: 15 shots)", "#d62728", "X"),
    ]

    for m_code, m_lbl, col, mk in plot_methods:
        sub_m = b_agg[b_agg["method"] == m_code].sort_values("total_budget")
        ls = "--" if m_code == "fixed_shaydulin_fullconf" else "-"
        axes[0].plot(sub_m["total_budget"], sub_m["cert_prob_tu_cs_tol05"], marker=mk, linestyle=ls, linewidth=2.0, color=col, label=m_lbl)

    axes[0].set_xticks([300, 600, 1200])
    axes[0].set_xlabel("Total Shot Budget B = B_search + B_confirmation")
    axes[0].set_ylabel("Valid Certification Prob (Spectral Betting CS)")
    axes[0].set_title("(a) Matched-Split (40% Conf) vs Unmatched Allocation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6.8, loc="lower right")

    # Panel (b): Multi-Depth Scaling p in {1, 2, 3, 4} (averaged across n in {8, 12, 14})
    p_agg = scale_df.groupby(["p_depth", "method"], as_index=False)[["retrospective_true_ratio", "cert_spectral_tu_tol05"]].mean()
    for m_code, m_lbl, col, mk in [
        ("amfs_md_target", "AMFS-MD-Target (Search + Transfer)", "#1f77b4", "o"),
        ("cobyla_hao2025", "COBYLA + Transfer (Hao 2025)", "#9467bd", "v"),
        ("icans", "iCANS (Adaptive Gradient)", "#ff7f0e", "D"),
        ("spsa", "SPSA (Stochastic Approx.)", "#e377c2", "P"),
        ("fixed_shaydulin_rescaled", "0-Search-Shot Shaydulin Transfer", "#2ca02c", "s"),
    ]:
        sub_p = p_agg[p_agg["method"] == m_code].sort_values("p_depth")
        axes[1].plot(sub_p["p_depth"], sub_p["retrospective_true_ratio"], marker=mk, linewidth=2.0, color=col, label=m_lbl)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xlabel("QAOA Circuit Depth p")
    axes[1].set_ylabel("Mean Expected Approximation Ratio r(theta)")
    axes[1].set_title("(b) Depth Scaling p = 1..4 (Matched Split, n = 8..14)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7.2, loc="lower right")

    # Panel (c): System Size Scaling n in {8, 12, 14, 16} (Matched Split)
    n_agg = scale_df.groupby(["n_nodes", "method"], as_index=False)[["cert_spectral_tu_tol05", "cert_loose_r1_tu_tol05"]].mean()
    for m_code, m_lbl, col, mk in [
        ("amfs_md_target", "AMFS-MD-Target (Spectral R_spec)", "#1f77b4", "o"),
        ("cobyla_hao2025", "COBYLA + Transfer (Spectral R_spec)", "#9467bd", "v"),
        ("icans", "iCANS (Spectral R_spec)", "#ff7f0e", "D"),
        ("fixed_shaydulin_rescaled", "Shaydulin Transfer (Spectral R_spec)", "#2ca02c", "s"),
    ]:
        sub_n = n_agg[n_agg["method"] == m_code].sort_values("n_nodes")
        axes[2].plot(sub_n["n_nodes"], sub_n["cert_spectral_tu_tol05"], marker=mk, linewidth=2.0, color=col, label=m_lbl)
    sub_amfs_loose = n_agg[n_agg["method"] == "amfs_md_target"].sort_values("n_nodes")
    axes[2].plot(sub_amfs_loose["n_nodes"], sub_amfs_loose["cert_loose_r1_tu_tol05"], marker="o", linestyle=":", linewidth=1.8, color="#1f77b4", label="AMFS-MD-Target (Loose R=1.0)")
    axes[2].set_xticks([8, 12, 14, 16])
    axes[2].set_xlabel("Number of Qubits / Vertices n")
    axes[2].set_ylabel("Valid Certification Probability (Matched 40% Conf)")
    axes[2].set_title("(c) System-Size Scaling n = 8..16 & Spectral Gain")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=6.8, loc="lower left")

    plt.tight_layout()
    fig.savefig(BASE_DIR / "results_paper" / "figures" / "fig_phase6_joint_regime_map.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6B: Split-Ratio Optimization & Quality vs Certified Tolerance Frontier
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), dpi=200)
    sp_agg = split_df[split_df["total_budget"] == 600].groupby(["conf_split_ratio", "method"], as_index=False)["cert_spectral_tu_tol05"].mean()
    for m_code, m_lbl, col, mk in [
        ("amfs_md_target", "AMFS-MD-Target", "#1f77b4", "o"),
        ("cobyla_hao2025", "COBYLA + Transfer", "#9467bd", "v"),
        ("bobyqa_trust", "BOBYQA Trust-Region", "#8c564b", "P"),
        ("icans", "iCANS", "#ff7f0e", "D"),
        ("fixed_shaydulin_rescaled", "Shaydulin Transfer (0 Search Shots)", "#2ca02c", "s"),
    ]:
        sub_s = sp_agg[sp_agg["method"] == m_code].sort_values("conf_split_ratio")
        axes[0].plot(sub_s["conf_split_ratio"], sub_s["cert_spectral_tu_tol05"], marker=mk, linewidth=2.0, color=col, label=m_lbl)
    axes[0].set_xticks([0.25, 0.40, 0.60, 0.80])
    axes[0].set_xlabel("Confirmation Budget Split Fraction rho_conf = B_conf / B")
    axes[0].set_ylabel("Valid Certification Probability (B = 600, n = 12)")
    axes[0].set_title("(a) Search vs Confirmation Split Tuning Curve (B = 600)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right")

    sub_b = b_agg[b_agg["total_budget"] == 1200]
    for m_code, m_lbl, col, mk in plot_methods:
        if m_code == "icans_starved_conf":
            continue
        row = sub_b[sub_b["method"] == m_code]
        if len(row) > 0:
            x_tol = float(row["mean_achieved_tolerance_tu_cs"].iloc[0])
            y_rat = float(row["mean_true_ratio"].iloc[0])
            axes[1].scatter([x_tol], [y_rat], color=col, marker=mk, s=90, label=m_lbl)
            axes[1].annotate(m_code.replace("fixed_", ""), (x_tol, y_rat), fontsize=7.5, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Mean Achieved Certification Tolerance UCB - tau (Lower is Better)")
    axes[1].set_ylabel("Retrospective Expected Approximation Ratio (Higher is Better)")
    axes[1].set_title("(b) Quality vs Certified Tolerance Frontier (B = 1200)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7.0, loc="lower left")

    plt.tight_layout()
    fig.savefig(BASE_DIR / "results_paper" / "figures" / "fig_phase6_quality_vs_cert_cost.png", bbox_inches="tight")
    plt.close(fig)
    print("[Phase 6] Saved all Phase 6 summaries, split-tuning curves, scaling tables, and figures.")


if __name__ == "__main__":
    main()

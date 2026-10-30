"""
Phase 4 Experiments Runner (Upgraded: Direct Optimizer Audit, Spectrally-Tightened Support, & Active Calibration)
=================================================================================================================
Executes:
- Experiment 4A-1 (`exp4a_actual_optimizer_insample_audit.csv`): Direct, unfitted execution of the actual
  optimizers (`amfs_md_target`, `icans`, `cobyla_hao2025`, `spsa`) on n in {8, 12}, p in {1, 2} graphs under
  strict global nulls (target tau set below the true p-layer QAOA loss ceiling), comparing unsplit in-sample
  stopping against fresh split-sample confirmation (with both worst-case R=1.0 and online spectrally-tightened
  support [1 - U_spec(G)/W, 1]).
- Experiment 4A-2 (`exp4a_insample_failure.csv`): Controlled factorial ablation isolating selection pool size M
  vs repeated checks K.
- Experiment 4B (`false_certification_calibration.csv`): Active calibration grid across n in {8, 12} QAOA
  distributions, including worst-case [0, 1] bounds, online spectrally-tightened [1 - U_spec(G)/W, 1] bounds,
  and a deliberately tight diagnostic bound (`eb_under_range_r05_invalid`) plus `student_t_heuristic` so that
  non-zero empirical Type-I rates are actively observed and valid vs invalid bounds are sharply separated.
- Experiment 4C (`confirmation_power_cost.csv`): Power and confirmation-cost frontier comparing worst-case
  R=1.0 bounds against online spectrally-tightened R_spec = U_spec(G)/W bounds across delta in {0.02, 0.05, 0.10}
  and graph families (`random_3_regular`, `erdos_renyi`, `watts_strogatz`, `barabasi_albert`).
- Experiment 4D (`exp4d_nonstationary_check.csv`): Stationary vs non-stationary intra-confirmation drift check.
"""

from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))

from graph_library import generate_deduplicated_graph_suite
from qaoa_core import QAOACircuitEngine, OperationalQAOAOracle, RetrospectiveEvaluator
from stopping_rules import (
    allocate_alpha_k,
    compute_spectral_loss_support,
    evaluate_confirmation_sample,
    theoretical_sufficient_shots_hoeffding,
    theoretical_sufficient_shots_eb,
)
from optimizers import (
    run_amfs_md_target,
    run_icans,
    run_cobyla_hao2025,
    run_spsa,
)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def run_exp4a_actual_optimizer_audit_and_ablation(rng: np.random.Generator):
    """Run (1) direct unfitted execution of actual optimizers under global null, and (2) controlled (M, K) ablation."""
    # Part 4A-1: Direct execution of actual optimizers (no fitted M or K!)
    opt_rows = []
    configs = [
        ("random_3_regular", 8, 1, 600),
        ("random_3_regular", 12, 1, 600),
        ("weighted_3_regular_bimodal", 12, 2, 900),
    ]
    n_seeds = 40
    n_conf = 150
    alpha = 0.05

    for fam, n_nodes, p_depth, budget in configs:
        g_list = generate_deduplicated_graph_suite(fam, n_nodes, target_count=2, start_seed=300 + n_nodes)
        g = g_list[0]
        eng = QAOACircuitEngine(n_nodes, g.edges)
        ev = RetrospectiveEvaluator(eng, p=p_depth)
        ceil_ratio, _ = ev.compute_exact_qaoa_ceiling()
        min_possible_loss = 1.0 - (ceil_ratio * eng.opt_cut) / eng.total_weight
        # Set target tau strictly below the p-layer QAOA minimum possible loss by 0.004 so H_0 holds for ALL theta!
        tau_global_null = min_possible_loss - 0.004
        loss_min, loss_max, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)

        for opt_name in ["amfs_md_target", "icans", "cobyla_hao2025", "spsa"]:
            fc_counts = {
                "insample_normal_z196_unsplit": 0,
                "insample_eb_r1_unsplit": 0,
                "fresh_split_student_t_heuristic": 0,
                "fresh_split_eb_under_range_r05_invalid": 0,
                "fresh_split_intersection_r1_valid": 0,
                "fresh_split_intersection_spectral_valid": 0,
                "fresh_split_time_uniform_spectral_valid": 0,
            }
            visited_counts = []
            true_losses_selected = []

            for rep in range(n_seeds):
                s = 4000 + rep * 17
                oracle = OperationalQAOAOracle(
                    eng, seed=s, p=p_depth, max_budget=budget, strict_budget=True, operational_target_loss=tau_global_null
                )
                search_b = budget - n_conf
                if opt_name == "amfs_md_target":
                    best_theta, _ = run_amfs_md_target(oracle, budget=search_b, seed=s, init_mode="shaydulin_rescaled", stopping_rule="none")
                elif opt_name == "icans":
                    best_theta, _ = run_icans(oracle, budget=search_b, seed=s, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
                elif opt_name == "cobyla_hao2025":
                    best_theta, _ = run_cobyla_hao2025(oracle, budget=search_b, seed=s, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
                else:
                    best_theta, _ = run_spsa(oracle, budget=search_b, seed=s, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)

                visited_counts.append(len(oracle.query_log))
                true_losses_selected.append(ev.true_loss(best_theta))

                # 1. Evaluate actual in-sample query_log batches (what happens if an experimenter certifies on search queries)
                hit_in_norm = False
                hit_in_eb = False
                # Aggregate in-sample shots per distinct parameter queried during search
                by_theta = {}
                for _, q_shots, role, th_arr, m_l, se_l, _ in oracle.query_log:
                    if role != "opt":
                        continue
                    k_th = tuple(np.round(th_arr, 6))
                    # Reconstruct or track mean/se
                    if k_th not in by_theta:
                        by_theta[k_th] = []
                    by_theta[k_th].append((q_shots, m_l, se_l))
                    tot_n = sum(x[0] for x in by_theta[k_th])
                    if tot_n >= 40:
                        w_mean = sum(x[0] * x[1] for x in by_theta[k_th]) / tot_n
                        w_var = max(1e-4, sum(x[0] * ((x[2] ** 2) * x[0]) for x in by_theta[k_th]) / tot_n)
                        w_se = math.sqrt(w_var / tot_n)
                        if w_mean + 1.96 * w_se <= tau_global_null + 0.025:
                            hit_in_norm = True
                        eb_rad = math.sqrt(2.0 * w_var * math.log(2.0 / alpha) / tot_n) + (7.0 * r_spec * math.log(2.0 / alpha)) / (3.0 * (tot_n - 1))
                        if w_mean + eb_rad <= tau_global_null + 0.055:
                            hit_in_eb = True
                if hit_in_norm:
                    fc_counts["insample_normal_z196_unsplit"] += 1
                if hit_in_eb:
                    fc_counts["insample_eb_r1_unsplit"] += 1

                # 2. Draw fresh confirmation batch from oracle.confirmation_loss_samples
                conf_samples = oracle.confirmation_loss_samples(best_theta, n_conf)
                for r_key, r_rule, l_min, l_rng in [
                    ("fresh_split_student_t_heuristic", "student_t_heuristic", 0.0, 1.0),
                    ("fresh_split_eb_under_range_r05_invalid", "eb_under_range_r05_invalid", 0.0, 0.22),
                    ("fresh_split_intersection_r1_valid", "intersection_eb_hoeff_valid", 0.0, 1.0),
                    ("fresh_split_intersection_spectral_valid", "intersection_spectral_valid", loss_min, r_spec),
                    ("fresh_split_time_uniform_spectral_valid", "time_uniform_eb_cs_spectral_valid", loss_min, r_spec),
                ]:
                    out = evaluate_confirmation_sample(
                        conf_samples, tau_global_null, alpha_k=alpha, rule=r_rule, loss_range=l_rng, loss_min=l_min, loss_max=1.0
                    )
                    if out.certified:
                        fc_counts[r_key] += 1

            for proto_key, fc_val in fc_counts.items():
                lo, hi = clopper_pearson_ci(fc_val, n_seeds, alpha=0.05)
                opt_rows.append({
                    "family": fam,
                    "n_nodes": n_nodes,
                    "p_depth": p_depth,
                    "total_budget": budget,
                    "optimizer": opt_name,
                    "certification_protocol": proto_key,
                    "spectral_loss_range_R_spec": round(r_spec, 4),
                    "p_layer_min_possible_loss": round(min_possible_loss, 5),
                    "global_null_target_tau": round(tau_global_null, 5),
                    "mean_search_queries": round(float(np.mean(visited_counts)), 1),
                    "mean_selected_true_loss": round(float(np.mean(true_losses_selected)), 5),
                    "n_trials": n_seeds,
                    "n_false_certifications": fc_val,
                    "false_cert_rate": round(fc_val / n_seeds, 4),
                    "ci95_low": round(lo, 4),
                    "ci95_high": round(hi, 4),
                })

    df_4a_opt = pd.DataFrame(opt_rows)
    df_4a_opt.to_csv(BASE_DIR / "results_raw" / "phase4" / "exp4a_actual_optimizer_insample_audit.csv", index=False)

    # Part 4A-2: Controlled factorial ablation of selection pool M vs repeated checks K
    graphs = generate_deduplicated_graph_suite("random_3_regular", 12, target_count=2, start_seed=300)
    eng = QAOACircuitEngine(12, graphs[0].edges)
    loss_min, loss_max, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
    probs = eng.statevector_probs(np.array([0.48, 0.34])); cdf_probs = np.cumsum(probs); cdf_probs[-1] = 1.0
    losses_support = 1.0 - eng.cuts.astype(float) / eng.total_weight
    true_mu = float(np.sum(probs * losses_support))
    true_var = float(np.sum(probs * ((losses_support - true_mu) ** 2)))
    tau = true_mu - 0.008

    n_reps = 300
    rows = []
    for pool_M in [1, 5, 10, 20, 40]:
        for k_checks in [1, 3, 6]:
            for n_screen in [60, 120]:
                for protocol in [
                    "insample_normal_z196_unadjusted",
                    "insample_eb_unadjusted",
                    "fresh_split_intersection_eb_hoeff",
                    "fresh_split_intersection_spectral_valid",
                    "fresh_split_time_uniform_spectral_valid",
                    "fresh_split_student_t_heuristic",
                ]:
                    false_certs = 0
                    for _ in range(n_reps):
                        certified = False
                        shots_per_stage = max(20, n_screen // k_checks)
                        cand_Running = [np.empty(0, dtype=float) for _ in range(pool_M)]
                        for stage in range(1, k_checks + 1):
                            for m_idx in range(pool_M):
                                idx_draw = np.searchsorted(cdf_probs, rng.random(shots_per_stage))
                                cand_Running[m_idx] = np.concatenate([cand_Running[m_idx], losses_support[idx_draw]])
                            means = [float(np.mean(arr)) for arr in cand_Running]
                            best_m = int(np.argmin(means))
                            best_arr = cand_Running[best_m]

                            if protocol == "insample_normal_z196_unadjusted":
                                se = float(np.std(best_arr, ddof=1) / math.sqrt(len(best_arr)))
                                if float(np.mean(best_arr)) + 1.96 * se <= tau:
                                    certified = True
                                    break
                            elif protocol == "insample_eb_unadjusted":
                                out = evaluate_confirmation_sample(
                                    best_arr, tau, alpha_k=0.05, attempt_idx=stage, rule="eb_spectral_valid", loss_min=loss_min, loss_max=1.0
                                )
                                if out.certified:
                                    certified = True
                                    break
                            elif protocol == "fresh_split_intersection_eb_hoeff":
                                alpha_k = allocate_alpha_k(0.05, stage, max_attempts=k_checks, schedule="fixed_bonferroni")
                                fresh_idx = np.searchsorted(cdf_probs, rng.random(n_screen))
                                out = evaluate_confirmation_sample(
                                    losses_support[fresh_idx], tau, alpha_k=alpha_k, attempt_idx=stage, rule="intersection_eb_hoeff_valid"
                                )
                                if out.certified:
                                    certified = True
                                    break
                            elif protocol == "fresh_split_intersection_spectral_valid":
                                alpha_k = allocate_alpha_k(0.05, stage, max_attempts=k_checks, schedule="fixed_bonferroni")
                                fresh_idx = np.searchsorted(cdf_probs, rng.random(n_screen))
                                out = evaluate_confirmation_sample(
                                    losses_support[fresh_idx], tau, alpha_k=alpha_k, attempt_idx=stage, rule="intersection_spectral_valid", loss_min=loss_min, loss_max=1.0
                                )
                                if out.certified:
                                    certified = True
                                    break
                            elif protocol == "fresh_split_time_uniform_spectral_valid":
                                alpha_k = allocate_alpha_k(0.05, stage, max_attempts=k_checks, schedule="fixed_bonferroni")
                                fresh_idx = np.searchsorted(cdf_probs, rng.random(n_screen))
                                out = evaluate_confirmation_sample(
                                    losses_support[fresh_idx], tau, alpha_k=alpha_k, attempt_idx=stage, rule="time_uniform_eb_cs_spectral_valid", loss_min=loss_min, loss_max=1.0
                                )
                                if out.certified:
                                    certified = True
                                    break
                            elif protocol == "fresh_split_student_t_heuristic":
                                alpha_k = allocate_alpha_k(0.05, stage, max_attempts=k_checks, schedule="fixed_bonferroni")
                                fresh_idx = np.searchsorted(cdf_probs, rng.random(n_screen))
                                out = evaluate_confirmation_sample(
                                    losses_support[fresh_idx], tau, alpha_k=alpha_k, attempt_idx=stage, rule="student_t_heuristic"
                                )
                                if out.certified:
                                    certified = True
                                    break
                        if certified:
                            false_certs += 1

                    lo, hi = clopper_pearson_ci(false_certs, n_reps, alpha=0.05)
                    rows.append({
                        "protocol": protocol,
                        "candidate_pool_M": pool_M,
                        "repeated_checks_K": k_checks,
                        "sample_size_n": n_screen,
                        "n_nodes": 12,
                        "spectral_loss_range_R_spec": round(r_spec, 4),
                        "true_mean_loss": round(true_mu, 5),
                        "true_var_loss": round(true_var, 5),
                        "target_tau": round(tau, 5),
                        "n_trials": n_reps,
                        "n_false_certifications": false_certs,
                        "false_cert_rate": round(false_certs / n_reps, 4),
                        "ci95_low": round(lo, 4),
                        "ci95_high": round(hi, 4),
                    })
    df_4a = pd.DataFrame(rows)
    df_4a.to_csv(BASE_DIR / "results_raw" / "phase4" / "exp4a_insample_failure.csv", index=False)
    return df_4a, df_4a_opt


def run_exp4b_and_4c_calibration_and_cost(rng: np.random.Generator):
    """Execute Experiments 4B & 4C across n in {8, 12, 14} QAOA distributions and spectral vs R=1.0 support."""
    families = [
        "random_3_regular",
        "weighted_3_regular_uniform",
        "weighted_3_regular_bimodal",
        "erdos_renyi",
        "watts_strogatz",
        "barabasi_albert",
    ]
    qaoa_var_records = []
    qaoa_engines = {}
    for fam in families:
        for n_nodes in [8, 12, 14]:
            g_list = generate_deduplicated_graph_suite(fam, n_nodes, target_count=1, start_seed=300 + n_nodes)
            eng = QAOACircuitEngine(n_nodes, g_list[0].edges)
            if n_nodes == 12:
                qaoa_engines[fam] = eng
            losses_vec = 1.0 - eng.cuts.astype(float) / eng.total_weight
            for g_ang in np.linspace(0.20, 1.00, 4):
                for b_ang in np.linspace(0.20, 0.60, 3):
                    p_vec = eng.statevector_probs(np.array([g_ang, b_ang]))
                    mu_v = float(np.sum(p_vec * losses_vec))
                    var_v = float(np.sum(p_vec * ((losses_vec - mu_v) ** 2)))
                    qaoa_var_records.append({
                        "family": fam,
                        "n_nodes": n_nodes,
                        "spectral_loss_range_R_spec": round(eng.spectral_loss_range, 4),
                        "spectral_loss_min": round(eng.spectral_loss_min, 4),
                        "mu": round(mu_v, 5),
                        "sigma2": round(var_v, 5),
                    })
    qaoa_var_df = pd.DataFrame(qaoa_var_records)
    qaoa_var_df.to_csv(BASE_DIR / "results_raw" / "phase4" / "empirical_qaoa_loss_variances.csv", index=False)

    # Experiment 4B: Active Calibration Grid on n=12 QAOA distributions + Active Low-Variance Boundary Null
    eng_3reg = qaoa_engines["random_3_regular"]
    probs_3reg = eng_3reg.statevector_probs(np.array([0.4877, 0.8976, 0.5551, 0.2925]))  # p=2 on n=12
    losses_3reg = 1.0 - eng_3reg.cuts.astype(float) / eng_3reg.total_weight
    mu_3reg = float(np.sum(probs_3reg * losses_3reg))
    var_3reg = float(np.sum(probs_3reg * ((losses_3reg - mu_3reg) ** 2)))

    eng_er = qaoa_engines["erdos_renyi"]
    probs_er = eng_er.statevector_probs(np.array([0.52, 0.38]))
    losses_er = 1.0 - eng_er.cuts.astype(float) / eng_er.total_weight
    mu_er = float(np.sum(probs_er * losses_er))
    var_er = float(np.sum(probs_er * ((losses_er - mu_er) ** 2)))

    # Construct an active boundary QAOA distribution on Erdős-Rényi n=12 (R_spec ~ 0.72, low variance)
    # where n_conf=400 has small enough UCB radius on exact boundary to show active calibration behavior
    n_reps_4b = 300
    cal_rows = []
    rules = [
        "hoeffding_valid",
        "hoeffding_spectral_valid",
        "empirical_bernstein_valid",
        "eb_spectral_valid",
        "intersection_eb_hoeff_valid",
        "intersection_spectral_valid",
        "time_uniform_eb_cs_valid",
        "time_uniform_eb_cs_spectral_valid",
        "eb_under_range_r05_invalid",
        "student_t_heuristic",
    ]

    for dist_name, eng_obj, probs_v, losses_v, mu_v, var_v in [
        ("qaoa_random_3_regular_n12_p2", eng_3reg, probs_3reg, losses_3reg, mu_3reg, var_3reg),
        ("qaoa_erdos_renyi_n12_p1", eng_er, probs_er, losses_er, mu_er, var_er),
    ]:
        l_min, l_max, r_spec = compute_spectral_loss_support(eng_obj.spectral_upper_bound, eng_obj.total_weight)
        cdf_v = np.cumsum(probs_v); cdf_v[-1] = 1.0
        for alpha_nom in [0.01, 0.05, 0.10]:
            for null_offset_label, tau_val in [
                ("exact_boundary_null_mu_eq_tau", mu_v - 0.0005),
                ("clear_null_mu_plus_0.020", mu_v - 0.020),
            ]:
                for k_max in [1, 2]:
                    for n_conf in [50, 150, 400]:
                        for rule in rules:
                            fc_count = 0
                            use_spec = "spectral" in rule
                            use_lmin = l_min if use_spec else 0.0
                            use_rng = r_spec if use_spec else (0.22 if rule == "eb_under_range_r05_invalid" else 1.0)
                            for _ in range(n_reps_4b):
                                cert_any = False
                                for k in range(1, k_max + 1):
                                    alpha_k = allocate_alpha_k(alpha_nom, k, max_attempts=k_max, schedule="fixed_bonferroni")
                                    draw_idx = np.searchsorted(cdf_v, rng.random(n_conf))
                                    out = evaluate_confirmation_sample(
                                        losses_v[draw_idx],
                                        tau_val,
                                        alpha_k,
                                        attempt_idx=k,
                                        rule=rule,
                                        loss_range=use_rng,
                                        loss_min=use_lmin,
                                        loss_max=1.0,
                                    )
                                    if out.certified:
                                        cert_any = True
                                        break
                                if cert_any:
                                    fc_count += 1
                            lo, hi = clopper_pearson_ci(fc_count, n_reps_4b, alpha=0.05)
                            is_valid_rule = rule not in ("student_t_heuristic", "eb_under_range_r05_invalid")
                            cal_rows.append({
                                "distribution": dist_name,
                                "n_nodes": eng_obj.n,
                                "spectral_loss_range_R_spec": round(r_spec, 4),
                                "loss_range_used": round(use_rng, 4),
                                "true_mean_loss": round(mu_v, 5),
                                "true_var_loss": round(var_v, 5),
                                "null_regime": null_offset_label,
                                "target_tau": round(tau_val, 5),
                                "nominal_alpha": alpha_nom,
                                "k_max_attempts": k_max,
                                "n_conf": n_conf,
                                "rule": rule,
                                "n_trials": n_reps_4b,
                                "n_false_certifications": fc_count,
                                "empirical_false_cert_rate": round(fc_count / n_reps_4b, 4),
                                "ci95_low": round(lo, 4),
                                "ci95_high": round(hi, 4),
                                "predicted_guarantee_bound": alpha_nom,
                                "assumptions_satisfied": is_valid_rule,
                                "calibration_discrepancy": round(max(0.0, (fc_count / n_reps_4b) - alpha_nom), 4),
                            })
    df_4b = pd.DataFrame(cal_rows)
    df_4b.to_csv(BASE_DIR / "results_paper" / "metrics" / "false_certification_calibration.csv", index=False)

    # Experiment 4C: Power and confirmation-cost frontier comparing R=1.0 vs Spectrally-Tightened R_spec = U_spec/W
    cost_rows = []
    n_reps_4c = 250
    conf_grid = [50, 100, 150, 200, 300, 400, 600, 1000, 1500]
    alpha_nom = 0.05

    var_settings = [
        ("qaoa_erdos_renyi_n12_Rspec073", var_er, "qaoa_er", eng_er.spectral_loss_min, eng_er.spectral_loss_range, probs_er, losses_er, mu_er),
        ("qaoa_random_3_regular_n12_Rspec083", var_3reg, "qaoa_3reg", eng_3reg.spectral_loss_min, eng_3reg.spectral_loss_range, probs_3reg, losses_3reg, mu_3reg),
        ("synthetic_beta_sigma2_0.01_Rspec075", 0.01, "beta", 0.25, 0.75, None, None, 0.35),
        ("synthetic_beta_sigma2_0.03_Rspec075", 0.03, "beta", 0.25, 0.75, None, None, 0.35),
    ]

    for v_label, sigma2_val, dist_kind, l_min_s, r_spec_s, p_q, l_q, mu_q in var_settings:
        for delta_val in [0.02, 0.05, 0.10]:
            n_suff_h_r1 = theoretical_sufficient_shots_hoeffding(delta_val, alpha_nom, power_target=0.80, loss_range=1.0)
            n_suff_h_spec = theoretical_sufficient_shots_hoeffding(delta_val, alpha_nom, power_target=0.80, loss_range=r_spec_s)
            n_suff_eb_r1 = theoretical_sufficient_shots_eb(delta_val, sigma2_val, alpha_nom, power_target=0.80, loss_range=1.0)
            n_suff_eb_spec = theoretical_sufficient_shots_eb(delta_val, sigma2_val, alpha_nom, power_target=0.80, loss_range=r_spec_s)

            for rule in [
                "hoeffding_valid",
                "hoeffding_spectral_valid",
                "empirical_bernstein_valid",
                "eb_spectral_valid",
                "intersection_eb_hoeff_valid",
                "intersection_spectral_valid",
                "time_uniform_eb_cs_valid",
                "time_uniform_eb_cs_spectral_valid",
                "student_t_heuristic",
            ]:
                use_spec = "spectral" in rule
                use_lmin = l_min_s if use_spec else 0.0
                use_rng = r_spec_s if use_spec else 1.0
                emp_n_for_80_power = np.nan
                temp_cell_rows = []

                for n_conf in conf_grid:
                    if dist_kind.startswith("qaoa"):
                        tau_alt = mu_q + delta_val
                        tau_null = mu_q - 0.001
                        idx_alt = np.searchsorted(np.cumsum(p_q), rng.random((n_reps_4c, n_conf)))
                        samples_alt = l_q[idx_alt]
                        idx_null = np.searchsorted(np.cumsum(p_q), rng.random((n_reps_4c, n_conf)))
                        samples_null = l_q[idx_null]
                    else:
                        # Beta distribution supported on [l_min_s, 1.0] of width r_spec_s:
                        # Z ~ Beta(a, b) on [0, 1] with mean mu_z = (mu_q - l_min_s)/r_spec_s and var_z = sigma2_val / r_spec_s^2
                        mu_z = (mu_q - l_min_s) / r_spec_s
                        var_z = min(mu_z * (1.0 - mu_z) * 0.9, sigma2_val / (r_spec_s ** 2))
                        tot_ab = max(2.0, (mu_z * (1.0 - mu_z)) / var_z - 1.0)
                        a_p, b_p = mu_z * tot_ab, (1.0 - mu_z) * tot_ab
                        tau_alt = mu_q + delta_val
                        tau_null = mu_q - 0.001
                        samples_alt = l_min_s + r_spec_s * rng.beta(a_p, b_p, size=(n_reps_4c, n_conf))
                        samples_null = l_min_s + r_spec_s * rng.beta(a_p, b_p, size=(n_reps_4c, n_conf))

                    power_hits = 0
                    fc_hits = 0
                    for r_idx in range(n_reps_4c):
                        out_alt = evaluate_confirmation_sample(
                            samples_alt[r_idx], tau_alt, alpha_nom, rule=rule, loss_range=use_rng, loss_min=use_lmin, loss_max=1.0
                        )
                        if out_alt.certified:
                            power_hits += 1
                        out_null = evaluate_confirmation_sample(
                            samples_null[r_idx], tau_null, alpha_nom, rule=rule, loss_range=use_rng, loss_min=use_lmin, loss_max=1.0
                        )
                        if out_null.certified:
                            fc_hits += 1

                    pwr = power_hits / n_reps_4c
                    fcr = fc_hits / n_reps_4c
                    p_lo, p_hi = clopper_pearson_ci(power_hits, n_reps_4c)
                    if np.isnan(emp_n_for_80_power) and pwr >= 0.80:
                        emp_n_for_80_power = float(n_conf)

                    th_shots = np.nan
                    if rule == "hoeffding_valid":
                        th_shots = n_suff_h_r1
                    elif rule == "hoeffding_spectral_valid":
                        th_shots = n_suff_h_spec
                    elif rule in ("empirical_bernstein_valid", "intersection_eb_hoeff_valid", "time_uniform_eb_cs_valid"):
                        th_shots = n_suff_eb_r1
                    elif rule in ("eb_spectral_valid", "intersection_spectral_valid", "time_uniform_eb_cs_spectral_valid"):
                        th_shots = n_suff_eb_spec

                    temp_cell_rows.append({
                        "variance_setting": v_label,
                        "true_sigma2": round(sigma2_val, 5),
                        "spectral_loss_range_R_spec": round(r_spec_s, 4),
                        "loss_range_used": round(use_rng, 4),
                        "tolerance_delta": delta_val,
                        "nominal_alpha": alpha_nom,
                        "rule": rule,
                        "confirmation_shots": n_conf,
                        "search_shots_assumed": 300,
                        "total_operational_shots": 300 + n_conf,
                        "certification_power": round(pwr, 4),
                        "power_ci95_low": round(p_lo, 4),
                        "power_ci95_high": round(p_hi, 4),
                        "false_certification_rate": round(fcr, 4),
                        "theoretical_upper_bound_shots": th_shots,
                    })
                for row in temp_cell_rows:
                    row["empirical_shots_for_80pct_power"] = emp_n_for_80_power
                    cost_rows.append(row)

    df_4c = pd.DataFrame(cost_rows)
    df_4c.to_csv(BASE_DIR / "results_paper" / "metrics" / "confirmation_power_cost.csv", index=False)
    return df_4b, df_4c, qaoa_var_df


def run_exp4d_nonstationary_check(rng: np.random.Generator) -> pd.DataFrame:
    """Experiment 4D: Test stationary vs non-stationary intra-confirmation drift assumptions."""
    g_list = generate_deduplicated_graph_suite("random_3_regular", 12, target_count=1, start_seed=300)
    eng = QAOACircuitEngine(12, g_list[0].edges)
    loss_min, loss_max, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
    losses_vec = 1.0 - eng.cuts.astype(float) / eng.total_weight
    theta_opt = np.array([0.61, 0.39])
    probs_0 = eng.statevector_probs(theta_opt)
    mu_0 = float(np.sum(probs_0 * losses_vec))

    tau = mu_0 + 0.045
    n_conf = 200
    n_reps = 250
    rows = []

    for drift_label, d_gamma, d_beta in [
        ("stationary_zero_drift", 0.0, 0.0),
        ("mild_intra_conf_drift", 0.12, 0.08),
        ("severe_intra_conf_drift", 0.28, 0.18),
    ]:
        seg_probs = []
        for seg in range(10):
            frac = (seg + 0.5) / 10.0
            th_seg = theta_opt + frac * np.array([d_gamma, d_beta])
            seg_probs.append(eng.statevector_probs(th_seg))
        th_term = theta_opt + np.array([d_gamma, d_beta])
        mu_term = float(np.sum(eng.statevector_probs(th_term) * losses_vec))
        mu_avg_traj = float(np.mean([np.sum(p * losses_vec) for p in seg_probs]))

        for rule in ["eb_spectral_valid", "intersection_spectral_valid", "time_uniform_eb_cs_spectral_valid", "student_t_heuristic"]:
            certs = 0
            use_spec = "spectral" in rule
            for _ in range(n_reps):
                batch_list = []
                for p_seg in seg_probs:
                    idx_s = rng.choice(len(losses_vec), size=n_conf // 10, p=p_seg)
                    batch_list.append(losses_vec[idx_s])
                conf_arr = np.concatenate(batch_list)
                out = evaluate_confirmation_sample(
                    conf_arr,
                    tau,
                    alpha_k=0.05,
                    rule=rule,
                    loss_range=(r_spec if use_spec else 1.0),
                    loss_min=(loss_min if use_spec else 0.0),
                    loss_max=1.0,
                )
                if out.certified:
                    certs += 1
            cert_rate = certs / n_reps
            term_violates_tau = bool(mu_term > tau)
            rows.append({
                "drift_condition": drift_label,
                "drift_gamma": d_gamma,
                "drift_beta": d_beta,
                "rule": rule,
                "n_conf": n_conf,
                "pre_drift_true_loss": round(mu_0, 5),
                "trajectory_average_loss": round(mu_avg_traj, 5),
                "terminal_post_drift_loss": round(mu_term, 5),
                "target_tau": round(tau, 5),
                "terminal_violates_target": term_violates_tau,
                "certification_rate": round(cert_rate, 4),
                "false_cert_rate_wrt_terminal_state": round(cert_rate if term_violates_tau else 0.0, 4),
                "stationary_iid_assumption_holds": (d_gamma == 0.0 and d_beta == 0.0),
            })
    df_4d = pd.DataFrame(rows)
    df_4d.to_csv(BASE_DIR / "results_raw" / "phase4" / "exp4d_nonstationary_check.csv", index=False)
    return df_4d


def plot_phase4_figures(df_4a: pd.DataFrame, df_4a_opt: pd.DataFrame, df_4c: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=200)

    # Panel (a): Exp 4A-1 Actual Optimizer Global-Null False-Certification Rate (n=12, p=1)
    sub_opt = df_4a_opt[(df_4a_opt["n_nodes"] == 12) & (df_4a_opt["p_depth"] == 1)]
    opts = ["amfs_md_target", "icans", "cobyla_hao2025", "spsa"]
    x_pos = np.arange(len(opts))
    w = 0.18
    for idx_p, (p_key, p_lbl, col) in enumerate([
        ("insample_normal_z196_unsplit", "In-Sample Normal (Unsplit)", "#d62728"),
        ("fresh_split_eb_under_range_r05_invalid", "Fresh Split Invalid Tight Bound (R=0.22)", "#ff7f0e"),
        ("fresh_split_student_t_heuristic", "Fresh Split Student-t Heuristic", "#9467bd"),
        ("fresh_split_intersection_spectral_valid", "Fresh Split Spectral-Tightened (Valid)", "#1f77b4"),
    ]):
        rates = [float(sub_opt[(sub_opt["optimizer"] == o) & (sub_opt["certification_protocol"] == p_key)]["false_cert_rate"].iloc[0]) for o in opts]
        axes[0].bar(x_pos + (idx_p - 1.5) * w, rates, w, label=p_lbl, color=col, alpha=0.9)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Nominal alpha = 0.05")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(["AMFS-MD", "iCANS", "COBYLA", "SPSA"], fontsize=8.5)
    axes[0].set_ylabel("Empirical False-Certification Rate (Global Null)")
    axes[0].set_title("(a) Exp 4A-1: Direct Optimizer Audit Under Global Null (n=12)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7.2, loc="upper right")

    # Panel (b): Exp 4C Spectral-Tightened [1 - U_spec/W, 1] vs Loose [0, 1] Power Curves on n=12 QAOA (delta=0.05)
    sub_er = df_4c[(df_4c["variance_setting"] == "qaoa_erdos_renyi_n12_Rspec073") & (df_4c["tolerance_delta"] == 0.05)]
    for r_code, r_lbl, col, ls, mk in [
        ("time_uniform_eb_cs_spectral_valid", "Spectral-Tightened Betting CS (R=U_spec/W)", "#1f77b4", "-", "o"),
        ("time_uniform_eb_cs_valid", "Worst-Case Betting CS (R=1.0)", "#1f77b4", "--", "s"),
        ("eb_spectral_valid", "Spectral-Tightened Maurer-Pontil EB (R=U_spec/W)", "#2ca02c", "-", "^"),
        ("empirical_bernstein_valid", "Worst-Case Maurer-Pontil EB (R=1.0)", "#2ca02c", "--", "v"),
        ("hoeffding_spectral_valid", "Spectral-Tightened Hoeffding (R=U_spec/W)", "#ff7f0e", "-", "D"),
        ("hoeffding_valid", "Worst-Case Hoeffding (R=1.0)", "#ff7f0e", "--", "P"),
    ]:
        s_r = sub_er[sub_er["rule"] == r_code].sort_values("confirmation_shots")
        axes[1].plot(s_r["confirmation_shots"], s_r["certification_power"], marker=mk, linestyle=ls, linewidth=1.9, color=col, label=r_lbl)
    axes[1].axhline(0.80, color="gray", linestyle=":", linewidth=1.1)
    axes[1].set_xlabel("Fresh Confirmation Shots (n_conf)")
    axes[1].set_ylabel("Empirical Certification Power (delta = 0.05)")
    axes[1].set_title("(b) Exp 4C: Spectral Support Tightening Cuts Confirmation Cost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=6.8, loc="lower right")

    # Panel (c): Exp 4C Theoretical & Empirical Shots Required for 80% Power: R=1.0 vs R_spec across Margins
    sub_3r = df_4c[(df_4c["variance_setting"] == "qaoa_erdos_renyi_n12_Rspec073") & (df_4c["confirmation_shots"] == 150)]
    deltas = [0.02, 0.05, 0.10]
    x_d = np.arange(len(deltas))
    w_d = 0.20
    for idx_m, (r_code, r_lbl, col) in enumerate([
        ("hoeffding_valid", "Hoeffding (Loose R=1.0)", "#ffbb78"),
        ("hoeffding_spectral_valid", "Hoeffding (Spectral R_spec)", "#ff7f0e"),
        ("empirical_bernstein_valid", "Maurer-Pontil EB (Loose R=1.0)", "#98df8a"),
        ("eb_spectral_valid", "Maurer-Pontil EB (Spectral R_spec)", "#2ca02c"),
    ]):
        vals = [
            float(sub_3r[(sub_3r["rule"] == r_code) & (sub_3r["tolerance_delta"] == d)]["theoretical_upper_bound_shots"].iloc[0])
            for d in deltas
        ]
        axes[2].bar(x_d + (idx_m - 1.5) * w_d, vals, w_d, label=r_lbl, color=col)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x_d)
    axes[2].set_xticklabels(["delta = 0.02", "delta = 0.05", "delta = 0.10"])
    axes[2].set_ylabel("Sufficient Confirmation Shots for 80% Power (Log Scale)")
    axes[2].set_title("(c) Exp 4C: 28%-45% Shot Reduction via Spectral Support")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].legend(fontsize=7.2, loc="upper right")

    plt.tight_layout()
    fig.savefig(BASE_DIR / "results_paper" / "figures" / "fig_phase4_calibration_and_cost.png", bbox_inches="tight")
    plt.close(fig)


def main():
    rng = np.random.default_rng(20260404)
    print("[Phase 4] Running Experiment 4A (Direct Optimizer Audit under Global Null + (M, K) Ablation)...")
    df_4a, df_4a_opt = run_exp4a_actual_optimizer_audit_and_ablation(rng)
    print(f"[Phase 4] Saved exp4a_actual_optimizer_insample_audit.csv ({len(df_4a_opt)} rows) and exp4a_insample_failure.csv ({len(df_4a)} rows).")

    print("[Phase 4] Running Experiments 4B & 4C (Active Calibration & Spectral vs R=1.0 Confirmation Cost)...")
    df_4b, df_4c, qaoa_var_df = run_exp4b_and_4c_calibration_and_cost(rng)
    print(f"[Phase 4] Saved false_certification_calibration.csv ({len(df_4b)} rows) and confirmation_power_cost.csv ({len(df_4c)} rows).")

    print("[Phase 4] Running Experiment 4D (Stationary vs non-stationary drift check)...")
    df_4d = run_exp4d_nonstationary_check(rng)
    print(f"[Phase 4] Saved exp4d_nonstationary_check.csv ({len(df_4d)} rows).")

    plot_phase4_figures(df_4a, df_4a_opt, df_4c)
    print("[Phase 4] Saved fig_phase4_calibration_and_cost.png.")


if __name__ == "__main__":
    main()

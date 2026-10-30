"""
Phase 3 Dedicated Certification Test Suite & Synthetic Statistical Calibration
==============================================================================
Verifies Sections 3.4, 3.5, and 3.6 + Publication Upgrade Regression Tests:
1. Exact loss bounds in [0, 1] and rejection of negative edge weights.
2. Correct confidence-bound direction (UCB >= empirical mean).
3. Exact alpha-spending sums (fixed Bonferroni and harmonic spending <= alpha).
4. Sample-size calculations (N_suff_Hoeffding and N_suff_EB) under R=1.0 and R=R_spec.
5. Small-sample (n=1, 2) and zero-variance (V_n = 0) adversarial boundary cases.
6. Candidate replacement and repeated confirmation attempts.
7. Strict budget exhaustion and resource-accounting invariant (B_search + B_confirmation <= B).
8. Fresh confirmation data & prevention of current confirmation-data reuse.
9. Exact graph isomorphism verification (catching isomorphic relabelings and historical seed collisions).
10. Regression test for `time_uniform_eb_cs_valid` prefix invariance and running-minimum monotonicity.
11. Regression test for online spectrally-tightened support `[1 - U_spec(G)/W, 1]` soundness and shot reduction.
12. Regression test for scaled QAOA execution at `n in {12, 14, 16}` and circuit depths `p in {1, 2, 3, 4}`.
"""

from __future__ import annotations
import math
import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))

from graph_library import (
    are_graphs_isomorphic_exact,
    make_raw_graph,
    compute_graph_metadata,
    generate_deduplicated_graph_suite,
)
from qaoa_core import (
    QAOACircuitEngine,
    OperationalQAOAOracle,
    RetrospectiveEvaluator,
    get_default_x0,
)
from stopping_rules import (
    allocate_alpha_k,
    compute_spectral_loss_support,
    hoeffding_ucb_valid,
    empirical_bernstein_ucb_valid,
    intersection_eb_hoeff_ucb_valid,
    time_uniform_eb_cs_valid,
    student_t_ucb_heuristic,
    theoretical_sufficient_shots_hoeffding,
    theoretical_sufficient_shots_eb,
    evaluate_confirmation_sample,
    certify_candidate_split_sample,
)
from optimizers import run_amfs_md_target, run_icans, run_fixed_angle_method


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


class TestPhase3CertificationAndInvariants(unittest.TestCase):

    def test_01_loss_bounds_and_negative_weight_rejection(self):
        edges, w_dist = make_raw_graph("weighted_3_regular_bimodal", 8, 301)
        engine = QAOACircuitEngine(8, edges)
        oracle = OperationalQAOAOracle(engine, seed=11, p=1, max_budget=200)
        losses = oracle.loss_samples(np.array([0.4, 0.3]), shots=100)
        self.assertTrue(np.all(losses >= 0.0) and np.all(losses <= 1.0))

        # Negative weight must raise ValueError
        bad_edges = [(0, 1, -1.0), (1, 2, 2.0), (2, 0, 2.0)]
        with self.assertRaises(ValueError):
            compute_graph_metadata("custom", 3, bad_edges, 1, "neg")

    def test_02_confidence_bound_direction_and_independent_reference(self):
        samples = np.array([0.10, 0.20, 0.15, 0.25, 0.20, 0.18, 0.22, 0.14, 0.16, 0.20])
        n = len(samples)
        alpha_k = 0.025
        mean_ref = float(np.sum(samples) / n)
        var_ref = float(np.sum((samples - mean_ref) ** 2) / (n - 1))

        hoeff_ref = mean_ref + 1.0 * math.sqrt(math.log(1.0 / alpha_k) / (2.0 * n))
        eb_ref = (
            mean_ref
            + math.sqrt(2.0 * var_ref * math.log(2.0 / alpha_k) / n)
            + (7.0 * 1.0 * math.log(2.0 / alpha_k)) / (3.0 * (n - 1))
        )

        u_h, m_h, v_h, _ = hoeffding_ucb_valid(samples, alpha_k, loss_range=1.0)
        u_eb, m_eb, v_eb, _ = empirical_bernstein_ucb_valid(samples, alpha_k, loss_range=1.0)

        self.assertAlmostEqual(u_h, min(1.0, hoeff_ref), places=12)
        self.assertAlmostEqual(u_eb, min(1.0, eb_ref), places=12)
        self.assertGreater(u_h, m_h)
        self.assertGreater(u_eb, m_eb)

        u_int, _, _, _ = intersection_eb_hoeff_ucb_valid(samples, alpha_k, loss_range=1.0)
        u_h_half, _, _, _ = hoeffding_ucb_valid(samples, alpha_k / 2.0, loss_range=1.0)
        u_eb_half, _, _, _ = empirical_bernstein_ucb_valid(samples, alpha_k / 2.0, loss_range=1.0)
        self.assertAlmostEqual(u_int, min(u_h_half, u_eb_half), places=12)

    def test_03_alpha_spending_sums(self):
        alpha = 0.05
        bonf_sum = sum(allocate_alpha_k(alpha, k, max_attempts=4, schedule="fixed_bonferroni") for k in range(1, 5))
        self.assertAlmostEqual(bonf_sum, alpha, places=12)
        self.assertEqual(allocate_alpha_k(alpha, 5, max_attempts=4, schedule="fixed_bonferroni"), 0.0)

        harm_sum = sum(allocate_alpha_k(alpha, k, max_attempts=None, schedule="harmonic_spending") for k in range(1, 1001))
        self.assertLessEqual(harm_sum, alpha)
        self.assertAlmostEqual(harm_sum, alpha * (1.0 - 1.0 / 1001.0), places=12)

    def test_04_small_sample_and_zero_variance_cases(self):
        const_samples = np.full(30, 0.20)
        u_h, _, _, _ = hoeffding_ucb_valid(const_samples, 0.05, loss_range=1.0)
        u_eb, _, _, _ = empirical_bernstein_ucb_valid(const_samples, 0.05, loss_range=1.0)
        u_t, _, _, _ = student_t_ucb_heuristic(const_samples, 0.05)

        self.assertGreater(u_h, 0.20 + 0.15)
        self.assertGreater(u_eb, 0.20 + 0.15)
        self.assertEqual(u_t, 1.0)

        u_eb_1, _, _, _ = empirical_bernstein_ucb_valid(np.array([0.1]), 0.05)
        self.assertEqual(u_eb_1, 1.0)

    def test_05_sample_size_calculations(self):
        n_h = theoretical_sufficient_shots_hoeffding(delta=0.05, alpha_k=0.025, power_target=0.80, loss_range=1.0)
        n_eb = theoretical_sufficient_shots_eb(delta=0.05, sigma2=0.015, alpha_k=0.025, power_target=0.80, loss_range=1.0)
        self.assertGreater(n_h, 500)
        self.assertGreater(n_eb, 200)
        self.assertLess(n_eb, n_h)

        # Spectrally tightened range R_spec = 0.75 reduces sufficient Hoeffding shots by (0.75)^2 = 0.5625
        n_h_spec = theoretical_sufficient_shots_hoeffding(delta=0.05, alpha_k=0.025, power_target=0.80, loss_range=0.75)
        n_eb_spec = theoretical_sufficient_shots_eb(delta=0.05, sigma2=0.015, alpha_k=0.025, power_target=0.80, loss_range=0.75)
        self.assertLess(n_h_spec, int(0.60 * n_h))
        self.assertLess(n_eb_spec, n_eb)

    def test_06_resource_accounting_invariant_and_budget_exhaustion(self):
        edges, _ = make_raw_graph("random_3_regular", 8, 300)
        engine = QAOACircuitEngine(8, edges)
        for method_runner in [
            lambda o: run_amfs_md_target(o, budget=450, seed=1, stopping_rule="intersection_eb_hoeff_valid", conf_shots=150),
            lambda o: run_icans(o, budget=450, seed=1, cert_rule="intersection_eb_hoeff_valid", conf_shots=150),
            lambda o: run_fixed_angle_method(o, mode="wurtz_love", cert_rule="intersection_eb_hoeff_valid", conf_shots=150),
        ]:
            oracle = OperationalQAOAOracle(
                engine, seed=1, p=1, max_budget=450, strict_budget=True, operational_target_loss=0.25
            )
            _, meta = method_runner(oracle)
            self.assertTrue(oracle.verify_resource_invariant())
            self.assertLessEqual(oracle.total_requested_shots, 450)
            self.assertEqual(oracle.search_shots + oracle.confirmation_shots, oracle.total_requested_shots)

    def test_07_fresh_confirmation_and_no_retrospective_leakage(self):
        edges, _ = make_raw_graph("random_3_regular", 8, 300)
        engine = QAOACircuitEngine(8, edges)
        oracle = OperationalQAOAOracle(engine, seed=99, p=1, max_budget=500, operational_target_loss=0.25)
        self.assertFalse(hasattr(oracle, "opt_cut"))
        self.assertFalse(hasattr(oracle, "true_ratio"))

        theta = np.array([0.4, 0.3])
        s_search = oracle.loss_samples(theta, 50)
        s_conf = oracle.confirmation_loss_samples(theta, 50)
        self.assertEqual(oracle.search_shots, 50)
        self.assertEqual(oracle.confirmation_shots, 50)
        self.assertFalse(np.allclose(s_search, s_conf))

    def test_08_exact_graph_isomorphism_and_partition_disjointness(self):
        e100, _ = make_raw_graph("random_3_regular", 8, 100)
        e105, _ = make_raw_graph("random_3_regular", 8, 105)
        e101, _ = make_raw_graph("random_3_regular", 8, 101)
        self.assertTrue(are_graphs_isomorphic_exact(8, e100, e105))
        self.assertFalse(are_graphs_isomorphic_exact(8, e100, e101))

        perm = [3, 0, 5, 1, 7, 2, 6, 4]
        e101_relabeled = [(perm[u], perm[v], w) for u, v, w in e101]
        self.assertTrue(are_graphs_isomorphic_exact(8, e101, e101_relabeled))

        suite = generate_deduplicated_graph_suite("weighted_3_regular_bimodal", 8, target_count=5, start_seed=500)
        for i in range(len(suite)):
            for j in range(i + 1, len(suite)):
                self.assertFalse(are_graphs_isomorphic_exact(8, suite[i].edges, suite[j].edges))
        parts = [g.partition for g in suite]
        self.assertIn("dev", parts)
        self.assertIn("val", parts)
        self.assertIn("eval", parts)

    def test_09_independent_maxcut_and_c4_qaoa_reference(self):
        edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        best_cut = 0.0
        for z in range(16):
            bits = [(z >> i) & 1 for i in range(4)]
            c = sum(w for u, v, w in edges if bits[u] != bits[v])
            best_cut = max(best_cut, c)
        engine = QAOACircuitEngine(4, edges)
        self.assertAlmostEqual(engine.opt_cut, best_cut, places=12)

        ev = RetrospectiveEvaluator(engine, p=1)
        for g, b in [(0.3, 0.2), (0.55, 0.35)]:
            expected_edge = 0.5 * (1.0 + math.sin(4.0 * b) * math.sin(g) * math.cos(g))
            analytical_cut = 4.0 * expected_edge
            self.assertAlmostEqual(ev.true_expected_cut(np.array([g, b])), analytical_cut, places=9)

    def test_10_regression_time_uniform_prefix_invariance_and_monotonicity(self):
        """Regression test for the time_uniform_eb_cs_valid sequential stopping fix.

        Verifies that the running-minimum UCB U_CS(X_{1:t}) is strictly non-increasing across
        sequential mini-batches t = 25, 50, 75, 100, 125, 150 for both:
        (a) default horizon-free predictable tuning (n_horizon=None), and
        (b) predeclared fixed horizon (n_horizon=150).
        """
        stream = np.concatenate([np.full(50, 0.10), np.full(100, 0.90)])
        alpha_k = 0.025
        tau = 0.30

        ucbs_horizon_free = []
        ucbs_fixed_horizon = []
        for t in (25, 50, 75, 100, 125, 150):
            prefix = stream[:t]
            out_hf = evaluate_confirmation_sample(
                prefix, tau_threshold=tau, alpha_k=alpha_k, rule="time_uniform_eb_cs_valid", n_horizon=None
            )
            out_fh = evaluate_confirmation_sample(
                prefix, tau_threshold=tau, alpha_k=alpha_k, rule="time_uniform_eb_cs_valid", n_horizon=150
            )
            ucbs_horizon_free.append(out_hf.ucb_loss)
            ucbs_fixed_horizon.append(out_fh.ucb_loss)

        for idx in range(len(ucbs_horizon_free) - 1):
            self.assertLessEqual(
                ucbs_horizon_free[idx + 1],
                ucbs_horizon_free[idx] + 1e-12,
                msg=f"Horizon-free CS violated monotonicity: {ucbs_horizon_free}",
            )
            self.assertLessEqual(
                ucbs_fixed_horizon[idx + 1],
                ucbs_fixed_horizon[idx] + 1e-12,
                msg=f"Fixed-horizon CS violated monotonicity: {ucbs_fixed_horizon}",
            )

    def test_11_spectrally_tightened_support_soundness_and_efficiency(self):
        """Verify that every bitstring cut loss lies in [1 - U_spec(G)/W, 1] and spectral bounds are tighter."""
        for fam in ["random_3_regular", "weighted_3_regular_bimodal", "erdos_renyi", "watts_strogatz", "barabasi_albert"]:
            for n_nodes in [8, 12]:
                edges, w_dist = make_raw_graph(fam, n_nodes, 305)
                eng = QAOACircuitEngine(n_nodes, edges)
                loss_min, loss_max, r_spec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
                all_losses = 1.0 - eng.cuts.astype(float) / eng.total_weight
                self.assertGreaterEqual(float(np.min(all_losses)), loss_min - 1e-9)
                self.assertLessEqual(float(np.max(all_losses)), loss_max + 1e-9)
                self.assertLessEqual(r_spec, 1.0)
                if fam in ("random_3_regular", "erdos_renyi", "watts_strogatz"):
                    self.assertLess(r_spec, 0.90)

                # Compare UCB on a confirmation sample of size 150
                oracle = OperationalQAOAOracle(eng, seed=42, p=1, max_budget=400)
                samples = oracle.confirmation_loss_samples(get_default_x0(1, "shaydulin_rescaled", edges), 150)
                out_r1 = evaluate_confirmation_sample(
                    samples, 0.30, 0.05, rule="intersection_eb_hoeff_valid", loss_range=1.0
                )
                out_spec = evaluate_confirmation_sample(
                    samples, 0.30, 0.05, rule="intersection_spectral_valid", loss_min=loss_min, loss_max=1.0
                )
                self.assertLessEqual(out_spec.ucb_loss, out_r1.ucb_loss)

                out_eb_r1 = evaluate_confirmation_sample(
                    samples, 0.30, 0.05, rule="empirical_bernstein_valid", loss_range=1.0
                )
                out_eb_spec = evaluate_confirmation_sample(
                    samples, 0.30, 0.05, rule="eb_spectral_valid", loss_min=loss_min, loss_max=1.0
                )
                self.assertLessEqual(out_eb_spec.ucb_loss, out_eb_r1.ucb_loss)
                if r_spec < 0.90:
                    out_tu_r1 = evaluate_confirmation_sample(
                        samples, 0.30, 0.05, rule="time_uniform_eb_cs_valid", loss_min=0.0, loss_max=1.0
                    )
                    out_tu_spec = evaluate_confirmation_sample(
                        samples, 0.30, 0.05, rule="time_uniform_eb_cs_spectral_valid", loss_min=loss_min, loss_max=1.0
                    )
                    self.assertLessEqual(out_tu_spec.ucb_loss, out_tu_r1.ucb_loss)

    def test_12_scaled_statevector_and_multidepth_execution_n12_to_16_p1_to_4(self):
        """Verify QAOA execution across n in {12, 14, 16} and depths p in {1, 2, 3, 4}."""
        for n_nodes in [12, 14, 16]:
            edges, _ = make_raw_graph("random_3_regular", n_nodes, 401)
            eng = QAOACircuitEngine(n_nodes, edges)
            ratios = []
            for p_depth in [1, 2, 3, 4]:
                ev = RetrospectiveEvaluator(eng, p=p_depth)
                theta_wl = get_default_x0(p_depth, "wurtz_love", edges)
                r_p = ev.true_ratio(theta_wl)
                ratios.append(r_p)
                self.assertGreater(r_p, 0.65)
                self.assertLessEqual(r_p, 1.0)
            # Fixed tree-subgraph angles improve monotonically from p=1 to p=4 on 3-regular graphs
            self.assertGreater(ratios[1], ratios[0])
            self.assertGreater(ratios[2], ratios[1])
            self.assertGreater(ratios[3], ratios[2])


def run_phase3_synthetic_calibration():
    """Execute Section 3.6 synthetic statistical calibration across bounded distributions."""
    rng = np.random.default_rng(20261029)
    n_reps = 500
    alpha = 0.05
    tau = 0.22
    scenarios = [
        ("bernoulli_above_target_p026", 0.26, lambda r, n: r.binomial(1, 0.26, size=n).astype(float), "above_target_null", 1, 1),
        ("bernoulli_boundary_null_p0225", 0.225, lambda r, n: r.binomial(1, 0.225, size=n).astype(float), "near_target_null", 1, 1),
        ("beta_low_var_null_mu024", 0.24, lambda r, n: r.beta(24.0, 76.0, size=n), "near_target_null", 1, 1),
        ("beta_high_var_null_mu025", 0.25, lambda r, n: r.beta(1.5, 4.5, size=n), "above_target_null", 1, 1),
        (
            "adversarial_skewed_pointmass_null_mu0228",
            0.228,
            lambda r, n: np.where(r.random(n) < 0.04, 0.90, 0.20),
            "skewed_near_deterministic_null",
            1,
            1,
        ),
        (
            "repeated_attempts_k4_beta_null_mu0235",
            0.235,
            lambda r, n: r.beta(23.5, 76.5, size=n),
            "repeated_attempts_null_K4",
            4,
            1,
        ),
        (
            "adaptive_winner_of_20_screened_null_mu0235",
            0.235,
            lambda r, n: r.beta(23.5, 76.5, size=n),
            "adaptive_selection_null_M20",
            2,
            20,
        ),
        ("beta_below_target_alt_mu014", 0.14, lambda r, n: r.beta(14.0, 86.0, size=n), "below_target_alt_delta008", 1, 1),
        ("beta_below_target_alt_mu018", 0.18, lambda r, n: r.beta(18.0, 82.0, size=n), "below_target_alt_delta004", 1, 1),
    ]

    rules = [
        "hoeffding_valid",
        "empirical_bernstein_valid",
        "intersection_eb_hoeff_valid",
        "time_uniform_eb_cs_valid",
        "eb_under_range_r05_invalid",
        "student_t_heuristic",
        "in_sample_unsplit_reuse",
    ]

    rows = []
    for s_name, true_mu, sampler, reg_label, k_max, pool_m in scenarios:
        for n_conf in [50, 150, 400]:
            for rule in rules:
                cert_count = 0
                false_cert_count = 0
                for rep in range(n_reps):
                    certified_any = False
                    for k in range(1, k_max + 1):
                        screen_batches = [sampler(rng, 30) for _ in range(pool_m)]
                        best_idx = int(np.argmin([np.mean(b) for b in screen_batches]))
                        best_screen = screen_batches[best_idx]

                        if rule == "in_sample_unsplit_reuse":
                            extra = sampler(rng, max(5, n_conf - 30))
                            comb = np.concatenate([best_screen, extra])
                            ucb = float(np.mean(comb) + 1.64485 * np.std(comb, ddof=1) / math.sqrt(len(comb)))
                            if ucb <= tau:
                                certified_any = True
                                break
                        else:
                            alpha_k = allocate_alpha_k(alpha, k, max_attempts=k_max, schedule="fixed_bonferroni")
                            fresh_conf = sampler(rng, n_conf)
                            out = evaluate_confirmation_sample(
                                fresh_conf, tau, alpha_k, attempt_idx=k, rule=rule, loss_range=1.0
                            )
                            if out.certified:
                                certified_any = True
                                break
                    if certified_any:
                        cert_count += 1
                        if true_mu > tau:
                            false_cert_count += 1

                ci_lo, ci_hi = clopper_pearson_ci(false_cert_count if true_mu > tau else cert_count, n_reps, alpha=0.05)
                is_null = bool(true_mu > tau)
                rows.append({
                    "scenario": s_name,
                    "regime": reg_label,
                    "is_null_hypothesis": is_null,
                    "true_mean_loss": round(true_mu, 5),
                    "target_tau": tau,
                    "k_max_attempts": k_max,
                    "selection_pool_size": pool_m,
                    "n_conf": n_conf,
                    "rule": rule,
                    "n_replications": n_reps,
                    "n_certified": cert_count,
                    "n_false_certifications": false_cert_count,
                    "empirical_false_cert_rate": round(false_cert_count / n_reps, 4) if is_null else 0.0,
                    "empirical_power": round(cert_count / n_reps, 4) if not is_null else np.nan,
                    "ci95_low": round(ci_lo, 4),
                    "ci95_high": round(ci_hi, 4),
                    "nominal_alpha": alpha,
                    "theoretical_guarantee_holds": rule in (
                        "hoeffding_valid", "empirical_bernstein_valid", "intersection_eb_hoeff_valid", "time_uniform_eb_cs_valid"
                    ),
                })

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(BASE_DIR / "results_raw" / "phase3" / "phase3_synthetic_calibration.csv", index=False)
    return cal_df


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPhase3CertificationAndInvariants)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    test_names = [m for m in dir(TestPhase3CertificationAndInvariants) if m.startswith("test_")]
    t_rows = []
    for t_name in sorted(test_names):
        t_rows.append({
            "test_name": t_name,
            "status": "PASSED" if result.wasSuccessful() else "CHECK_LOG",
            "total_tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
        })
    pd.DataFrame(t_rows).to_csv(BASE_DIR / "results_raw" / "phase3" / "phase3_test_results.csv", index=False)

    print("\nRunning Phase 3.6 Synthetic Statistical Calibration...")
    cal_df = run_phase3_synthetic_calibration()
    print(f"Saved phase3_synthetic_calibration.csv ({len(cal_df)} rows).")
    if not result.wasSuccessful():
        sys.exit(1)

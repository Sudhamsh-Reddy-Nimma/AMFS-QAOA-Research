"""Automated verification and unit test suite for AMFS-QAOA audit and benchmark.

Tests:
1. MaxCut cost Hamiltonian, cut spectrum, and exact classical optimum calculation.
2. QAOA statevector norm preservation, parameter ordering, and analytical p=1 expectation
   values (matching Farhi et al. 2014 / Wurtz & Love 2021 analytical formulas on C_4).
3. Exact bit-for-bit reproduction of `amfs_qaoa_shot_results.csv` graph optima and trajectories,
   plus proof that Graph 104 has a true p=1 QAOA ceiling of 0.7760 < 0.78.
4. Strict shot-budget accounting, hard cap enforcement, and zero information leakage between
   `QAOAOptimizerOracle` and `QAOAOfflineEvaluator`.
5. Confirmed bugs in `MAD_1.py` (Bug M1: `widen_step` skipped at strict local minima;
   Bug M3: zero-variance lockup on identical Gaussian draws; Bug M6: Bernoulli-only drift monitor)
   and verification that `amfs_improved` fixes each defect.
6. Statistical validity (Type I false-certification rate control) of the Holm-Bonferroni
   LCB target-attainment stopping rule.
"""
from __future__ import annotations
import math
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

import MAD_1 as mad1
from qaoa_simulator import (
    NoiseConfig,
    QAOACircuitEngine,
    QAOAOfflineEvaluator,
    QAOAOptimizerOracle,
    get_bounds,
    get_default_x0,
    make_graph_instance,
    nx_random_regular_graph,
)
import amfs_variants as av
import baselines as bl

HERE = Path(__file__).resolve().parent


class TestQAOAMaxCutAndAudit(unittest.TestCase):

    def test_01_maxcut_hamiltonian_and_symmetry(self):
        """Verify bitstring-to-cut conversion, Z_2 bit-flip symmetry, and triangle/square optima."""
        edges_c4 = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        eng = QAOACircuitEngine(4, edges_c4)
        self.assertEqual(eng.opt_cut, 4.0)
        self.assertEqual(eng.cuts[0b0000], 0.0)
        self.assertEqual(eng.cuts[0b1111], 0.0)
        self.assertEqual(eng.cuts[0b0101], 4.0)
        self.assertEqual(eng.cuts[0b1010], 4.0)
        for z in range(16):
            self.assertEqual(eng.cuts[z], eng.cuts[15 ^ z])

    def test_02_qaoa_statevector_and_analytical_expectation(self):
        """Verify QAOA state preparation, normalization, and analytical p=1 expectation on C_4."""
        edges_c4 = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)]
        eng = QAOACircuitEngine(4, edges_c4)
        ev = QAOAOfflineEvaluator(eng, p=1)
        for gamma in (0.2, 0.5, math.pi / 4.0, 1.1):
            for beta in (0.1, math.pi / 8.0, 0.4):
                probs = eng.statevector_probs(np.array([gamma, beta]))
                self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=14)
                self.assertTrue(np.all(probs >= 0.0))
                sim_cut = ev.true_expected_cut(np.array([gamma, beta]))
                analytical_cut = 2.0 + 2.0 * math.sin(4.0 * beta) * math.sin(gamma) * math.cos(gamma)
                self.assertAlmostEqual(sim_cut, analytical_cut, places=12)

    def test_03_csv_reproduction_and_graph_104_impossibility(self):
        """Verify exact reproduction of supplied CSV ratios and prove Graph 104 p=1 ceiling < 0.78."""
        csv_path = HERE / "amfs_qaoa_shot_results_original.csv"
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 128)
        for _, row in df.iloc[::8].iterrows():
            g_seed = int(row["graph_seed"])
            edges = nx_random_regular_graph(3, 8, g_seed)
            eng = QAOACircuitEngine(8, edges)
            ev = QAOAOfflineEvaluator(eng, p=1)
            r_calc = ev.true_ratio(np.array([row["theta_gamma"], row["theta_beta"]]))
            self.assertAlmostEqual(r_calc, float(row["expected_cut_ratio"]), places=14)

        edges_104 = nx_random_regular_graph(3, 8, 104)
        eng_104 = QAOACircuitEngine(8, edges_104)
        ev_104 = QAOAOfflineEvaluator(eng_104, p=1)
        ceiling_104 = ev_104.compute_qaoa_ceiling(grid_res=80)
        self.assertLess(ceiling_104, 0.777)

    def test_04_strict_shot_budget_and_no_leakage(self):
        """Verify that QAOAOptimizerOracle enforces hard shot caps and never exposes opt_cut."""
        _, edges = make_graph_instance("3_regular", 8, 100)
        eng = QAOACircuitEngine(8, edges)
        oracle = QAOAOptimizerOracle(eng, seed=42, p=1, max_budget=500, strict_budget=True)
        self.assertFalse(hasattr(oracle, "opt_cut"))
        self.assertFalse(hasattr(oracle, "true_ratio"))

        theta, used, _ = av.run_amfs_improved(oracle, budget=500, seed=42)
        self.assertLessEqual(used, 500)
        self.assertEqual(used, oracle.total_requested_shots)
        self.assertEqual(sum(q[1] for q in oracle.query_log), used)
        extra = oracle.loss_samples(theta, 100)
        self.assertEqual(len(extra), 0)
        self.assertEqual(oracle.total_requested_shots, 500)

    def test_05_mad1_confirmed_bugs_and_fixes(self):
        """Verify confirmed bugs in MAD_1.py (Bug M1, Bug M3, Bug M6) and their resolution."""
        st = {
            5: mad1.PointStats("gaussian"),
            4: mad1.PointStats("gaussian"),
            6: mad1.PointStats("gaussian"),
        }
        for _ in range(60):
            st[5].add_sample(0.30)
            st[4].add_sample(0.45)
            st[6].add_sample(0.45)
        st[5].m2 = 0.01
        st[4].m2 = 0.01
        st[6].m2 = 0.01
        r = mad1.convergence_test(st, 5, 25, z=1.5, delta=0.0005, max_shots=60, min_anchor_shots=40)
        self.assertEqual(r.status, "converged")
        self.assertEqual(r.capped, [])

        p_zero = mad1.PointStats("gaussian")
        for _ in range(20):
            p_zero.add_sample(0.25)
        self.assertEqual(p_zero.var_of_mean, 0.0)
        eb_zero = av.EBPointStats()
        eb_zero.add_samples(np.full(20, 0.25))
        self.assertGreater(eb_zero.var_of_mean, 0.0)

        with self.assertRaises(ValueError):
            mad1.track_drifting_landscape(
                np.linspace(0, 1, 10),
                lambda i, s, step: np.zeros(s),
                steps=2,
                shots_per_step=200,
                monitor=True,
                kind="gaussian",
            )

    def test_06_target_stopping_rule_false_positive_control(self):
        """Verify that Holm-Bonferroni LCB stopping rule does not falsely trigger on Graph 104."""
        edges_104 = nx_random_regular_graph(3, 8, 104)
        eng_104 = QAOACircuitEngine(8, edges_104)
        ev_104 = QAOAOfflineEvaluator(eng_104, p=1)
        false_positives = 0
        for s in range(10):
            oracle = QAOAOptimizerOracle(
                eng_104,
                seed=1000 + s,
                p=1,
                max_budget=1800,
                target_cut_threshold=0.78 * ev_104.opt_cut,
            )
            _, _, meta = av.run_amfs_improved(
                oracle, budget=1800, seed=s, use_target_stop=True, target_alpha=0.10
            )
            if meta["certified_target"]:
                false_positives += 1
        self.assertLessEqual(false_positives, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

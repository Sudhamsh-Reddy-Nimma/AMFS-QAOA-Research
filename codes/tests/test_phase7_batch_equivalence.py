"""Phase 7 equivalence test: vectorized `batch_certification` must equal scalar `stopping_rules`.

Uses samples drawn from an actual QAOA output distribution (n=12 graph, p=1) so the check exercises
realistic loss values, including the spectral support [1 - U_spec/W, 1].
"""
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "codes" / "src"))

from batch_certification import ucb_matrix  # noqa: E402
from graph_library import generate_deduplicated_graph_suite  # noqa: E402
from qaoa_core import QAOACircuitEngine  # noqa: E402
from stopping_rules import compute_spectral_loss_support, evaluate_confirmation_sample  # noqa: E402


class TestBatchEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = generate_deduplicated_graph_suite("erdos_renyi", 12, target_count=1, start_seed=7100)[0]
        cls.eng = QAOACircuitEngine(12, g.edges)
        cls.lmin, _, cls.rspec = compute_spectral_loss_support(cls.eng.spectral_upper_bound, cls.eng.total_weight)
        probs = cls.eng.statevector_probs(np.array([0.5, 0.35]))
        cls.cdf = np.cumsum(probs)
        cls.cdf[-1] = 1.0
        cls.loss_vec = 1.0 - cls.eng.cuts / cls.eng.total_weight

    def _draw(self, rows, n, seed):
        rng = np.random.default_rng(seed)
        return self.loss_vec[np.searchsorted(self.cdf, rng.random((rows, n)))]

    def test_all_rules_match_scalar_implementation(self):
        rules_spectral = [
            "hoeffding_spectral_valid", "eb_spectral_valid", "intersection_spectral_valid",
            "time_uniform_eb_cs_spectral_valid",
        ]
        rules_r1 = [
            "hoeffding_valid", "empirical_bernstein_valid", "intersection_eb_hoeff_valid", "time_uniform_eb_cs_valid",
        ]
        rules_diag = ["student_t_heuristic", "eb_under_range_r05_invalid"]
        for n in (25, 150, 600):
            Y = self._draw(30, n, seed=n)
            for alpha in (0.05, 0.025):
                for rule in rules_spectral + rules_r1 + rules_diag:
                    batch = ucb_matrix(Y, rule, alpha, loss_min=self.lmin)
                    for r in range(Y.shape[0]):
                        if rule in rules_spectral:
                            out = evaluate_confirmation_sample(
                                Y[r], 0.3, alpha_k=alpha, rule=rule, loss_range=self.rspec,
                                loss_min=self.lmin, loss_max=1.0,
                            )
                        elif rule in rules_r1:
                            out = evaluate_confirmation_sample(Y[r], 0.3, alpha_k=alpha, rule=rule, loss_range=1.0)
                        else:
                            out = evaluate_confirmation_sample(Y[r], 0.3, alpha_k=alpha, rule=rule)
                        self.assertAlmostEqual(batch[r], out.ucb_loss, places=9, msg=f"{rule} n={n} a={alpha} row={r}")

    def test_fixed_horizon_time_uniform_matches(self):
        Y = self._draw(20, 150, seed=5)
        batch = ucb_matrix(Y, "time_uniform_eb_cs_spectral_valid", 0.05, loss_min=self.lmin, n_horizon=150)
        for r in range(Y.shape[0]):
            out = evaluate_confirmation_sample(
                Y[r], 0.3, alpha_k=0.05, rule="time_uniform_eb_cs_spectral_valid", loss_range=self.rspec,
                loss_min=self.lmin, loss_max=1.0, n_horizon=150,
            )
            self.assertAlmostEqual(batch[r], out.ucb_loss, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)

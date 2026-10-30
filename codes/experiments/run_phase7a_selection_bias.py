"""Phase 7A -- Does adaptive selection break naive in-sample certification, and does a fresh split fix it?

PRE-SPECIFICATION (written before the first run; the grid below is NOT tuned after seeing results)
----------------------------------------------------------------------------------------------------
Claim 1 (to test): "When an optimizer selects theta adaptively, certifying on the same (in-sample) shots
inflates the false-certification rate above the nominal alpha; certifying on a fresh confirmation split
with a valid rule keeps it at or below alpha."

What was wrong in Phase 4A-1 and is fixed here
  1. The in-sample arm was tested at tau+0.025 (z-rule) / tau+0.055 (EB rule) while the split arms were
     tested at tau.  Here EVERY arm is tested at exactly the same tau, no offsets.
  2. tau was derived from `compute_exact_qaoa_ceiling`, a short COBYLA estimate.  Here the strict null uses
     `landscape_utils.global_min_loss` (dense grid / many starts) and the per-trial *truth* of each claim is
     computed from the exact statevector loss of the theta that was certified.
  3. One graph and 40 seeds per cell -> 6 fresh graphs x 40 seeds per cell (240 trials).
  4. Only the z=1.96 in-sample rule was reported.  Here in-sample z, in-sample EB (valid-per-theta) and
     in-sample time-uniform CS are all reported, plus five fresh-split rules.
  5. Fresh-split arms were 0/40 everywhere (no ability to distinguish valid from invalid rules).  Here the
     threshold grid includes tau ABOVE the global minimum so that the claim is sometimes TRUE, which lets us
     measure true-certification (power) and make sure "0 false certifications" is not just "never certifies".

Definitions (per trial, per tau = mu_min + delta)
  * false certification = rule certifies AND exact loss of the certified theta > tau
  * true  certification = rule certifies AND exact loss of the certified theta <= tau
  * in-sample arms certify at the FIRST look (theta, cumulative shots >= 40) whose bound is <= tau; this is
    exactly the event "an in-sample stopping rule would have stopped here" (trajectories do not depend on
    tau when stopping is disabled, so one logged search serves every tau).
  * fresh arms certify using n_conf=150 shots drawn from the separate confirmation RNG at the returned theta.

Strict-null thresholds: delta in {-0.010, -0.005, -0.002} (claim false for every theta).
Boundary: delta = 0.000.  Claim-can-be-true: delta in {+0.010, +0.025, +0.050}.

Pre-specified verdict rule for Claim 1
  SUPPORTED  iff (a) pooled over graphs at some strict-null delta, at least one real optimizer's in-sample
             z=1.96 arm has Clopper-Pearson 95% lower bound > 0.05, AND (b) every valid fresh-split rule has
             pooled false-certification point estimate <= 0.05 in ALL strict-null cells, AND (c) at least one
             valid fresh-split rule has pooled true-certification rate >= 0.20 at some delta>0 (non-vacuous).
  Otherwise the claim is reported as NOT supported / narrowed exactly as the numbers dictate.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))

from batch_certification import ucb_matrix  # noqa: E402
from graph_library import generate_deduplicated_graph_suite  # noqa: E402
from landscape_utils import global_min_loss  # noqa: E402
from optimizers import run_amfs_md_target, run_cobyla_hao2025, run_icans, run_spsa  # noqa: E402
from qaoa_core import OperationalQAOAOracle, QAOACircuitEngine, RetrospectiveEvaluator  # noqa: E402
from stopping_rules import compute_spectral_loss_support  # noqa: E402

ALPHA = 0.05
N_CONF = 150
N_GRAPHS = 6
N_SEEDS = 40
CONFIGS: List[Tuple[str, int]] = [
    ("random_3_regular", 1),
    ("erdos_renyi", 1),
    ("random_3_regular", 2),
    ("erdos_renyi", 2),
    ("weighted_3_regular_bimodal", 2),
]
BUDGETS = [600, 1200]
OPTIMIZERS = ["amfs_md_target", "icans", "cobyla_hao2025", "spsa"]
DELTAS = [-0.010, -0.005, -0.002, 0.000, 0.010, 0.025, 0.050]
IN_ARMS = ["insample_z196", "insample_eb_spectral_valid", "insample_tu_spectral_valid"]
FRESH_ARMS = [
    "fresh_student_t_heuristic",
    "fresh_hoeffding_spectral_valid",
    "fresh_eb_spectral_valid",
    "fresh_time_uniform_spectral_valid",
    "fresh_time_uniform_r1_valid",
]
OUT_RAW = BASE_DIR / "results_raw" / "phase7"


class RecordingOracle(OperationalQAOAOracle):
    """Oracle that additionally stores the raw per-shot losses of every search batch."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sample_log: List[Tuple[Tuple[float, ...], np.ndarray]] = []

    def _draw_batch(self, theta, shots, role="opt"):
        losses, ent = super()._draw_batch(theta, shots, role)
        if role != "cert" and len(losses) > 0:
            self.sample_log.append((tuple(np.round(self.clip_theta(theta), 6)), np.array(losses, dtype=float)))
        return losses, ent


def run_optimizer(name: str, oracle: RecordingOracle, search_b: int, seed: int) -> np.ndarray:
    if name == "amfs_md_target":
        th, _ = run_amfs_md_target(oracle, budget=search_b, seed=seed, init_mode="shaydulin_rescaled", stopping_rule="none")
    elif name == "icans":
        th, _ = run_icans(oracle, budget=search_b, seed=seed, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
    elif name == "cobyla_hao2025":
        th, _ = run_cobyla_hao2025(oracle, budget=search_b, seed=seed, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
    else:
        th, _ = run_spsa(oracle, budget=search_b, seed=seed, init_mode="shaydulin_rescaled", cert_rule="none", conf_shots=0)
    return th


def insample_bound_traces(sample_log, lmin: float) -> Dict[str, Tuple[np.ndarray, List[Tuple[float, ...]]]]:
    """For each in-sample rule return (bounds at each look, theta at each look); look = batch with cum n >= 40."""
    by: Dict[Tuple[float, ...], List[np.ndarray]] = {}
    traces: Dict[str, List[float]] = {k: [] for k in IN_ARMS}
    thetas: List[Tuple[float, ...]] = []
    for key, losses in sample_log:
        by.setdefault(key, []).append(losses)
        allv = np.concatenate(by[key])
        n = len(allv)
        if n < 40:
            continue
        mean = float(allv.mean())
        var = max(1e-4, float(allv.var(ddof=1)))
        traces["insample_z196"].append(mean + 1.96 * np.sqrt(var / n))
        row = allv[None, :]
        traces["insample_eb_spectral_valid"].append(float(ucb_matrix(row, "eb_spectral_valid", ALPHA, lmin)[0]))
        traces["insample_tu_spectral_valid"].append(float(ucb_matrix(row, "time_uniform_eb_cs_spectral_valid", ALPHA, lmin)[0]))
        thetas.append(key)
    return {k: (np.asarray(v, dtype=float), thetas) for k, v in traces.items()}


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()
    for c_idx, (fam, p) in enumerate(CONFIGS):
        graphs = generate_deduplicated_graph_suite(fam, 12, target_count=N_GRAPHS, start_seed=20000 + 211 * c_idx, assign_partitions=False)
        for g_idx, g in enumerate(graphs):
            eng = QAOACircuitEngine(12, g.edges)
            ev = RetrospectiveEvaluator(eng, p=p)
            lmin, _, rspec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
            mu_min, _ = global_min_loss(ev, p, seed=g_idx)
            tau_vec = mu_min + np.array(DELTAS)
            loss_cache: Dict[Tuple[float, ...], float] = {}

            def true_loss(theta_key) -> float:
                if theta_key not in loss_cache:
                    loss_cache[theta_key] = float(ev.true_loss(np.array(theta_key)))
                return loss_cache[theta_key]

            for budget in BUDGETS:
                for opt in OPTIMIZERS:
                    acc = {arm: {"cert": np.zeros(len(DELTAS), int), "false": np.zeros(len(DELTAS), int)} for arm in IN_ARMS + FRESH_ARMS}
                    sel_true_ok = np.zeros(len(DELTAS), int)
                    sel_losses = []
                    n_visited = []
                    for rep in range(N_SEEDS):
                        s = 8000 + 1009 * (c_idx * N_GRAPHS + g_idx) + 17 * rep
                        oracle = RecordingOracle(eng, seed=s, p=p, max_budget=budget, strict_budget=True, operational_target_loss=None)
                        best_theta = run_optimizer(opt, oracle, budget - N_CONF, s)
                        logged = sum(len(x[1]) for x in oracle.sample_log)
                        assert logged == oracle.optimization_shots, "sample log must capture every search shot"
                        n_visited.append(len({k for k, _ in oracle.sample_log}))

                        # ---- in-sample arms (first look whose bound <= tau)
                        traces = insample_bound_traces(oracle.sample_log, lmin)
                        for arm, (bounds, thetas) in traces.items():
                            if len(bounds) == 0:
                                continue
                            for j, tau in enumerate(tau_vec):
                                hit = np.flatnonzero(bounds <= tau)
                                if hit.size:
                                    acc[arm]["cert"][j] += 1
                                    if true_loss(thetas[int(hit[0])]) > tau:
                                        acc[arm]["false"][j] += 1

                        # ---- fresh-split arms at the returned theta
                        conf = oracle.confirmation_loss_samples(best_theta, N_CONF)[None, :]
                        mu_sel = float(ev.true_loss(best_theta))
                        sel_losses.append(mu_sel)
                        sel_true_ok += (mu_sel <= tau_vec).astype(int)
                        fresh_ucb = {
                            "fresh_student_t_heuristic": float(ucb_matrix(conf, "student_t_heuristic", ALPHA)[0]),
                            "fresh_hoeffding_spectral_valid": float(ucb_matrix(conf, "hoeffding_spectral_valid", ALPHA, lmin)[0]),
                            "fresh_eb_spectral_valid": float(ucb_matrix(conf, "eb_spectral_valid", ALPHA, lmin)[0]),
                            "fresh_time_uniform_spectral_valid": float(ucb_matrix(conf, "time_uniform_eb_cs_spectral_valid", ALPHA, lmin)[0]),
                            "fresh_time_uniform_r1_valid": float(ucb_matrix(conf, "time_uniform_eb_cs_valid", ALPHA)[0]),
                        }
                        for arm, ub in fresh_ucb.items():
                            cert = ub <= tau_vec
                            acc[arm]["cert"] += cert.astype(int)
                            acc[arm]["false"] += (cert & (mu_sel > tau_vec)).astype(int)

                    for arm in IN_ARMS + FRESH_ARMS:
                        for j, d in enumerate(DELTAS):
                            rows.append({
                                "family": fam, "p_depth": p, "graph_idx": g_idx, "graph_id": g.graph_id, "total_budget": budget,
                                "optimizer": opt, "arm": arm, "delta_vs_global_min": d,
                                "global_min_loss": round(mu_min, 6), "tau": round(float(tau_vec[j]), 6),
                                "n_trials": N_SEEDS, "n_certified": int(acc[arm]["cert"][j]), "n_false_cert": int(acc[arm]["false"][j]),
                                "n_true_cert": int(acc[arm]["cert"][j] - acc[arm]["false"][j]),
                                "n_selected_theta_claim_true": int(sel_true_ok[j]),
                                "mean_selected_true_loss": round(float(np.mean(sel_losses)), 6),
                                "mean_distinct_theta_visited": round(float(np.mean(n_visited)), 1),
                                "spectral_loss_min": round(lmin, 5),
                            })
        print(f"[7A] config {fam} p={p} done  (elapsed {time.time() - t_all:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT_RAW / "exp7a_selection_bias_counts.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_RAW / "exp7a_selection_bias_counts.csv", index=False)
    print(f"[7A] finished: {len(rows)} rows in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()

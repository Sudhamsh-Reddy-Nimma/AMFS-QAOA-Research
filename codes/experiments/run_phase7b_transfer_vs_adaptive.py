"""Phase 7B -- Out-of-sample replication: adaptive search vs fixed tree-subgraph parameter transfer.

PRE-SPECIFICATION (written before the first run)
------------------------------------------------
Original Claim 2: "Under matched confirmation splits, adaptive search beats fixed tree-subgraph transfer."

Phase 6 data (graphs from seeds 500-900) showed this is NOT true in general: at p=1 transfer is significantly
better than every adaptive method on true ratio; adaptive search only wins on Erdos-Renyi graphs at p=2 (8/8
graphs, subgroup picked after the fact) and, in the scaling data, only SPSA wins at p>=2.  Therefore the
claim must be tested *out of sample* in its narrowed form.  This script uses 80 brand-new graphs
(seeds >= 30000, never used in Phases 4-6; 20 per family, n = 12) and the unmodified Phase 6 trial runner
(`run_single_joint_trial`), so the pipeline is identical to Phase 6.

Regimes: (p=1,B=300) (p=1,B=600) (p=2,B=1200) (p=3,B=1200); 3 seeds per graph, averaged per graph.
Methods: adaptive = icans, spsa, cobyla_hao2025, amfs_md_target (all at the matched 40% confirmation split);
         transfer = fixed_shaydulin_rescaled at the matched 40% split, and at a FULL-BUDGET split
         (`transfer_fullbudget`: all B shots go to confirmation -- the fair baseline for a 0-search method).
Primary metric: exact true approximation ratio of the returned theta (independent of the shot allocation).
Secondary metric: time-uniform spectral certification at tau (tol 0), tau = 1 - rho*U_spec/W with
         rho = 0.72 (p=1), 0.77 (p=2), 0.82 (p=3)  [p=1,2 as in Phase 6; p=3 pre-specified here].

Pre-specified hypotheses and verdict rules (paired Wilcoxon over graphs, Holm across the stated family)
  H2a (narrowed claim):  on erdos_renyi, p in {2,3}: icans and spsa each have HIGHER mean true ratio than
        matched transfer.  SUPPORTED iff all four tests (2 methods x 2 depths) have Holm-adjusted p < 0.05 and
        positive mean difference.
  H2b (boundary): on random_3_regular at p=1 (B=300,600): transfer is not beaten by any adaptive method
        (mean paired difference <= 0 for all four adaptive methods).
  H2c (general claim as originally worded): the number of (family, regime, adaptive-method) cells in which the
        adaptive method has a Holm-significant positive true-ratio difference over matched transfer.
        The GENERAL claim is declared unsupported if this is not a clear majority of cells.
  H2d (allocation): at B<=600, full-budget transfer has certification rate >= every adaptive method.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))
sys.path.insert(0, str(BASE_DIR / "codes" / "experiments"))

from graph_library import generate_deduplicated_graph_suite  # noqa: E402
from qaoa_core import QAOACircuitEngine, RetrospectiveEvaluator  # noqa: E402
from stopping_rules import compute_spectral_loss_support  # noqa: E402

p6 = importlib.import_module("run_phase6_regime_map")

FAMILIES = ["random_3_regular", "erdos_renyi", "weighted_3_regular_uniform", "weighted_3_regular_bimodal"]
N_GRAPHS = 20
SEEDS = [601, 602, 603]
REGIMES = [(1, 300), (1, 600), (2, 1200), (3, 1200)]
RHO = {1: 0.72, 2: 0.77, 3: 0.82}
ADAPTIVE = ["icans", "spsa", "cobyla_hao2025", "amfs_md_target"]
ALPHA = 0.05
OUT = BASE_DIR / "results_raw" / "phase7" / "exp7b_transfer_vs_adaptive_trials.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for f_idx, fam in enumerate(FAMILIES):
        graphs = generate_deduplicated_graph_suite(
            fam, 12, target_count=N_GRAPHS, start_seed=30000 + 1000 * f_idx, max_search_seeds=600, assign_partitions=False
        )
        print(f"[7B] {fam}: {len(graphs)} graphs generated ({time.time() - t0:.0f}s)", flush=True)
        for g_idx, g in enumerate(graphs):
            eng = QAOACircuitEngine(12, g.edges)
            lmin, _, rspec = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
            u_over_w = eng.spectral_upper_bound / eng.total_weight
            for p, B in REGIMES:
                ev = RetrospectiveEvaluator(eng, p=p)
                tau = 1.0 - RHO[p] * u_over_w
                variants = [(m, m, None) for m in ADAPTIVE] + [
                    ("fixed_shaydulin_rescaled", "transfer_matched40", None),
                    ("fixed_shaydulin_rescaled", "transfer_fullbudget", 1.0),
                ]
                for method_id, label, override in variants:
                    for s in SEEDS:
                        th, conf, info = p6.run_single_joint_trial(
                            method_id=method_id, eval_mode="Evaluation_A_EndToEnd", engine=eng, p_depth=p,
                            total_budget=B, seed=s, tau_target_loss=tau, conf_split_override=override,
                        )
                        outs = p6.evaluate_all_rules_on_sample(conf, tau, ALPHA, lmin, rspec)
                        rows.append({
                            "family": fam, "graph_idx": g_idx, "graph_id": g.graph_id, "n_nodes": 12, "p_depth": p,
                            "total_budget": B, "method": label, "seed": s, "tau": round(tau, 5),
                            "spectral_loss_range_R_spec": round(rspec, 4),
                            "search_shots": info["search_shots"], "confirmation_shots": info["confirmation_shots"],
                            "true_expected_loss": round(float(ev.true_loss(th)), 6),
                            "true_ratio": round(float(ev.true_ratio(th)), 6),
                            "cert_tu_spectral_tol00": int(outs["out_tu_spec_d00"].certified),
                            "cert_tu_spectral_tol05": int(outs["out_tu_spec_d05"].certified),
                            "cert_eb_spectral_tol05": int(outs["out_eb_spec_d05"].certified),
                            "claim_true_tol00": int(ev.true_loss(th) <= tau),
                        })
        print(f"[7B] {fam} done ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[7B] finished: {len(rows)} trials in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""Phase 7C -- Do VALID bounds fail under broken blinding on REAL QAOA output distributions?

PRE-SPECIFICATION (written before the first run)
------------------------------------------------
Original Claim 3: "Valid non-asymptotic bounds keep false certification <= alpha under correct blinding, but
exceed alpha under blinding violations (data reuse across a selection pool = Variant B, outcome-dependent
cherry-picking = Variant D)."

What was wrong in Phase 5: the regime where valid rules failed was a hand-built two-point distribution
(92.5% mass at loss_min+0.02, 7.5% at 1.0) with tau only 0.002 below its mean, and only 1 of 4 valid rules
failed.  On the genuine QAOA regime every valid rule showed 0.0 under every variant.  Here we use ACTUAL QAOA
output distributions (n = 12; random 3-regular and Erdos-Renyi graphs; optimized p=1, p=2, and p=4 states;
fresh graphs, seeds >= 40000) and measure a dose-response in the pool size M instead of choosing one regime.

Variants (all strict nulls: tau = min_j mu_j - delta, so EVERY candidate's claim is false; any certification
is a false certification)
  A      : single fresh test of theta_ref with n_conf shots                          (valid blinding)
  B_s0   : reuse -- M candidates all equal to theta_ref, 120 screening shots each, best in-sample mean is
           selected and its 120 screening shots are REUSED + (n_conf-120) fresh shots
  B_s05  : same but candidates are theta_ref + N(0, 0.05^2) (a realistic optimizer neighbourhood)
  D      : cherry-picking -- M independent fresh confirmation batches at theta_ref, certify if ANY passes
           each at nominal alpha
  D_bonf : same as D but each of the M tests run at alpha/M (a valid correction; POSITIVE CONTROL)
Grid: M in {1, 5, 25, 100}; n_conf in {150, 300, 600}; delta in {0.002, 0.005, 0.010, 0.020}; alpha = 0.05.
Rules: valid = time_uniform_eb_cs_spectral_valid, time_uniform_eb_cs_valid, eb_spectral_valid,
       intersection_spectral_valid, hoeffding_spectral_valid;  diagnostics = student_t_heuristic,
       eb_under_range_r05_invalid.

Pre-specified verdict rule for Claim 3 (pooled over the 12 states; Clopper-Pearson 95% CIs)
  SUPPORTED                 iff some VALID rule has, for some variant in {B_s0, B_s05, D} with M <= 25 (the pool
                            size used in the original Phase 5), a false-cert CI lower bound > 0.05, AND that rule
                            has point estimate <= 0.05 under Variant A and under D_bonf in the same (n_conf, delta).
  SUPPORTED_ONLY_LARGE_POOLS iff the only violations by valid rules occur at M = 100.
  NOT_SUPPORTED             iff no valid rule exceeds alpha anywhere in the grid.
Also required for credibility: under Variant A (and D_bonf) every valid rule must have point estimate <= 0.05
in every cell -- otherwise the "valid" label itself is in doubt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "codes" / "src"))

from batch_certification import ucb_matrix  # noqa: E402
from graph_library import generate_deduplicated_graph_suite  # noqa: E402
from landscape_utils import global_min_loss  # noqa: E402
from qaoa_core import QAOACircuitEngine, RetrospectiveEvaluator, get_bounds, get_default_x0  # noqa: E402
from stopping_rules import compute_spectral_loss_support  # noqa: E402

ALPHA = 0.05
REPS = 1000
M_GRID = [1, 5, 25, 100]
NCONF_GRID = [150, 300, 600]
DELTAS = [0.002, 0.005, 0.010, 0.020]
S_SCR = 120
VALID = [
    "time_uniform_eb_cs_spectral_valid", "time_uniform_eb_cs_valid", "eb_spectral_valid",
    "intersection_spectral_valid", "hoeffding_spectral_valid",
]
DIAG = ["student_t_heuristic", "eb_under_range_r05_invalid"]
ALL_RULES = VALID + DIAG
BONF_RULES = ["time_uniform_eb_cs_spectral_valid", "eb_spectral_valid"]
STATES = [("random_3_regular", 2), ("erdos_renyi", 2)]
DEPTHS = [1, 2, 4]
OUT = BASE_DIR / "results_raw" / "phase7" / "exp7c_blinding_dose_response.csv"
CHUNK_ELEMS = 3_000_000


def ucb_chunked(Y: np.ndarray, rule: str, alpha: float, lmin: float) -> np.ndarray:
    rows, n = Y.shape
    step = max(1, CHUNK_ELEMS // n)
    out = np.empty(rows)
    for i in range(0, rows, step):
        out[i:i + step] = ucb_matrix(Y[i:i + step], rule, alpha, lmin)
    return out


def sample_losses(cdf: np.ndarray, loss_vec: np.ndarray, rng: np.random.Generator, shape) -> np.ndarray:
    return loss_vec[np.searchsorted(cdf, rng.random(shape))]


def optimize_state(ev: RetrospectiveEvaluator, p: int, seed: int) -> np.ndarray:
    if p <= 2:
        return global_min_loss(ev, p, seed=seed)[1]
    b = get_bounds(p)
    x0 = get_default_x0(p, "wurtz_love", ev.engine.edges)
    r = minimize(lambda x: ev.true_loss(np.asarray(x)), x0, method="L-BFGS-B", bounds=list(map(tuple, b)), options={"maxiter": 60})
    return np.asarray(r.x, dtype=float)


def cdf_of(eng: QAOACircuitEngine, theta: np.ndarray) -> np.ndarray:
    pr = eng.statevector_probs(np.asarray(theta, dtype=float))
    c = np.cumsum(pr)
    c[-1] = 1.0
    return c


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    t0 = time.time()
    state_id = 0
    # Resume support: every state is seeded from its state_id only, so skipping already-saved states
    # reproduces exactly what an uninterrupted run would have produced.
    done_states = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done_states = set(prev.state_id.unique().tolist())
        print(f"[7C] resuming: {len(done_states)} states already saved", flush=True)
    for f_idx, (fam, n_graphs) in enumerate(STATES):
        graphs = generate_deduplicated_graph_suite(fam, 12, target_count=n_graphs, start_seed=40000 + 313 * f_idx, assign_partitions=False)
        for g_idx, g in enumerate(graphs):
            eng = QAOACircuitEngine(12, g.edges)
            lmin, _, _ = compute_spectral_loss_support(eng.spectral_upper_bound, eng.total_weight)
            loss_vec = 1.0 - eng.cuts / eng.total_weight
            for p in DEPTHS:
                if state_id in done_states:
                    state_id += 1
                    continue
                ev = RetrospectiveEvaluator(eng, p=p)
                theta_ref = optimize_state(ev, p, seed=g_idx)
                cdf_ref = cdf_of(eng, theta_ref)
                pr_ref = np.diff(np.concatenate([[0.0], cdf_ref]))
                mu_ref = float(pr_ref @ loss_vec)
                sd_ref = float(np.sqrt(pr_ref @ (loss_vec - mu_ref) ** 2))
                rng = np.random.default_rng(70000 + state_id)
                b = get_bounds(p)
                # fixed neighbourhood pool of 100 candidates (sigma_theta = 0.05) for B_s05
                pool_theta = [np.clip(theta_ref + rng.normal(0, 0.05, theta_ref.shape), b[:, 0], b[:, 1]) for _ in range(max(M_GRID))]
                pool_cdf = [cdf_of(eng, th) for th in pool_theta]
                pool_mu = np.array([float(np.diff(np.concatenate([[0.0], c])) @ loss_vec) for c in pool_cdf])
                meta = dict(family=fam, graph_idx=g_idx, graph_id=g.graph_id, p_depth=p, state_id=state_id,
                            mu_ref=round(mu_ref, 6), sd_ref=round(sd_ref, 5), ratio_ref=round((1 - mu_ref) * eng.total_weight / eng.opt_cut, 5),
                            spectral_loss_min=round(lmin, 5), reps=REPS)

                def emit(variant, M, n_conf, rule, tau_vec, flags_by_delta, note=""):
                    for d, k in zip(DELTAS, flags_by_delta):
                        rows.append({**meta, "variant": variant, "M": M, "n_conf": n_conf, "rule": rule, "delta": d,
                                     "n_false_cert": int(k), "note": note})

                # ----- screening for B variants (independent of n_conf since s_scr = 120 <= n_conf - 20)
                scr_cache = {}
                for sigma_key, cdfs in (("B_s0", None), ("B_s05", pool_cdf)):
                    for M in M_GRID:
                        if cdfs is None:
                            S = sample_losses(cdf_ref, loss_vec, rng, (REPS, M, S_SCR))
                            mus = np.full(M, mu_ref)
                        else:
                            S = np.stack([sample_losses(cdfs[j], loss_vec, rng, (REPS, S_SCR)) for j in range(M)], axis=1)
                            mus = pool_mu[:M]
                        best = S.mean(axis=2).argmin(axis=1)
                        chosen = S[np.arange(REPS), best, :]
                        scr_cache[(sigma_key, M)] = (chosen, best, mus)

                for n_conf in NCONF_GRID:
                    # -------- Variant A and D draws: 100 independent fresh batches per replication at theta_ref
                    # (processed in replication chunks to bound memory)
                    ucb_big = {r: np.empty((REPS, max(M_GRID))) for r in ALL_RULES}
                    bonf_min = {(r, M): np.empty(REPS) for r in BONF_RULES for M in M_GRID if M > 1}
                    CH = 100
                    for c0 in range(0, REPS, CH):
                        c1 = min(REPS, c0 + CH)
                        big = sample_losses(cdf_ref, loss_vec, rng, (c1 - c0, max(M_GRID), n_conf))
                        flat = big.reshape(-1, n_conf)
                        for r in ALL_RULES:
                            ucb_big[r][c0:c1] = ucb_chunked(flat, r, ALPHA, lmin).reshape(c1 - c0, max(M_GRID))
                        for r in BONF_RULES:
                            for M in M_GRID:
                                if M > 1:
                                    sub = big[:, :M, :].reshape(-1, n_conf)
                                    bonf_min[(r, M)][c0:c1] = ucb_chunked(sub, r, ALPHA / M, lmin).reshape(c1 - c0, M).min(axis=1)
                        del big, flat
                    tau_vec = mu_ref - np.array(DELTAS)
                    for r in ALL_RULES:
                        for M in M_GRID:
                            if M == 1:
                                emit("A", 1, n_conf, r, tau_vec, [(ucb_big[r][:, 0] <= t).sum() for t in tau_vec])
                            else:
                                mn = ucb_big[r][:, :M].min(axis=1)
                                emit("D", M, n_conf, r, tau_vec, [(mn <= t).sum() for t in tau_vec])
                    for (r, M), ub in bonf_min.items():
                        emit("D_bonf", M, n_conf, r, tau_vec, [(ub <= t).sum() for t in tau_vec])
                    del ucb_big, bonf_min

                    # -------- Variant B (reuse of screening shots)
                    for sigma_key, cdfs in (("B_s0", None), ("B_s05", pool_cdf)):
                        for M in M_GRID:
                            chosen, best, mus = scr_cache[(sigma_key, M)]
                            if cdfs is None:
                                extra = sample_losses(cdf_ref, loss_vec, rng, (REPS, n_conf - S_SCR))
                            else:
                                extra = np.empty((REPS, n_conf - S_SCR))
                                for j in np.unique(best):
                                    idx = np.flatnonzero(best == j)
                                    extra[idx] = sample_losses(cdfs[int(j)], loss_vec, rng, (len(idx), n_conf - S_SCR))
                            Y = np.concatenate([chosen, extra], axis=1)
                            tau_vec = mus.min() - np.array(DELTAS)
                            for r in ALL_RULES:
                                ub = ucb_chunked(Y, r, ALPHA, lmin)
                                emit(sigma_key, M, n_conf, r, tau_vec, [(ub <= t).sum() for t in tau_vec],
                                     note=f"tau=min_j mu_j - delta; min_mu={mus.min():.5f}")
                state_id += 1
                print(f"[7C] {fam} g{g_idx} p={p} done (mu_ref={mu_ref:.4f}, sd={sd_ref:.3f})  elapsed {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[7C] finished: {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

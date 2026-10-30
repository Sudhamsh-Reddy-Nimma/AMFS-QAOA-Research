# Phase 1 — Forensic Code and Data Audit

**Document Status:** Phase 1 Complete (Technical Research Document — Not a Manuscript Section)
**Outputs Verified:** `results_paper/metrics/source_result_map.csv`, `results_paper/metrics/bug_register.csv`, `results_paper/metrics/historical_reconciliation.csv`, `results_paper/metrics/claim_evidence_matrix.csv`.

---

## 1. Source-to-Result Map & Codebase Inventory (Task 1.1)

We inspected all 16 historical source and data files across three generations of the project preserved in `codes/legacy/` and `results_raw/historical/`:
1. **Generation 0 (Original Bundle):** `MAD_1.py`, `amfs_qaoa_shot_benchmark.py`, `amfs_qaoa_shot_results.csv`, `README_AMFS_QAOA.txt`.
2. **Generation 1 (First-Iteration Suites 1–4):** `amfs_variants.py`, `baselines.py`, `qaoa_simulator.py`, `run_verified_benchmark.py`, `test_amfs_qaoa.py`, `suite1_original_vs_corrected.csv`, `suite2_multigraph_multidepth.csv`, `suite3_ablation_study.csv`, `suite4_drift_tracking.csv`, `suite4_noise_models.csv`.
3. **Generation 2 (Previous Workstream Iteration `ws_prev_*`):** `ws_prev_graph_library.py`, `ws_prev_qaoa_core.py`, `ws_prev_stopping_rules.py`, `ws_prev_warmstart_transfer.py`, `ws_prev_multifidelity_surrogate.py`, `ws_prev_optimizers.py`, `ws_prev_run_all_workstreams.py`, and their 14 trial-level and summary CSVs.

Every major historical claim and metric has been mapped to its exact script, entry point, configuration, and raw CSV in `results_paper/metrics/source_result_map.csv`.

---

## 2. Audit of Historical & Prior-Iteration Methodological Issues (Task 1.2)

### A. Exact-Oracle Leakage
- **Generation 0 (`amfs_qaoa_shot_benchmark.py`):** Line 116 divided noisy sample cuts by `self.opt_cut` (the brute-force $2^n$ MaxCut optimum), and Line 183 stopped `run_amfs` when `obj.exact_ratio(g, b) >= 0.78` using the noiseless statevector.
- **Generation 1 (`amfs_variants.py`):** Replaced statevector stopping with `check_target_certified`, but still defined the target loss threshold via `1.0 - 0.78 * (opt_cut / W_tot)`, leaking the NP-hard `opt_cut` into the online stopping rule.
- **Generation 2 (`ws_prev_qaoa_core.py`):** Correctly isolated `OperationalQAOAOracle` from `opt_cut` and `true_ratio`, defining operational targets via $W_{\mathrm{tot}}$ or the polynomial-time Mohar–Poljak spectral bound $U_{\mathrm{spec}}(G)$.

### B. In-Sample Certification vs Split-Sample Defects
- **Generation 1 (`amfs_variants.py`):** Evaluated a normal LCB (`mean - 1.96 * se`) directly on the **same screening observations used to select the winning candidate** and repeated the test after every evaluation without multiplicity correction. In `suite1_original_vs_corrected.csv`, **2 out of 28 certified runs (7.14%)** had true expected cut ratio $< 0.78$ (`graph_seed=106, seed=104`: `0.777110`; `seed=105`: `0.778019`). In `suite2_multigraph_multidepth.csv`, **6 out of 117 certified runs (5.13%)** were false certifications.
- **Generation 2 (`ws_prev_stopping_rules.py`) — Newly Uncovered Mathematical Defects (`BUG-M01` to `BUG-M04`):**
  1. **`BUG-M01` (Invalid Support Range $R \in \{0.5, 0.65\}$):** `compute_split_eb_ucb` defaulted to `loss_range=0.5` and `certify_candidate_split_sample` passed `loss_range=0.65`, despite $\ell(z) = 1 - C(z)/W_{\mathrm{tot}}$ having worst-case support $[0, 1]$ ($R=1.0$). Using $R < 1.0$ without an online proof that $\ell(z) \in [a, b]$ with $b-a \le R$ invalidates the non-asymptotic concentration guarantee.
  2. **`BUG-M02` (Unadjusted Minimum of Two $(1-\alpha_k)$ Bounds):** `compute_split_eb_ucb` returned `mean_val + min(eb_rad(alpha_m), hoeff_rad(alpha_m))`, where both radii were evaluated at error level $\alpha_m$. By Boole's inequality, the intersection of two $(1-\alpha_m)$ bounds has worst-case error probability up to $2\alpha_m$ unless each bound is evaluated at $\alpha_m / 2$.
  3. **`BUG-M03` (Mislabelled "Holm–Bonferroni" Procedure):** `certify_candidate_split_sample` used fixed equal Bonferroni allocation $\alpha_k = \alpha / K_{\max}$ (`alpha_total / max_attempts`) while naming the rule `split_student_t_holm`. A fixed per-attempt allocation is **Bonferroni**, not the Holm step-down procedure.
  4. **`BUG-M04` (Asymptotic Student-$t$ vs Non-Asymptotic Guarantee):** `compute_split_student_ucb` relies on the Student-$t$ quantile plus a heuristic $t^2/(6n)$ skewness pad. While useful as an empirical plug-in baseline, it is **not** a distribution-free finite-sample guarantee for arbitrary bounded skewed distributions.

### C. Incorrect QAOA Ceilings & Terminology
- **Generation 1 (`run_verified_benchmark.py`):** Hardcoded `qaoa_ceiling = 0.865` for all $p=2$ graphs; **128/540 (23.70%)** of $p=2$ runs in `suite2_multigraph_multidepth.csv` exceeded `0.865` (up to `0.985708`).
- **Generation 2 (`ws_prev_qaoa_core.py`):** Computed per-graph ceilings via 16-start L-BFGS-B on the noiseless statevector. Per Section 1.2.C, because L-BFGS-B is a numerical multi-start local solver on a non-convex trigonometric landscape, this value must be explicitly labeled `numerically_estimated_p_ceiling` rather than a mathematically certified global QAOA ceiling (whereas `exact_opt_cut` via $2^n$ enumeration for $n \le 12$ is a mathematically exact global MaxCut optimum).

### D. Graph Duplication & Exact Isomorphism Verification
- **Generation 1 (`qaoa_simulator.py`):** Had severe seed collisions on $n=8$ 3-regular graphs (`100,105,107` identical; `101,102,106` identical; `400..404` identical).
- **Generation 2 (`ws_prev_graph_library.py`):** Used a SHA-256 hash of eigenvalues, degrees, and cut quantiles (`spectral_hash`), and did not separate graphs into disjoint **development, validation, and evaluation partitions**.
- **Phase 3 Correction (`BUG-H04`):** We implement an **exact permutation/VF2 weighted graph isomorphism verifier** (`are_graphs_isomorphic_exact`) and partition our deduplicated graph bank into disjoint `dev`, `val`, and `eval` splits.

### E. Warm-Start Confounding
- Recomputing `suite2_multigraph_multidepth.csv` at $p=2, B=2400$ confirmed that `amfs_imp_warmstart` achieved `0.880891` (unweighted) and `0.854877` (weighted), while 0-shot `fixed_wurtz_love` achieved `0.863042` and `0.845395` on the exact same instances. In Phase 6, we strictly separate **Evaluation A (End-to-End Strategy)** from **Evaluation B (Matched Initialization)**.

### F. Mislabelled Multifidelity Optimization & Exactness Conditions of $F_0$
- `MAD_1.py` and `amfs_variants.py` varied only shot counts on the same circuit (single-fidelity adaptive sampling).
- In `ws_prev_qaoa_core.py`, `classical_fidelity0_loss` computes a closed-form local 1-hop correlator. Per Section 1.2.F, we verify and document that this formula is **mathematically exact only at circuit depth $p=1$ on graphs whose 1-hop edge neighborhoods contain no triangles (girth $\ge 4$, such as $C_4$)**; on graphs with triangles or at depths $p \ge 2$, it is an **approximate low-fidelity heuristic proxy**. Moreover, as shown in `claim_evidence_matrix.csv` (`CLM-06`), reducing holdout surrogate RMSE by $33.7\%\text{–}57.3\%$ did **not** increase the certified success rate at $B=600$ (`13.89%` for `amfs_mf_full_f0_f1_f2` vs `16.67%` for `amfs_ablate_no_f0`).

---

## 3. Smallest Defensible Correction Set for Phase 3 (Task 1.4)

To evaluate the central research hypothesis cleanly without unnecessary code churn, we separate our Phase 3 modifications into:
1. **Essential Mathematical Corrections (`codes/src/stopping_rules.py`):**
   - Implement exact support width $R = 1.0$ on $[0, 1]$ (with optional observable upper bound $R_{\mathrm{obs}} = 1.0$ when $C_{\max}$ is unknown).
   - Provide distinct, rigorously verified confidence-bound functions:
     - `hoeffding_ucb_valid(samples, alpha_k, R=1.0)` (exact non-asymptotic Hoeffding bound at level $\alpha_k$).
     - `empirical_bernstein_ucb_valid(samples, alpha_k, R=1.0)` (exact Maurer & Pontil 2009 Theorem 4 bound at level $\alpha_k$ with $R=1.0$).
     - `intersection_eb_hoeff_ucb_valid(samples, alpha_k, R=1.0)` (evaluates both EB and Hoeffding at split level $\alpha_k / 2$ so $\min(U_{\mathrm{EB}}(\alpha_k/2), U_{\mathrm{Hoeff}}(\alpha_k/2))$ is provably valid at level $1-\alpha_k$ by the union bound).
     - `time_uniform_eb_cs_valid(samples_stream, alpha_k, R=1.0)` (exact predictable-plugin Empirical-Bernstein confidence sequence, Waudby-Smith & Ramdas 2024).
     - `student_t_ucb_heuristic(samples, alpha_k)` (explicitly labeled as an **asymptotic/heuristic** comparison baseline, never claimed as a finite-sample non-asymptotic guarantee).
   - Implement exact $\alpha$-spending schedules (`fixed_bonferroni`: $\alpha_k = \alpha / K_{\max}$ for bounded $K_{\max}$, and `harmonic_spending`: $\alpha_k = \frac{\alpha}{k(k+1)}$ for arbitrary $k \ge 1$ since $\sum_{k=1}^\infty \frac{1}{k(k+1)} = 1$).
2. **Essential Experimental Controls (`codes/src/graph_library.py` & `codes/src/optimizers.py`):**
   - Add exact weighted graph isomorphism verification (`are_graphs_isomorphic_exact`) and explicit `partition` tags (`dev`, `val`, `eval`).
   - Enforce the strict resource-accounting invariant $B_{\mathrm{search}} + B_{\mathrm{confirmation}} = B_{\mathrm{total}} \le B$ across all methods.

# Phase 7 — Claim validation (pre-specified, out-of-sample, fair thresholds)

Phase 7 re-tests the three empirical claims that review identified as the load-bearing ones. The hypotheses,
grids and verdict rules were written into the header of each experiment script **before** it was run, and the
verdicts below are produced mechanically by `codes/analysis/analyze_phase7.py`. All numbers are read from
`results_paper/metrics/phase7_*.csv` and `phase7_claim_verdicts.json`, and re-derived by the independent
`codes/analysis/verify_phase7_independently.py`.

| # | Claim as originally worded | Verdict | What the data actually support |
|---|---|---|---|
| 1 | Adaptive selection breaks naive in-sample certification; a fresh split with a valid rule fixes it | **Supported, but much narrower than first reported** | Inflation is real for the optimizer that stops on in-sample evidence (`amfs_md_target`: 7.2% at tau = global_min − 0.002, CI 6.2–8.3%, rising to 14–15% in some configurations at the larger budget). Not 70%. The other three optimizers show no inflation above alpha. |
| 2 | Under matched confirmation splits adaptive search beats fixed tree-subgraph transfer | **General claim not supported; narrowed claim strongly supported** | Adaptive search beats transfer on non-regular (Erdős–Rényi) graphs at p ≥ 2 by 2–4.5 ratio points on 20/20 fresh graphs. At p = 1, and on regular graphs, transfer wins. |
| 3 | Valid bounds fail under broken blinding (reuse / cherry-picking) | **Not supported for valid bounds — dropped** | On real QAOA output distributions no valid rule ever exceeds alpha (max 0.39% at M ≤ 25, 1.56% at M = 100). What *is* supported is a different statement: a rule that is calibrated under correct blinding but not conservative (Student-t) is destroyed by blinding violations. |

---

## Claim 1 — selection bias in in-sample certification (Exp 7A)

**What was wrong before.** Phase 4A-1 tested its "unsplit in-sample" arm at `tau + 0.025` (z-rule) and
`tau + 0.055` (EB rule) while the split arms were tested at `tau`. At the true null boundary the headline
70–77.5% drops to ≈10% (4/40) for `amfs_md_target` and 0–2.5% for the rest; with the offset, most of the
"false certifications" were certifications of a claim (`mu <= tau + 0.025`) that was in fact true.

**Design.** 5 configurations (random 3-regular p=1,2; Erdős–Rényi p=1,2; weighted bimodal p=2), n = 12,
6 fresh non-isomorphic graphs each, 40 seeds, budgets 600 and 1200, four real optimizers → 9,600 searches
(2,400 trials per optimizer and threshold). Every arm is tested at exactly the same `tau = mu_min + delta`, with
`mu_min` from a dense-grid/multistart global minimum (`landscape_utils.py`), not the short COBYLA "ceiling".
"False certification" = certifies **and** the exact loss of the certified theta is above tau.

**Results (strict null: every theta violates the claim).**

| In-sample z = 1.96 arm, false-cert rate | delta = −0.010 | −0.005 | −0.002 |
|---|---|---|---|
| `amfs_md_target` | 0.9% | 4.0% | **7.2%** [6.2, 8.3] |
| `cobyla_hao2025` | 0.1% | 0.8% | 1.5% |
| `icans` | 0.0% | 0.3% | 0.6% |
| `spsa` | 0.0% | 0.1% | 0.2% |

* Every **fresh-split valid rule** (Hoeffding, EB, time-uniform CS, spectral and R = 1): **0 false certifications in all strict-null cells**.
* In-sample *valid* rules (EB, time-uniform CS applied per theta) are also at 0 false certifications — the inflation is a property of the z-rule, not of in-sample use per se.
* The inflation **grows with search effort**: for `amfs_md_target` (tau = global_min − 0.002), going from B = 600 to B = 1200 raises the mean number of distinct thetas visited from 18.6 to 33.1 and the false-cert rate from 9.2→10.8% (ER p=1), 10.8→14.2% (3-regular p=1), 2.1→14.6% (3-regular p=2), 0.4→5.0% (ER p=2), 0.4→4.6% (bimodal p=2). 18 of 30 graphs exceed 5%.
* **Non-vacuity.** At `n_conf = 150` the time-uniform spectral rule has 0.0% / 1.8% / 69.0% true-certification at delta = +0.010 / +0.025 / +0.050; Student-t has 18.1% / 70.4% / 96.5% with ≤ 1.5% false certification in the strict null. The valid rule is conservative, not broken, and needs more shots (or a larger margin) than 150 to be useful.

**Defensible statement.** *For optimizers that stop on in-sample evidence, certifying with a naive z-interval on the same shots exceeds its nominal level (up to ~7% on average and ~14–15% in some configurations, growing with the number of candidate parameters examined), whereas certifying on fresh shots with any of the rules tested stays at or below alpha.* It should **not** be stated as a general failure of in-sample certification: three of four optimizers show no excess, and valid in-sample bounds show none either.

---

## Claim 2 — adaptive search vs fixed transfer (Exp 7B)

**Design.** 80 brand-new graphs (n = 12; 20 per family; seeds ≥ 30000, never used before), regimes
(p=1,B=300), (p=1,B=600), (p=2,B=1200), (p=3,B=1200), 3 seeds per graph, the unmodified Phase 6 trial runner,
four adaptive methods, matched-40%-split transfer **and** a full-budget transfer baseline (all B shots to
confirmation — the fair baseline for a method that needs no search). Paired Wilcoxon over graphs, Holm-corrected.

**Pre-specified hypotheses and outcomes**

* **H2a (narrowed claim)** — on Erdős–Rényi, p ∈ {2,3}, `icans` and `spsa` beat matched transfer: **SUPPORTED**, Holm p ≤ 1e-5.
  `icans` +0.034 (p=2; 0.844 vs 0.810) and +0.045 (p=3), `spsa` +0.027 and +0.030; better on 20/20 graphs in 3 of 4 tests (19/20 for `icans` p=3). `cobyla_hao2025` (+0.021, +0.029) and `amfs_md_target` (+0.020 at p=2) also win.
* **H2b (boundary)** — on random 3-regular graphs at p=1 transfer is not beaten by any adaptive method: **holds** (mean differences −0.002 to −0.060; `spsa` is worse on 20/20 graphs).
* **H2c (general claim as worded)** — of 64 (family, regime, method) cells, adaptive search is Holm-significantly better in **12** and significantly **worse in 28**: **general claim NOT supported.** Outside Erdős–Rényi the only consistent adaptive win is `spsa` at p ≥ 2 on weighted graphs (+0.018 on bimodal; ≈ +0.002 on regular/uniform).
* **H2d (allocation)** — at p=1, B ≤ 600 full-budget transfer certifies more often than every adaptive method (pooled: 0.867 vs 0.12–0.41 at B=300; 0.942 vs 0.30–0.82 at B=600). The "matched 40%" split handicaps transfer; it should be shown as one view, with full-budget transfer as the primary baseline.
* On Erdős–Rényi at p ≥ 2, adaptive search also certifies more often **even against full-budget transfer** (p=3: `icans` 0.967 vs 0.367; p=2: 1.000 vs 0.817).

**Defensible statement.** *When the graph is outside the regime tree-subgraph transfer was derived for (non-regular, Erdős–Rényi) and the circuit has p ≥ 2, adaptive search outperforms parameter transfer by 2–4.5 ratio points and certifies more reliably; for regular graphs and shallow circuits, zero-shot transfer is at least as good.*

---

## Claim 3 — valid bounds under broken blinding (Exp 7C)

**What was wrong before.** The Phase 5 regime in which valid rules failed was a hand-built two-point distribution
(92.5% mass at loss_min + 0.02, 7.5% at 1.0) with tau 0.002 below its mean, and only one of four valid rules failed.

**Design.** 12 real QAOA output distributions (random 3-regular and Erdős–Rényi, n = 12, optimized p = 1, 2, 4;
mean approximation ratios 0.775 / 0.847 / 0.923), 12,000 replications per cell, strict nulls
(`tau = min mu − delta`, delta ∈ {0.002, 0.005, 0.01, 0.02}), pool sizes M ∈ {1, 5, 25, 100},
n_conf ∈ {150, 300, 600}; variants A (valid), B (reuse of 120 screening shots; identical or perturbed candidates),
D (cherry-pick the best of M fresh batches), and D_bonf (Bonferroni α/M — positive control).

**Results.**

* **Valid rules never exceed alpha.** Max false-cert rate across all valid rules, variants B and D: **0.39%** for M ≤ 25 and **1.56%** for M = 100 (time-uniform spectral rule, n_conf = 600, delta = 0.002). Under correct blinding (A) the max is 0.008%; D_bonf is 0.0%. Pre-specified verdict: **NOT_SUPPORTED**.
* **Non-conservative rules do break.** Student-t is calibrated under correct blinding (3.5% at n_conf=150, 1.5% at 600) but under blinding violations reaches **55.8%** (D, M=25), **42.7%** (B, M=25, identical candidates), **16.8%** (B, M=25, realistic neighbourhood) at n_conf=150; 31.9% (D, M=25) at n_conf=600. 53 diagnostic-rule cells have a CI lower bound above 0.05.
* The finite-sample valid bounds have so much slack on real QAOA distributions (per-test false-cert ≈ 1e-4 at n_conf = 600, ≈ 0 at 150) that, extrapolating the measured M = 100 rate linearly, it would take on the order of hundreds of cherry-picks at n_conf = 600 (and far more at smaller n_conf) before the union bound reaches 5%. That extrapolation is an estimate, not a measurement; the grid stops at M = 100.

**Defensible statement (replaces the dropped claim).** *Blinding is necessary for any rule whose per-test error is close to its nominal level (Student-t: 3.5% → 56%); finite-sample valid bounds are conservative enough on real QAOA distributions to tolerate pools of ≥ 100 reuses, so their failure under broken blinding can only be shown on engineered distributions such as the Phase 5 two-point regime.* The second half of this statement is exploratory (the Student-t / invalid-rule rows were pre-specified as diagnostics, but their interpretation here was not a pre-specified verdict).

---

## Caveats that apply to all three

* n = 12 qubits, noiseless statevector simulation; Phase 7 does not test hardware noise or n ≥ 14 (the larger-n scaling data from Phase 6 are not re-tested here).
* Four graph families, all 3-regular-or-Erdős–Rényi variants; conclusions about "non-regular" graphs rest on Erdős–Rényi only.
* Claim 1's effect depends on how an optimizer allocates shots across thetas; it is shown for four optimizers, not for adaptive methods in general.
* The valid rules are conservative. At `n_conf = 150` they certify nothing at margins ≤ 0.025; practical use requires larger confirmation budgets or the margin sweep in Phase 4C.
* 7C was interrupted once by a sandbox restart and resumed with `state_id`-keyed seeds, which reproduces an uninterrupted run exactly (the script's resume logic only skips already-saved states).
* The global minimum for p = 2 comes from multistart L-BFGS-B (40 starts); at p = 1 it is a dense grid with polishing. A residual error of ~2e-4 at p = 2 is immaterial against deltas ≥ 0.002.

## Reproduce

```
python codes/tests/test_phase7_batch_equivalence.py          # vectorized rules == scalar rules (1e-9)
python codes/experiments/run_phase7a_selection_bias.py       # ~7 min on one core
python codes/experiments/run_phase7b_transfer_vs_adaptive.py # ~2 min
python codes/experiments/run_phase7c_blinding_dose_response.py  # ~10 min (resumable)
python codes/analysis/analyze_phase7.py                      # tables, figures, verdict JSON
python codes/analysis/verify_phase7_independently.py         # independent recomputation
```

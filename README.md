# AMFS_QAOA_Research — Statistically Valid QAOA Target Certification Under Adaptive Selection and Finite Shot Budgets (Phases 0–6)

## Overview
This research repository contains the complete technical audit, formal statistical proofs, corrected software implementation, verification tests, and experimental data for **Phases 0 through 6** of *Statistically Valid QAOA Target Certification Under Adaptive Selection and Finite Shot Budgets*.

In strict compliance with **Section 1 and Rule 13 of the Master Research Instruction**, this package contains **no manuscript, paper draft, abstract, or submission files**. All outputs are technical research documents, mathematical proof notes, experiment protocols, test suites, analysis scripts, raw and summary CSVs, and plots.

---

## Directory Structure

```text
AMFS_QAOA_Research/
├── codes/
│   ├── legacy/          # Preserved untouched historical & prior-iteration scripts (MAD_1.py, etc.)
│   ├── src/             # Corrected Phase 3 implementation (stopping_rules.py, graph_library.py, qaoa_core.py, optimizers.py, warmstart_transfer.py, multifidelity_surrogate.py)
│   ├── tests/           # Phase 3 dedicated certification & reference test suite (test_phase3_certification.py)
│   ├── experiments/     # Reproducible experiment drivers (run_phase4_experiments.py, run_phase5_experiments.py, run_phase6_regime_map.py)
│   ├── analysis/        # Independent forensic & verification scripts (phase1_audit_runner.py, verify_all_phases_independently.py)
│   └── configs/         # Environment manifest & SHA-256 checksums
├── results_raw/
│   ├── historical/      # Untouched historical CSVs (amfs_qaoa_shot_results.csv, suite1..suite4, ws_prev_*)
│   ├── phase3/          # Phase 3 test results & synthetic statistical calibration (phase3_test_results.csv, phase3_synthetic_calibration.csv)
│   ├── phase4/          # Phase 4 raw decomposition, QAOA variances & drift check (exp4a_insample_failure.csv, empirical_qaoa_loss_variances.csv, exp4d_nonstationary_check.csv)
│   ├── phase5/          # Phase 5 confirmation-reuse raw data (phase5_reuse_ablation_raw.csv)
│   └── phase6/          # Phase 6 graph instances & trial-level data (phase6_graph_instances.csv, joint_regime_map_raw.csv)
├── results_paper/
│   ├── metrics/         # All required Phase 1, 4, 5, and 6 summary & comparison CSVs
│   └── figures/         # Reproducible Phase 4, 5, and 6 diagnostic plots
├── protocols/
│   └── phase4_experiment_protocol.md
├── docs/
│   ├── phase0_pivot_assessment.md
│   ├── phase1_forensic_audit.md
│   ├── phase2_formal_definitions.md
│   ├── phase2_theorem_notes.md
│   ├── phase2_proof_checks.md
│   ├── phase2_assumptions_and_limitations.md
│   ├── change_log.md
│   ├── phase3_implementation_validation.md
│   ├── phase4_results.md
│   ├── confirmation_information_flow.md
│   ├── phase5_validation.md
│   └── phase6_regime_map_analysis.md
└── README.md
```

---

## Reproducing All Phases

From the root of `AMFS_QAOA_Research/`, run:
```bash
python3 codes/analysis/phase1_audit_runner.py
python3 codes/tests/test_phase3_certification.py
python3 codes/experiments/run_phase4_experiments.py
python3 codes/experiments/run_phase5_experiments.py
python3 codes/experiments/run_phase6_regime_map.py
python3 codes/analysis/verify_all_phases_independently.py
```

---

## Phase 7 — claim validation (added after independent audit)

Phase 7 re-tests the three load-bearing empirical claims with **pre-specified** hypotheses, fair thresholds,
fresh graphs/seeds, and real QAOA output distributions. Read `docs/phase7_claim_validation.md` first and
`docs/ERRATA_phases4_to_6.md` before citing any Phase 4–6 number.

| Claim | Phase 7 verdict |
|---|---|
| 1. In-sample certification inflated by adaptive selection; fresh split fixes it | Supported, **narrower** than first reported (amfs_md_target 7.2%, up to ~14–15% in some configs; not 70%) |
| 2. Adaptive search beats transfer under matched splits | General claim **not supported**; **narrowed** claim (Erdős–Rényi, p ≥ 2) strongly supported (20/20 fresh graphs) |
| 3. Valid bounds fail under broken blinding | **Not supported for valid bounds (dropped)**; non-conservative rules (Student-t) do break (3.5% → 56%) |

New files: `codes/src/batch_certification.py`, `codes/src/landscape_utils.py`,
`codes/experiments/run_phase7{a,b,c}_*.py`, `codes/analysis/analyze_phase7.py`,
`codes/analysis/verify_phase7_independently.py`, `codes/analysis/audit_checks/`,
`codes/tests/test_phase7_batch_equivalence.py`, `results_raw/phase7/`, `results_paper/metrics/phase7*.csv`,
`results_paper/figures/fig_phase7*.png`.

Files cited by the earlier PDF but **not** present in this package (so the claims resting on them are
unverified): `split_ratio_optimization.csv`, `scaling_n12_to_16_p1_to_4_summary.csv`,
`docs/literature_review_and_citations.md`. A recomputed scaling summary is provided as
`results_paper/metrics/scaling_n8_to_16_p1_to_4_summary_recomputed.csv`.

# Phase 0 — Pivot Assessment & Reframed Contribution Positioning

## 1. Executive Framing: Benchmarking-Methodology & Spectrally-Tightened Certification

Following rigorous internal and external peer audit, this research repository is framed around two defensible, non-tautological contributions rather than claiming textbook sample splitting as a new mathematical theorem or claiming an in-house multifidelity surrogate as a universal winner:

1. **Benchmarking-Methodology Contribution ("In-Sample Target Certification in QAOA Benchmarks Inflates Success, and Here Is the Cost of Fixing It"):**
   - When QAOA optimizers (`AMFS-MD`, `iCANS`, `COBYLA`, `SPSA`) evaluate target attainment on the same finite-shot trajectory used to select or update parameters—or apply asymptotic Student-$t$/Normal intervals to skewed finite-shot QAOA cut distributions—they suffer severe selection bias ("winner's curse") and optional-stopping inflation.
   - In direct, unfitted executions of the actual optimizers under strict global nulls (`results_raw/phase4/exp4a_actual_optimizer_insample_audit.csv`, where $\tau$ is set below the exact $p$-layer QAOA minimum loss so $\mu(\theta) > \tau$ for all $\theta \in \Theta$), unsplit in-sample stopping falsely certifies target attainment on **`70.0%–77.5%` of runs for `amfs_md_target`**, **`72.5%–75.0%` for `cobyla_hao2025`**, and **`15.0%` for `icans` and `spsa`**.
   - Even after sample splitting, asymptotic Student-$t$ bounds (`student_t_heuristic`) reach **`11.0%–14.0%` false certification** at nominal $\alpha = 0.05$ (`false_certification_calibration.csv`, `confirmation_reuse_ablation.csv`), while reusing current screening shots in confirmation (Variant B) or cherry-picking across candidates (Variant D) drives false certification to **`15.25%–22.25%` even under valid anytime confidence sequences**.

2. **Domain-Specific Technical Contribution — Online Spectrally-Tightened Support Certification (`Proposition 2`):**
   - Applying textbook Hoeffding (1963), Maurer–Pontil (2009), or Waudby-Smith & Ramdas (2024) bounds naively to normalized QAOA MaxCut loss $Y_s = 1 - C(z_s)/W$ assumes worst-case support $[0, 1]$ ($R = 1.0$). Because the finite-sample range penalty scales with $R$ (and Hoeffding sample complexity scales with $R^2$), $R=1.0$ bounds are overly conservative and overstate the shot cost of valid confirmation.
   - We exploit the **Delorme–Poljak (1993) / Mohar–Poljak (1990) zero-sum diagonal-corrected Laplacian upper bound**:
     $$U_{\mathrm{spec}}(G) = \min_{u \in \mathbb{R}^n:\, \sum_{i=1}^n u_i = 0} \frac{n}{4}\,\lambda_{\max}\!\big(L_G + \mathrm{diag}(u)\big),$$
     which is computable online in $O(n^3)$ classical time before any quantum shots are taken and guarantees $0 \le C(z_s) \le U_{\mathrm{spec}}(G)$ for every bitstring $z_s \in \{0,1\}^n$.
   - Consequently, every single-shot loss $Y_s$ is deterministically confined to the **spectrally-tightened support**
     $$Y_s \in \left[1 - \frac{U_{\mathrm{spec}}(G)}{W},\, 1\right], \qquad R_{\mathrm{spec}}(G) = \frac{U_{\mathrm{spec}}(G)}{W} \in [0.73, 0.91].$$
   - Replacing $R = 1.0$ with the online-known $R_{\mathrm{spec}}(G)$ preserves 100% non-asymptotic validity while reducing theoretical and empirical confirmation shot requirements for $80\%$ power by **$28\%\text{--}40\%$** (`confirmation_power_cost.csv`).

3. **Fair, Matched-Split Joint Search-and-Certification Regime Map Across $n \in \{8, 12, 14, 16\}$ and $p \in \{1, 2, 3, 4\}$:**
   - Giving zero-search parameter transfer (`fixed_shaydulin_fullconf`) $450$ confirmation shots while giving search methods $120\text{--}200$ confirmation shots creates an allocation tautology.
   - In our upgraded Phase 6 design (`joint_regime_map.csv`, `split_ratio_optimization.csv`, `scaling_n12_to_16_p1_to_4_summary.csv`), we evaluate all methods under **matched confirmation splits** ($\rho_{\mathrm{conf}} = B_{\mathrm{confirmation}}/B \in \{0.25, 0.40, 0.60, 0.80\}$) across $68$ deduplicated non-isomorphic graphs spanning $n \in \{8, 12, 14, 16\}$ and depths $p \in \{1, 2, 3, 4\}$.
   - Under matched confirmation budgets, zero-search transfer and adaptive search tie on homogeneous unweighted 3-regular graphs at $p=1$, whereas **adaptive search (`icans`, `bobyqa_trust`, `amfs_md_target`, `cobyla_hao2025`) outperforms fixed tree-subgraph transfer on non-regular graphs (`erdos_renyi`), bimodal weighted graphs (`weighted_3_regular_bimodal`), and deeper circuits ($p \in \{2, 3, 4\}$)**.

---

## 2. Explicit Separation of Textbook Machinery vs. Domain Contributions

| Component | Classification | Primary Reference | Role in This Repository |
| :--- | :--- | :--- | :--- |
| Sample splitting + Boole/Bonferroni/harmonic $\alpha$-spending (`Proposition 1`) | **Textbook Statistical Machinery** | Cox (1975); Maurer & Pontil (2009) | Baseline mathematical framework ensuring FWER $\le \alpha$ across $K_{\max}$ adaptive confirmation attempts. **Not claimed as a novel mathematical theorem.** |
| Predictable-plugin Empirical-Bernstein test supermartingale (`Proposition 3`) | **Established Sequential Statistics (Bug-Fixed in Implementation)** | Waudby-Smith & Ramdas (*JRSSB* 2024, Prop. 2, Eqs. 14–15) | Provides anytime-valid sequential stopping within confirmation batches once historical tuning parameters $\lambda_i$ are made strictly $\mathcal{G}_{i-1}$-measurable (`n_horizon=None` or fixed predeclared horizon). |
| Online Delorme–Poljak / Mohar–Poljak spectrally-tightened support $[1 - U_{\mathrm{spec}}(G)/W, 1]$ (`Proposition 2`) | **Domain-Specific Technical Contribution** | Mohar & Poljak (1990); Delorme & Poljak (1993) + This Work | Shrinks deterministic QAOA loss support width from $R=1.0$ to $R_{\mathrm{spec}}(G) = U_{\mathrm{spec}}(G)/W$, cutting valid confirmation shot cost by $28\%\text{--}40\%$. |
| Direct unfitted optimizer global-null audit & Matched-split regime map ($n=8\dots 16, p=1\dots 4$) | **Empirical & Benchmarking Contribution** | This Work (`run_phase4_experiments.py`, `run_phase6_regime_map.py`) | Quantifies real optimizer false-certification rates under unsplit stopping and establishes fair matched-split performance across graph topologies, system sizes, and depths. |
| 3-Level Multifidelity Co-Kriging Surrogate (`amfs_mf_surrogate`) | **Negative / Diagnostic Ablation Result** | Kennedy & O'Hagan (2000) + This Work | Demonstrates why classical 1-hop tree-correlator surrogates lose rank correlation on cyclic graphs at $p \ge 2$; reported transparently as a secondary ablation rather than a headline method. |

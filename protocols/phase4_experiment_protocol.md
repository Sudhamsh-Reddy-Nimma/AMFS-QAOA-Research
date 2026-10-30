# Phase 4 — Predeclared Experimental Protocol: False-Certification Calibration and Confirmation-Cost Frontier

**Document Status:** Phase 4 Predeclared Protocol (Sections 4A–4D)

---

## 1. Experiment 4A — Reconstructing Historical In-Sample Failure & Factor Decomposition

We isolate the three distinct mechanisms responsible for the historical in-sample false certifications (`suite1` and `suite2`):
1. **Candidate Selection Bias ("Winner's Curse"):** Selecting $\widehat{\theta} = \arg\min_{j=1\dots M} \widehat{\mu}_{\mathrm{screen}}(\theta_j)$ from $M \in \{1, 5, 10, 20, 40\}$ candidates evaluated with $n_{\mathrm{screen}} \in \{40, 80, 120\}$ shots when all $M$ candidates lie in the null region ($\mu(\theta_j) = \tau + 0.008 > \tau$), and certifying using the **same** screening observations.
2. **Repeated Testing Without Alpha-Spending:** Checking in-sample certification across $K \in \{1, 2, 5, 10, 20\}$ sequential stages at fixed $\alpha = 0.05$ without Bonferroni/harmonic allocation.
3. **Invalid Normal Approximation vs Valid Fresh Split-Sample Confirmation:** Comparing in-sample normal UCB against fresh split-sample confirmation (`empirical_bernstein_valid`, `intersection_eb_hoeff_valid`, `time_uniform_eb_cs_valid`, `student_t_heuristic`).

Note: We do **not** assume the false-certification rate grows linearly with $M$; for Gaussian/sub-Gaussian screening means, $\mathbb{E}[\min_{j \le M} (\widehat{\mu}_j - \mu)] \asymp -\sigma_{\mathrm{screen}}\sqrt{2\ln M}$, so selection bias grows sublinearly and saturates at $1.0$.

---

## 2. Experiment 4B — Fresh-Sample Certification Calibration Grid

On actual QAOA simulator distributions across `random_3_regular` and `weighted_3_regular_bimodal` ($n=8$, $p \in \{1, 2\}$) and synthetic boundary distributions, we evaluate:
- Nominal error levels: $\alpha \in \{0.01, 0.05, 0.10\}$.
- Target regimes: Strict Null ($\mu(\theta) = \tau + 0.005$ and $\mu(\theta) = \tau + 0.03$), Boundary ($\mu(\theta) = \tau + 0.001$), and Indifference-Zone Interior Alternatives ($\mu(\theta) = \tau - \delta$ for $\delta \in \{0.02, 0.05, 0.10\}$).
- Confirmation sample sizes: $n_{\mathrm{conf}} \in \{50, 100, 150, 300, 500\}$.
- Permitted attempts: $K_{\max} \in \{1, 2, 4\}$ under `fixed_bonferroni` and `harmonic_spending`.
- Replications: $N_{\mathrm{rep}} = 300$ independent trials per condition with exact Clopper–Pearson 95% binomial CIs.

---

## 3. Experiment 4C — Power and Confirmation-Cost Frontier

1. **Empirical Measurement of Actual QAOA Cut-Loss Variances:** First, we measure the exact Born-rule loss variance $\sigma^2(\theta) = \mathrm{Var}_{z \sim P_\theta}[\ell(z)]$ across the parameter space $\Theta$ on all 6 graph families ($n \in \{8, 10, 12\}$, $p \in \{1, 2\}$) to determine how the specification's candidate variance values $\sigma^2 \in \{0.01, 0.03, 0.06\}$ compare to real QAOA distributions.
2. **Power vs Confirmation Shots Frontier:** Across tolerances $\delta \in \{0.02, 0.05, 0.10\}$, variances $\sigma^2 \in \{0.01, 0.022 (\text{empirical QAOA}), 0.03, 0.06\}$, and $n_{\mathrm{conf}} \in \{50, 100, 150, 250, 400, 600, 1000, 1500\}$, we record:
   - Empirical certification power ($1-\hat{\beta}$) and false-certification rate;
   - Theoretical **upper-bound sufficient sample size** $N_{\mathrm{suff}}^{\mathrm{EB}}$ and $N_{\mathrm{suff}}^{\mathrm{Hoeff}}$ (explicitly labeled as sufficient upper bounds, never as necessary lower bounds per Section 2.8 & 4C); and
   - Empirical sample size $N_{\mathrm{emp}, 0.80}$ required to reach $80\%$ certification power.

---

## 4. Experiment 4D — Stationary vs Non-Stationary Sampling Assumptions

We test what happens when hardware calibration drifts *during* or *immediately after* candidate selection ($\Delta\gamma \in \{0.0, 0.08, 0.18, 0.28\}$):
- **Case 1 (Stationary i.i.d. Confirmation):** Assumption A2 holds; false certification wrt the confirmation distribution is $\le \alpha$.
- **Case 2 (Post-Selection / Intra-Confirmation Drift):** If a candidate satisfies $\mu_{\mathrm{pre}}(\widehat{\theta}) \le \tau$ before drift, or if drift shifts the parameter linearly *during* the $n_{\mathrm{conf}}$ shots so that the terminal true loss is $\mu_{\mathrm{terminal}}(\widehat{\theta}) > \tau$ while the early confirmation shots had lower loss, the i.i.d. assumption of Theorem 1 is violated and the empirical false-certification rate wrt $\mu_{\mathrm{terminal}}$ can exceed $\alpha$.

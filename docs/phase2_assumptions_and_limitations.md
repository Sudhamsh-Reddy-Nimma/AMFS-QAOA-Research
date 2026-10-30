# Phase 2D — Explicit Assumptions, Scope Boundaries, and Limitations

**Document Status:** Phase 2 Technical Research Document (Sections 2.7–2.9 Exit Condition)

---

## 1. Summary of Required Assumptions for Theorem 1

| Assumption ID | Formal Statement | When It Holds in QAOA | When It Fails (Limitations) |
| :--- | :--- | :--- | :--- |
| **A1 (Bounded Support)** | Each single-shot loss $\ell(z) = 1 - C(z)/W_{\mathrm{tot}}$ satisfies $\ell(z) \in [0, 1]$ a.s. | Holds for **all** MaxCut graphs with nonnegative edge weights $w_{uv} \ge 0$ and $W_{\mathrm{tot}} > 0$. | If a graph has *negative* edge weights ($w_{uv} < 0$), $C(z)$ can be negative or exceed $W_{\mathrm{tot}}$, requiring normalization by $\sum_{(u,v)} |w_{uv}|$. |
| **A2 (Stationary Conditional i.i.d. Confirmation)** | Given $\mathcal{F}_{k-1}$ and $\{K \ge k\}$, the confirmation shots $\ell_{k,1}, \dots, \ell_{k,n_k}$ are i.i.d. draws from $P_{\widehat{\theta}_k}$. | Holds in stationary statevector/shot simulators and on hardware when all $n_k$ confirmation shots are executed within a single contiguous calibration window. | **Fails under non-stationary drift during confirmation** (e.g., if hardware parameters drift *while* the $n_k$ confirmation shots are being collected, or if confirmation certifies a pre-drift parameter that subsequently drifts). Investigated in Experiment 4D. |
| **A3 (Confirmation Blinding / No Current Reuse)** | The candidate $\widehat{\theta}_k$ is $\mathcal{F}_{k-1}$-measurable; $D_k^{\mathrm{conf}}$ is not used to select or modify $\widehat{\theta}_k$. | Holds whenever search shots and confirmation shots are strictly partitioned. | **Fails if confirmation shots are pooled into search selection before certifying** (Variant B and Variant D in Phase 5). |
| **A4 (Non-Asymptotic Bound Selection)** | The decision rule uses $U_{\mathrm{Hoeff}}$, $U_{\mathrm{EB}}$, $U_{\mathrm{Int}}$, or $U_{\mathrm{CS}}$ with $R=1.0$ and $\sum_k \alpha_k \le \alpha$. | Always holds by construction in `codes/src/stopping_rules.py`. | **Fails for `student_t_ucb_heuristic`** on small-sample skewed or near-deterministic adversarial distributions (verified in Phase 3.6). |

---

## 2. Explicit Scope Limitations Established in Phase 2

1. **Conservatism of Non-Asymptotic Bounds at Tight Shot Budgets ($B \le 600$):**
   Because $U_{\mathrm{EB}}$ and $U_{\mathrm{Int}}$ protect against *any* distribution on $[0, 1]$ (including rare point masses at $\ell = 1.0$), their confidence radii at $n_{\mathrm{conf}} = 150$ and $\alpha_k = 0.025$ are $\approx 0.085\text{–}0.105$. Consequently, non-asymptotic bounds achieve high certification power only when the candidate's true loss margin $\delta = \tau - \mu(\widehat{\theta})$ exceeds $\approx 0.08$ or when $n_{\mathrm{conf}} \ge 400\text{–}1000$ shots.
2. **Distinction Between Spectral Operational Certification and True MaxCut Ratio:**
   Certifying $\mu(\widehat{\theta}) \le 1 - \rho_{\mathrm{op}} U_{\mathrm{spec}}(G)/W_{\mathrm{tot}}$ guarantees $\mathbb{E}[C(\widehat{\theta})] \ge \rho_{\mathrm{op}} U_{\mathrm{spec}}(G) \ge \rho_{\mathrm{op}} C_{\max}(G)$, which is **sufficient** for $r(\widehat{\theta}) \ge \rho_{\mathrm{op}}$ but **more stringent** whenever $U_{\mathrm{spec}}(G) > C_{\max}(G)$.
3. **Phase 2 Exit Condition Status:**
   All mathematical definitions, confidence-bound formulas, $\alpha$-spending schedules, and proof steps are consistent and verified. We proceed to Phase 3 implementation and verification.

# Phase 2C — Line-by-Line Proof Checks & Numerical Boundary Audit

**Document Status:** Phase 2 Technical Research Document (Section 2.5–2.7 Proof Verification)

---

## 1. Line-by-Line Audit of Every Confidence-Bound Formula

| Bound Identifier | Exact Formula Implemented in Phase 3 (`codes/src/stopping_rules.py`) | Assumed Support | Sample Size Restriction | Variance Estimator | Log Factor | Level Allocation | Boundary Behavior ($V_n = 0$) | Guarantee Type |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `hoeffding_ucb_valid` | $\widehat{\mu}_n + R\sqrt{\frac{\ln(1/\alpha_k)}{2n}}$ | $[0, R]$ with $R=1.0$ | $n \ge 1$ | None | $\ln(1/\alpha_k)$ | $\alpha_k$ | Radius $> 0$ depends only on $n, R, \alpha_k$. | **Non-Asymptotic Fixed-Sample** |
| `empirical_bernstein_ucb_valid` | $\widehat{\mu}_n + \sqrt{\frac{2V_n\ln(2/\alpha_k)}{n}} + \frac{7R\ln(2/\alpha_k)}{3(n-1)}$ | $[0, R]$ with $R=1.0$ | $n \ge 2$ (returns $1.0$ if $n < 2$) | $V_n = \frac{1}{n-1}\sum (X_i-\widehat{\mu}_n)^2$ (`ddof=1`) | $\ln(2/\alpha_k)$ | $\alpha_k$ | First term $=0$, second term $= \frac{7R\ln(2/\alpha_k)}{3(n-1)} > 0$. | **Non-Asymptotic Fixed-Sample** |
| `intersection_eb_hoeff_ucb_valid` | $\min\!\left\{U_{\mathrm{Hoeff}}(\tfrac{\alpha_k}{2}, R),\, U_{\mathrm{EB}}(\tfrac{\alpha_k}{2}, R)\right\}$ | $[0, R]$ with $R=1.0$ | $n \ge 2$ | $V_n$ (`ddof=1`) in EB branch | $\ln(2/\alpha_k)$ in Hoeff; $\ln(4/\alpha_k)$ in EB | $\alpha_k/2 + \alpha_k/2 = \alpha_k$ | Strictly valid by Boole's inequality. | **Non-Asymptotic Fixed-Sample** |
| `time_uniform_eb_cs_valid` | $\min_{1\le s\le t}\frac{\ln(1/\alpha_k) + \sum_{i=1}^s 4(X_i-\widehat{\mu}_{i-1})^2\psi_E(\lambda_i) + \sum_{i=1}^s \lambda_i X_i}{\sum_{i=1}^s \lambda_i}$ | $[0, 1]$ ($R=1.0$) | Any $t \ge 1$ | Predictable $\widehat{\sigma}_{t-1}^2 = \frac{1/4+\sum_{i=1}^{t-1}(X_i-\widehat{\mu}_i)^2}{t}$ | $\ln(1/\alpha_k)$ via Ville's inequality | $\alpha_k$ | Prior $1/4$ keeps $\lambda_t \in (0, 1/2]$ finite when $X_i$ constant. | **Non-Asymptotic Time-Uniform** |
| `student_t_ucb_heuristic` | $\widehat{\mu}_n + t_{1-\alpha_k, n-1}\sqrt{\frac{V_n}{n}}$ | Finite variance | $n \ge 5$ (returns $1.0$ if $n < 5$) | $V_n$ (`ddof=1`) | Student-$t$ quantile | $\alpha_k$ | If $V_n = 0$, returns $1.0$ (refuses zero-variance degenerate certification without non-asymptotic term). | **Asymptotic / Heuristic Plug-in (UNPROVEN in finite samples)** |

---

## 2. Verification of Proof Steps Against Potential Failure Modes

### Check 1: Does Adaptive Stopping of the Search Loop Invalidate Conditional Coverage?
- **Question:** Suppose the search procedure decides *whether* to make attempt $k=2$ based on the fact that attempt $k=1$ failed to certify ($U_1(D_1^{\mathrm{conf}}; \alpha_1) > \tau$). Does this conditioning break Theorem 1?
- **Verification:** **No.** In Theorem 1, $D_1^{\mathrm{conf}} \in \mathcal{F}_1$. The decision $\{K \ge 2\}$ and the new candidate $\widehat{\theta}_2$ are $\mathcal{F}_1$-measurable. Conditional on $\mathcal{F}_1$, the second confirmation sample $D_2^{\mathrm{conf}}$ is drawn fresh from $P_{\widehat{\theta}_2}^{\otimes n_2}$. Therefore, $\Pr(E_2 \mid \mathcal{F}_1) \le \alpha_2$ holds pointwise on $\{K \ge 2\}$, and $\Pr(E_1 \cup E_2) \le \Pr(E_1) + \Pr(E_2) \le \alpha_1 + \alpha_2 \le \alpha$.

### Check 2: What Happens If the Same Candidate $\widehat{\theta}_1$ Is Tested Twice on Fresh Samples?
- **Question:** Suppose $\widehat{\theta}_2 = \widehat{\theta}_1$, and the algorithm draws a second fresh batch $D_2^{\mathrm{conf}} \sim P_{\widehat{\theta}_1}^{\otimes n_2}$. Does Theorem 1 still hold?
- **Verification:** **Yes**, provided $D_2^{\mathrm{conf}}$ is evaluated *alone* at level $\alpha_2$ (or if $D_1^{\mathrm{conf}}$ and $D_2^{\mathrm{conf}}$ are pooled via the time-uniform confidence sequence $U_{\mathrm{CS}}$ at level $\alpha_1$). What is **prohibited** is pooling $D_1^{\mathrm{conf}} \cup D_2^{\mathrm{conf}}$ into a fixed-sample bound $U_{\mathrm{EB}}$ without using a time-uniform boundary or pre-specifying the combined sample size!

### Check 3: Why Did `student_t_ucb_heuristic` Require Zero-Variance Protection?
- **Question:** Suppose a distribution places $99\%$ mass on $\ell = 0.20$ and $1\%$ mass on $\ell = 1.00$ (true mean $\mu = 0.208 > \tau = 0.205$). In $n = 30$ samples, with probability $0.99^{30} \approx 0.74$, all $30$ samples equal $0.20$, so $\widehat{\mu}_{30} = 0.20$ and $V_{30} = 0$.
- **Verification:** A naive Student-$t$ interval $\widehat{\mu}_n + t_{1-\alpha, n-1}\sqrt{V_n/n}$ gives $0.20 + 0 = 0.20 \le 0.205$, causing a **74% false-certification rate**! By contrast, `hoeffding_ucb_valid` and `empirical_bernstein_ucb_valid` have a strictly positive $O(R\sqrt{\ln(1/\alpha)/n})$ or $O(R\ln(2/\alpha)/(n-1))$ term even when $V_n = 0$, completely preventing zero-variance collapse. We explicitly verify this adversarial case in Phase 3 (`test_phase3_certification.py` and `phase3_synthetic_calibration.csv`).

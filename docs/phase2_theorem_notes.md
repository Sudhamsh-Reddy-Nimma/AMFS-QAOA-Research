# Phase 2 — Mathematical Notes: Split-Sample Certification, Online Spectral Support Tightening, and Anytime Confidence Sequences

## 1. Framing of the Statistical Framework

We explicitly distinguish between **standard statistical concentration machinery** (Propositions 1 and 3, which adapt established results from Maurer & Pontil 2009 and Waudby-Smith & Ramdas 2024 to the two-stream QAOA oracle) and our **domain-specific QAOA spectral support tightening** (Proposition 2, which exploits the online-computable Delorme–Poljak / Mohar–Poljak Laplacian bound to narrow the bounded loss support from $[0, 1]$ to $[1 - U_{\mathrm{spec}}(G)/W, 1]$).

---

## 2. Proposition 2 (Domain-Specific Technical Contribution): Online Spectrally-Tightened Support Bounds

### 2.1 Deterministic Support Reduction via Zero-Sum Diagonal-Corrected Laplacian Eigenvalues
Let $G = (V, E, w)$ be a weighted graph with $|V| = n$, non-negative edge weights $w_{uv} \ge 0$, total weight $W = \sum_{(u,v)\in E} w_{uv} > 0$, and weighted graph Laplacian $L_G = D_G - A_G$. For any bitstring $z \in \{0,1\}^n$, let $s = 2z - \mathbf{1} \in \{-1, +1\}^n$ be the corresponding spin vector. The cut value satisfies:
$$C(z) = \frac{1}{4} s^\top L_G s.$$
By Mohar & Poljak (*Czechoslovak Math. J.* 40(2):343–352, 1990) and Delorme & Poljak (*Math. Program.* 62:557–574, 1993), for **any** zero-sum diagonal correcting vector $u \in \mathbb{R}^n$ satisfying $\sum_{i=1}^n u_i = 0$, since $s_i^2 = 1$ for all $i \in \{1,\dots,n\}$, we have $\sum_{i=1}^n u_i s_i^2 = \sum_{i=1}^n u_i = 0$, and therefore:
$$C(z) = \frac{1}{4} s^\top \big(L_G + \mathrm{diag}(u)\big) s \le \frac{n}{4}\,\lambda_{\max}\!\big(L_G + \mathrm{diag}(u)\big).$$
Parameterizing $u_n = -\sum_{i=1}^{n-1} u_i$ guarantees that $\sum_{i=1}^n u_i = 0$ holds identically for every $u_{1:n-1} \in \mathbb{R}^{n-1}$. Defining the online polynomial-time spectral upper bound
$$U_{\mathrm{spec}}(G) := \min\!\left(W,\, \frac{n}{4}\lambda_{\max}(L_G),\, \frac{n}{4}\lambda_{\max}\!\big(L_G + \mathrm{diag}(\hat{u})\big)\right)$$
(and $\lfloor U_{\mathrm{spec}}(G) \rfloor$ when all edge weights are unit integers $w_{uv}=1$), we obtain the **deterministic, parameter-independent online support bound**:
$$0 \le C(z) \le U_{\mathrm{spec}}(G) \implies Y(z) = 1 - \frac{C(z)}{W} \in \big[a_{\mathrm{spec}}(G),\, 1\big],$$
where
$$a_{\mathrm{spec}}(G) := 1 - \frac{U_{\mathrm{spec}}(G)}{W} \ge 0, \qquad R_{\mathrm{spec}}(G) := 1 - a_{\mathrm{spec}}(G) = \frac{U_{\mathrm{spec}}(G)}{W} \le 1.$$

### 2.2 Spectrally-Tightened Non-Asymptotic Confidence Bounds
Because $Y_1, \dots, Y_n \in [a_{\mathrm{spec}}(G), 1]$ has deterministic support width $R_{\mathrm{spec}}(G) = U_{\mathrm{spec}}(G)/W$:
1. **Spectrally-Tightened Hoeffding Bound (`hoeffding_spectral_valid`):**
   $$U_n^{\mathrm{Hoeff,spec}}(\alpha_k) = \min\!\left(1,\, \bar{Y}_n + R_{\mathrm{spec}}(G)\sqrt{\frac{\ln(1/\alpha_k)}{2n}}\right),$$
   reducing the sufficient sample size for power $1-\beta$ at margin $\delta$ by the factor $R_{\mathrm{spec}}(G)^2$:
   $$N_{\mathrm{suff}}^{\mathrm{Hoeff,spec}}(\delta, \alpha_k, \beta) = \left\lceil \frac{R_{\mathrm{spec}}(G)^2 \big(\sqrt{\ln(1/\alpha_k)} + \sqrt{\ln(1/\beta)}\big)^2}{2\delta^2} \right\rceil.$$
2. **Spectrally-Tightened Maurer–Pontil Empirical-Bernstein Bound (`eb_spectral_valid`):**
   Applying Maurer & Pontil (*COLT* 2009, Theorem 4) to the affine-normalized variables $Z_s = (Y_s - a_{\mathrm{spec}}(G))/R_{\mathrm{spec}}(G) \in [0, 1]$ (where $V_n(Z) = V_n(Y)/R_{\mathrm{spec}}(G)^2$) and multiplying by $R_{\mathrm{spec}}(G)$ yields:
   $$U_n^{\mathrm{EB,spec}}(\alpha_k) = \min\!\left(1,\, \bar{Y}_n + \sqrt{\frac{2 V_n(Y)\ln(2/\alpha_k)}{n}} + \frac{7 R_{\mathrm{spec}}(G)\ln(2/\alpha_k)}{3(n-1)}\right).$$
3. **Spectrally-Tightened Intersection Bound (`intersection_spectral_valid`):**
   $$U_n^{\mathrm{Int,spec}}(\alpha_k) = \min\!\Big(U_n^{\mathrm{Hoeff,spec}}(\alpha_k/2),\, U_n^{\mathrm{EB,spec}}(\alpha_k/2)\Big).$$

---

## 3. Proposition 3: Anytime-Valid Predictable-Plugin Empirical-Bernstein Confidence Sequence (Fixed Horizon-Invariance)

### 3.1 Waudby-Smith & Ramdas (2024, Proposition 2 / Eqs. 14–15) on $[a_{\mathrm{spec}}(G), 1]$
Let $Y_1, Y_2, \dots \in [a_{\mathrm{spec}}(G), 1]$ be drawn sequentially from the independent confirmation stream, and define $Z_i = \frac{Y_i - a_{\mathrm{spec}}(G)}{R_{\mathrm{spec}}(G)} \in [0, 1]$ with true conditional mean $\mu_Z = \frac{\mu_Y - a_{\mathrm{spec}}(G)}{R_{\mathrm{spec}}(G)} \in [0, 1]$. Let $\mathcal{G}_{i-1} = \sigma(Z_1, \dots, Z_{i-1})$, and define the $\mathcal{G}_{i-1}$-measurable predictable mean and regularized variance estimators:
$$\widehat{\mu}_{Z,0} = \mathrm{clip}\!\left(\frac{0.5 - a_{\mathrm{spec}}(G)}{R_{\mathrm{spec}}(G)},\, 0,\, 1\right), \qquad \widehat{\mu}_{Z,i-1} = \frac{1}{i-1}\sum_{j=1}^{i-1} Z_j \quad (i \ge 2),$$
$$\widehat{\sigma}_{Z,i-1}^2 = \frac{0.25 + \sum_{j=1}^{i-1} (Z_j - \widehat{\mu}_{Z,j})^2}{i}.$$
Let $\lambda_i \in (0, 1/2]$ be a strictly $\mathcal{G}_{i-1}$-measurable predictable tuning sequence:
- **Horizon-free anytime schedule (`n_horizon=None`, Waudby-Smith & Ramdas 2024, Eq. 15):**
  $$\lambda_i = \min\!\left(\frac{1}{2},\, \sqrt{\frac{2\ln(1/\alpha_k)}{i \ln(i+1)\,\widehat{\sigma}_{Z,i-1}^2}}\right).$$
- **Predeclared fixed-horizon schedule (`n_horizon = H`, where $H$ is fixed before $Z_1$ is observed):**
  $$\lambda_i = \min\!\left(\frac{1}{2},\, \sqrt{\frac{2\ln(1/\alpha_k)}{H\,\widehat{\sigma}_{Z,i-1}^2}}\right).$$

**Crucial Invariance Requirement (Resolution of Audit Defect `BUG-M06`):**
For the exponential process
$$M_t(\mu_Z) = \exp\!\left(\sum_{i=1}^t \lambda_i (Z_i - \mu_Z) - \sum_{i=1}^t v_i \psi_E(\lambda_i)\right), \qquad v_i = 4(Z_i - \widehat{\mu}_{Z,i-1})^2, \quad \psi_E(\lambda) = \frac{-\ln(1-\lambda)-\lambda}{4}$$
to form a **single** nonnegative test supermartingale across $t = 1, 2, \dots$ to which Ville's maximal inequality $\Pr(\exists t \ge 1 : M_t(\mu_Z) \ge 1/\alpha_k) \le \alpha_k$ applies, each $\lambda_i$ must never be recomputed using a growing horizon $H(t) = \max(t, 50)$ as $t$ increases. In `codes/src/stopping_rules.py`, `time_uniform_eb_cs_valid` defaults to `n_horizon=None` (horizon-free step-index scaling $i\ln(i+1)$), and `certify_candidate_split_sample` sets `fixed_horizon = max(max_s, 50)` once prior to the sequential loop. The resulting upper confidence sequence
$$U_t^{\mathrm{CS,spec}} = \min\!\left(1,\, a_{\mathrm{spec}}(G) + R_{\mathrm{spec}}(G)\,\min_{1 \le s \le t} \frac{\ln(1/\alpha_k) + \sum_{i=1}^s v_i \psi_E(\lambda_i) + \sum_{i=1}^s \lambda_i Z_i}{\sum_{i=1}^s \lambda_i}\right)$$
is strictly non-increasing in $t$ (verified by `test_10_regression_time_uniform_prefix_invariance_and_monotonicity`) and satisfies $\Pr(\exists t \ge 1 : \mu_Y > U_t^{\mathrm{CS,spec}}) \le \alpha_k$.

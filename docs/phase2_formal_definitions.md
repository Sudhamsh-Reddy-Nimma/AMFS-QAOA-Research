# Phase 2A — Formal Statistical Definitions & Adaptive Search Protocol

**Document Status:** Phase 2 Technical Research Document (Sections 2.1–2.4, 2.9)

---

## 1. Objective and Bounded Loss Convention (Section 2.1)

Let $G = (V, E, w)$ be an undirected weighted graph with vertex set $V = \{0, 1, \dots, n-1\}$, edge set $E$, and **strictly nonnegative** edge weights $w_{uv} \ge 0$ such that the total edge weight satisfies:
$$W_{\mathrm{tot}} = \sum_{(u,v) \in E} w_{uv} > 0.$$

For any computational-basis bitstring $z = (z_0, \dots, z_{n-1}) \in \{0, 1\}^n$, the MaxCut value is:
$$C(z) = \sum_{(u,v) \in E} w_{uv} \mathbf{1}[z_u \neq z_v].$$

### 1.1 Verification of Bounded Loss Support
Because $w_{uv} \ge 0$ for every $(u,v) \in E$ and $\mathbf{1}[z_u \neq z_v] \in \{0, 1\}$, we have:
$$0 = C(0^n) \le C(z) \le \sum_{(u,v)\in E} w_{uv} = W_{\mathrm{tot}} \qquad \forall z \in \{0, 1\}^n.$$
Therefore, the normalized single-shot loss:
$$\ell(z) = 1 - \frac{C(z)}{W_{\mathrm{tot}}}$$
is **deterministically and genuinely bounded in $[0, 1]$** with worst-case support width $R = 1 - 0 = 1.0$.

*Remark on Tighter Support Widths:* Technically, $\ell(z) \in [1 - C_{\max}(G)/W_{\mathrm{tot}}, 1]$, so the true range width is $R_{\mathrm{true}}(G) = C_{\max}(G)/W_{\mathrm{tot}} \le U_{\mathrm{spec}}(G)/W_{\mathrm{tot}} \le 1.0$. However, because $z = 0^n$ and $z = 1^n$ yield $C(z) = 0 \implies \ell(z) = 1.0$, the upper endpoint $1.0$ is always in the support of the quantum state (unless symmetry forces zero amplitude on $0^n, 1^n$), and the lower endpoint is $\ge 1 - U_{\mathrm{spec}}(G)/W_{\mathrm{tot}}$. To avoid any unverified range assumptions, our primary non-asymptotic bounds use the universal support $[0, 1]$ ($R = 1.0$).

### 1.2 Expected Loss and Variance at Parameter $\theta$
For a depth-$p$ QAOA circuit parameterized by $\theta \in \Theta \subset \mathbb{R}^{2p}$ preparing the Born-rule distribution $P_\theta(z)$ (including any stationary quantum channel noise), define:
$$\mu(\theta) = \mathbb{E}_{z \sim P_\theta}[\ell(z)] = 1 - \frac{\mathbb{E}_{z \sim P_\theta}[C(z)]}{W_{\mathrm{tot}}} \in [0, 1], \qquad \sigma^2(\theta) = \mathrm{Var}_{z \sim P_\theta}[\ell(z)] \in \left[0, \frac{1}{4}\right].$$

---

## 2. Precise Definition of the Certification Target & Hypotheses (Section 2.2)

We maintain strictly distinct symbols for all six operational parameters:
1. $\mu(\theta) \in [0, 1]$: True expected loss of candidate $\theta$.
2. $\tau \in (0, 1)$: Predeclared operational target loss threshold (certification aims to establish $\mu(\theta) \le \tau$).
3. $\delta \ge 0$: Indifference-zone tolerance / suboptimality margin parameter (used either when defining a relaxed certification goal or when quantifying confirmation power for a candidate whose true loss satisfies $\mu(\theta) \le \tau - \delta$).
4. $\alpha \in (0, 1)$: Nominal family-wise Type-I false-certification error level.
5. $1 - \beta \in (0, 1)$: Desired statistical certification power when a candidate satisfies $\mu(\theta) \le \tau - \delta$.
6. $B \in \mathbb{Z}_+$: Total quantum measurement shot budget across search and confirmation ($B = B_{\mathrm{search}} + B_{\mathrm{confirmation}}$).

### 2.1 Consistent Formulation of Null, Alternative, and Indifference Region
We distinguish two mathematically distinct certification formulations and never mix their thresholds:

- **Formulation I (Exact Target Certification at Threshold $\tau$):**
  - **Null Hypothesis (Unsafe / Unacceptable Candidate):** $H_0(\theta): \mu(\theta) > \tau$.
  - **Alternative Hypothesis (Target-Attaining Candidate):** $H_1(\theta): \mu(\theta) \le \tau$.
  - **Decision Rule:** Certify candidate $\widehat{\theta}_k$ at attempt $k$ iff $U_{k}(\widehat{\theta}_k; \alpha_k) \le \tau$, where $U_k$ is a valid $(1-\alpha_k)$ one-sided upper confidence bound for $\mu(\widehat{\theta}_k)$.
  - **Role of Indifference Margin $\delta > 0$ in Power Analysis:** Because no finite-sample test can achieve nontrivial power $1-\beta > \alpha$ arbitrarily close to the boundary $\mu(\theta) = \tau$, statistical power is evaluated on the **indifference-zone interior** $\Theta_{\delta} = \{\theta : \mu(\theta) \le \tau - \delta\}$. Here, false certification is still strictly defined as certifying any $\theta$ with $\mu(\theta) > \tau$, while power $1-\beta$ is required whenever $\mu(\theta) \le \tau - \delta$.

- **Formulation II ($\delta$-Relaxed Target Certification at Threshold $\tau + \delta$):**
  - If the practitioner instead defines a false certification only when $\mu(\theta) > \tau + \delta$ (tolerating a $\delta$-relaxation of the target), then the null hypothesis is $H_0^{(\delta)}(\theta): \mu(\theta) > \tau + \delta$, and the decision rule compares $U_k(\widehat{\theta}_k; \alpha_k) \le \tau + \delta$. In this case, the indifference region is $\mu(\theta) \in (\tau, \tau + \delta]$, where certifying is neither required for power nor counted as a strict $> \tau + \delta$ error. In all our experiments, we explicitly label whether a column evaluates **Strict Error ($\mu(\widehat{\theta}) > \tau$)** or **$\delta$-Relaxed Error ($\mu(\widehat{\theta}) > \tau + \delta$)**.

---

## 3. Operational Meaning: Spectral Bound vs Exact MaxCut Optimum (Section 2.9)

Let $C_{\max}(G) = \max_{z \in \{0,1\}^n} C(z)$ be the exact MaxCut value, and let $L_G = D_G - A_G$ be the weighted graph Laplacian with largest eigenvalue $\lambda_{\max}(L_G)$. By Mohar & Poljak (1990), for any graph with nonnegative weights:
$$C_{\max}(G) \le U_{\mathrm{spec}}(G) := \min\!\left(W_{\mathrm{tot}},\, \frac{n}{4}\lambda_{\max}(L_G)\right).$$

Define the **tightness ratio** of the spectral bound on graph $G$:
$$\eta_{\mathrm{spec}}(G) := \frac{U_{\mathrm{spec}}(G)}{C_{\max}(G)} \ge 1.$$

Suppose an operational target requires expected cut at least $\rho_{\mathrm{op}} \cdot U_{\mathrm{spec}}(G)$ for a target fraction $\rho_{\mathrm{op}} \in (0, 1)$, corresponding to operational loss threshold:
$$\tau = 1 - \rho_{\mathrm{op}} \frac{U_{\mathrm{spec}}(G)}{W_{\mathrm{tot}}}.$$
Then the condition $\mu(\theta) \le \tau$ is equivalent to:
$$\mathbb{E}[C(\theta)] \ge \rho_{\mathrm{op}} U_{\mathrm{spec}}(G) \iff r(\theta) := \frac{\mathbb{E}[C(\theta)]}{C_{\max}(G)} \ge \rho_{\mathrm{op}} \cdot \eta_{\mathrm{spec}}(G).$$

**Crucial Mathematical Consequence:**
1. Whenever $U_{\mathrm{spec}}(G) > C_{\max}(G)$ ($\eta_{\mathrm{spec}}(G) > 1$), certifying the spectral operational target $\mu(\theta) \le \tau$ establishes that the retrospective approximation ratio satisfies $r(\theta) \ge \rho_{\mathrm{op}} \eta_{\mathrm{spec}}(G) > \rho_{\mathrm{op}}$ (a **strictly stronger** requirement than $r(\theta) \ge \rho_{\mathrm{op}}$).
2. Conversely, if $\rho_{\mathrm{op}} \eta_{\mathrm{spec}}(G) > r_p^*(G)$ (where $r_p^*(G)$ is the $p$-layer QAOA ceiling), then the spectral target is **physically unattainable** at depth $p$ even if $\rho_{\mathrm{op}} \le r_p^*(G)$!
3. Therefore, operational target attainment ($\mu(\theta) \le \tau$) and retrospective approximation ratio ($r(\theta) = \mathbb{E}[C(\theta)]/C_{\max}(G)$) are **distinct quantities** and are always reported in separate columns.

---

## 4. Adaptive Search Filtration and Conditional Confirmation Protocol (Sections 2.3 & 2.4)

Let the total algorithm run for at most $K \le K_{\max}$ confirmation attempts (where $K$ may be a data-dependent stopping time). For each attempt $k \in \{1, \dots, K_{\max}\}$:
1. **Pre-Confirmation Filtration $\mathcal{F}_{k-1}$:** Contains all search measurements $S_1, \dots, S_{t_k}$, all previously selected candidates $\widehat{\theta}_1, \dots, \widehat{\theta}_{k-1}$, all previous confirmation datasets $D_1^{\mathrm{conf}}, \dots, D_{k-1}^{\mathrm{conf}}$, and the adaptive decision $\mathbf{1}\{K \ge k\}$ to make a $k$-th confirmation attempt at the $\mathcal{F}_{k-1}$-measurable candidate $\widehat{\theta}_k \in \Theta$.
2. **Conditional Confirmation Sampling Assumption (Assumption A1):** Conditional on $\mathcal{F}_{k-1}$ and on the event $\{K \ge k\}$, the $k$-th confirmation dataset $D_k^{\mathrm{conf}} = (\ell_{k,1}, \dots, \ell_{k,n_k})$ consists of $n_k$ fresh observations drawn i.i.d. from $P_{\widehat{\theta}_k}$:
   $$D_k^{\mathrm{conf}} \,\big|\, \mathcal{F}_{k-1}, \{K \ge k\} \;\sim\; P_{\widehat{\theta}_k}^{\otimes n_k}.$$
   *Note:* $D_k^{\mathrm{conf}}$ is **not** marginally independent of $\widehat{\theta}_k$ (since its parameter is $\widehat{\theta}_k$!), nor is $\widehat{\theta}_k$ independent of prior confirmation outcomes $D_{1}^{\mathrm{conf}}, \dots, D_{k-1}^{\mathrm{conf}}$. What is required—and sufficient—is **conditional i.i.d. sampling given $\mathcal{F}_{k-1}$** and **zero leakage of $D_k^{\mathrm{conf}}$ into the choice of $\widehat{\theta}_k$**.
3. **Family-Wise False-Certification Event (Section 2.4):**
   For each attempt $k \ge 1$, define the $k$-th false-certification event:
   $$E_k := \{K \ge k\} \cap \left\{\mu(\widehat{\theta}_k) > \tau\right\} \cap \left\{U_k(D_k^{\mathrm{conf}}; \alpha_k) \le \tau\right\}.$$
   The family-wise false-certification event across the entire optimization run is:
   $$E_{\mathrm{FWER}} := \bigcup_{k=1}^{K_{\max}} E_k = \{\text{at least one candidate with } \mu(\widehat{\theta}_k) > \tau \text{ is certified}\}.$$

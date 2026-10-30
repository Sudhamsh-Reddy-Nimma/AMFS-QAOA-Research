# Phase 5 — Confirmation Information-Flow & Blinding-Condition Specification

**Document Status:** Phase 5 Technical Research Document (Sections 5.1–5.3)

---

## 1. Exact Information-Flow Specification Across Variants A, B, C, and D

Let $\mathcal{F}_{k-1}$ denote all search observations and prior confirmation results available before confirmation attempt $k \in \{1, \dots, K_{\max}\}$. We define four distinct information-flow protocols to test which blinding conditions are mathematically necessary for Theorem 1:

| Variant | Candidate Selection Measurability | Confirmation Sample Generation | Information Revealed Before Certification Decision | Does Theorem 1 Hold? |
| :--- | :--- | :--- | :--- | :--- |
| **Variant A (Valid Single Fresh Confirmation, $K_{\max}=1$)** | $\widehat{\theta}_1$ is selected using **only** search shots $S_1, \dots, S_T \in \mathcal{F}_0$ ($B_{\mathrm{search}}$). | A single fresh confirmation sample $D_1^{\mathrm{conf}}$ of size $n_{\mathrm{conf}}$ is drawn from $P_{\widehat{\theta}_1}$. | $D_1^{\mathrm{conf}}$ is never seen by the search procedure before $\widehat{\theta}_1$ is fixed. | **YES** ($\Pr(E_{\mathrm{FWER}}) \le \alpha$). |
| **Variant B (Reuse of Current Confirmation Observations)** | Search evaluates candidates $\theta_1, \dots, \theta_M$ using measurements that are **pooled directly into** (or identical to) the confirmation sample $D_1^{\mathrm{conf}}$ of the selected winner $\widehat{\theta}_1 = \arg\min_j \widehat{\mu}(D_{1,j}^{\mathrm{conf}})$. | No separate fresh sample is drawn after selecting $\widehat{\theta}_1$; the same $n_{\mathrm{conf}}$ observations that selected $\widehat{\theta}_1$ are used to compute $U_1(D_1^{\mathrm{conf}}; \alpha)$. | Full leakage of $D_1^{\mathrm{conf}}$ into candidate selection. | **NO** ($\widehat{\theta}_1$ is not $\mathcal{F}_0$-measurable; $D_1^{\mathrm{conf}} \mid \widehat{\theta}_1$ is selection-biased downward). |
| **Variant C (Repeated Adaptive Attempts with Fresh Samples, $K_{\max} > 1$)** | At attempt $k$, if attempts $1, \dots, k-1$ failed to certify, the search procedure observes the failure outcomes $D_1^{\mathrm{conf}}, \dots, D_{k-1}^{\mathrm{conf}} \in \mathcal{F}_{k-1}$, continues searching/adapting, and selects a new candidate $\widehat{\theta}_k \in \mathcal{F}_{k-1}$. | A **fresh** confirmation sample $D_k^{\mathrm{conf}} \sim P_{\widehat{\theta}_k}^{\otimes n_k}$ is drawn for attempt $k$ and evaluated at valid $\alpha$-spending level $\alpha_k$ ($\sum_{k=1}^{K_{\max}} \alpha_k \le \alpha$). | Past confirmation samples $D_1^{\mathrm{conf}}, \dots, D_{k-1}^{\mathrm{conf}}$ are known, but **current** $D_k^{\mathrm{conf}}$ is blinded until $\widehat{\theta}_k$ is fixed. | **YES** (Adaptivity across attempts is $\mathcal{F}_{k-1}$-measurable; conditional coverage + union bound holds!). |
| **Variant D (Deliberate Outcome-Dependent Post-Hoc Selection / Cherry-Picking)** | A pool of $M_{\mathrm{pool}}$ candidate parameter vectors is generated, and $n_{\mathrm{conf}}$ confirmation shots are drawn for each candidate. The procedure peeks at all $M_{\mathrm{pool}}$ confirmation UCBs and selects the candidate with the smallest confirmation UCB to report as "certified" at unadjusted level $\alpha$. | Reuses the confirmation UCBs themselves as the selection criterion without a fresh post-selection sample or $M_{\mathrm{pool}}$-way Bonferroni correction. | Complete post-hoc cherry-picking on confirmation data. | **NO** (Effective false-certification probability scales as $1 - (1 - p_{\mathrm{single}})^{M_{\mathrm{pool}}}$). |

---

## 2. Why RNG Object Sharing Is Distinct from Data Leakage (Section 5.3)

As emphasized in Section 5.3, whether `search_shots` and `confirmation_shots` call `rng.choice(...)` on the same underlying Mersenne Twister / PCG64 pseudorandom number generator object or two differently seeded `np.random.Generator` instances has **zero bearing** on mathematical validity, because sequential draws from a PRNG are statistically independent i.i.d. uniforms.
- **True data leakage (Variants B and D)** occurs when the **realized random values** $D_k^{\mathrm{conf}}$ enter the argmin/selection function that chooses $\widehat{\theta}_k$.
- We verify this explicitly in Phase 5 by running Variant A with both a **split PRNG object (`confirm_rng`)** and a **single shared PRNG object (`shared_rng`)** and showing that both preserve exact conditional independence and identical error control.

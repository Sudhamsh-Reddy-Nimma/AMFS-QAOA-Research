# Phase 4 — Calibration, Direct Optimizer Global-Null Audit, & Spectrally-Tightened Cost Analysis

## 1. Experiment 4A-1 & 4A-2: Direct Unfitted Optimizer Audit vs. Controlled $(M, K)$ Ablation

To address the methodological critique that a synthetic $(M, K)$ grid is a constructed model rather than a direct test of the optimizers, Experiment 4A is split into two distinct empirical studies:

### 1.1 Part 4A-1: Direct Unfitted Execution of Actual Optimizers Under Global Nulls (`exp4a_actual_optimizer_insample_audit.csv`)
We run the actual optimizers (`amfs_md_target`, `icans`, `cobyla_hao2025`, `spsa`) on benchmark graphs at $n \in \{8, 12\}$ and depths $p \in \{1, 2\}$ where the target threshold $\tau$ is set strictly below the exact $p$-layer QAOA minimum achievable loss $\mu_{\min}^{(p)}(G) - 0.004$. Because $\mu(\theta) > \tau$ for **every parameter $\theta \in \Theta$**, any certification by any protocol is a guaranteed false certification ($H_0$ holds globally). No parameters $M$ or $K$ are fitted.

| Optimizer ($n=12$) | Depth $p$ | In-Sample Unsplit Normal ($z=1.96$) | Fresh Split Student-$t$ Heuristic | Fresh Split Loose $R=1.0$ Intersection | Fresh Split Spectral $R_{\mathrm{spec}}$ Intersection | Fresh Split Spectral Anytime Betting CS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`amfs_md_target`** | $p=1$ | **`0.7000`** (`[0.5347, 0.8344]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |
| **`cobyla_hao2025`** | $p=1$ | **`0.7250`** (`[0.5611, 0.8540]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |
| **`icans`** | $p=1$ | **`0.1500`** (`[0.0571, 0.2984]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |
| **`spsa`** | $p=1$ | **`0.1500`** (`[0.0571, 0.2984]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |
| **`amfs_md_target`** | $p=2$ | **`0.7750`** (`[0.6155, 0.8916]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |
| **`cobyla_hao2025`** | $p=2$ | **`0.7500`** (`[0.5880, 0.8731]`) | `0.0000` | `0.0000` | **`0.0000`** (`[0.0000, 0.0881]`) | **`0.0000`** (`[0.0000, 0.0881]`) |

### 1.2 Part 4A-2: Controlled Factorial Decomposition of Selection Pool $M$ vs. Repeated Checks $K$ (`exp4a_insample_failure.csv`)
Across $n=12$ 3-regular QAOA distributions ($R_{\mathrm{spec}} = 0.8333$), increasing candidate screening pool size from $M=1$ to $M=40$ with $K=3$ repeated checks inflates in-sample Normal false certifications from `0.0700` to **`0.8167`**, and in-sample unsplit Empirical-Bernstein false certifications to **`0.1100`**, whereas fresh split-sample confirmation with `intersection_spectral_valid` and `time_uniform_eb_cs_spectral_valid` maintains **`0.0000`–`0.0033` $\le \alpha = 0.05$** across all $(M, K)$.

---

## 2. Experiment 4B: Active Calibration Across Worst-Case, Spectrally-Tightened, and Deliberately Tight Bounds (`false_certification_calibration.csv`)

Across 720 calibration cells on $n=12$ QAOA distributions ($\alpha \in \{0.01, 0.05, 0.10\}$, $K_{\max} \in \{1, 2\}$, $n_{\mathrm{conf}} \in \{50, 150, 400\}$):
- All 8 valid non-asymptotic bounds—both worst-case $R=1.0$ (`hoeffding_valid`, `empirical_bernstein_valid`, `intersection_eb_hoeff_valid`, `time_uniform_eb_cs_valid`) and online spectrally-tightened $R=R_{\mathrm{spec}}(G)$ (`hoeffding_spectral_valid`, `eb_spectral_valid`, `intersection_spectral_valid`, `time_uniform_eb_cs_spectral_valid`)—maintain **zero calibration discrepancy (`0.0000`)** above $\alpha$, with `time_uniform_eb_cs_spectral_valid` exhibiting non-zero active boundary detections (`mean 0.0003`, `max 0.0067` at $\alpha=0.05$, `max 0.0233` at $\alpha=0.10$) strictly below nominal $\alpha$.
- Conversely, the deliberately tight under-scaled bound (`eb_under_range_r05_invalid`) and asymptotic `student_t_heuristic` actively exceed nominal $\alpha$ on boundary nulls (`max 0.0833` at $\alpha=0.05$ and `0.1467` at $\alpha=0.10$ in Phase 4B, and **`0.0925` (`eb_under_range_r05_invalid`) / `0.0725` (`student_t_heuristic`)** on peaked near-optimal QAOA nulls in Phase 5).

---

## 3. Experiment 4C: How Online Spectral Support Tightening Changes the Confirmation Cost (`confirmation_power_cost.csv`)

Replacing the loose worst-case range $R = 1.0$ with the online Delorme–Poljak / Mohar–Poljak spectral range $R_{\mathrm{spec}}(G) = U_{\mathrm{spec}}(G)/W$ (`0.7917` on $n=12$ Erdős–Rényi QAOA, `0.7500` on bounded Beta) substantially reduces confirmation shot requirements while preserving 100% finite-sample validity:

| Margin $\delta$ | Certification Rule | Support Width Used $R$ | Theoretical Sufficient Shots ($80\%$ Power) | Empirical Shots for $\ge 80\%$ Power | Shot Reduction from Spectral Tightening |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **`0.02`** | `hoeffding_valid` (Loose) | `1.0000` | `11,246` | $> 1,500$ | — |
| **`0.02`** | `hoeffding_spectral_valid` (Spectral) | **`0.7917`** | **`7,049`** | $> 1,500$ | **`-37.3%` theoretical shots** |
| **`0.02`** | `intersection_eb_hoeff_valid` (Loose) | `1.0000` | `1,245` | `1,500` | — |
| **`0.02`** | `intersection_spectral_valid` (Spectral) | **`0.7917`** | **`1,090`** | **`1,000`** | **`-33.3%` empirical shots (`1500 -> 1000`)** |
| **`0.02`** | `time_uniform_eb_cs_valid` (Loose) | `1.0000` | `1,245` | `600` | — |
| **`0.02`** | `time_uniform_eb_cs_spectral_valid` (Spectral) | **`0.7917`** | **`1,090`** | **`400`** | **`-33.3%` empirical shots (`600 -> 400`)** |
| **`0.05`** | `hoeffding_valid` (Loose) | `1.0000` | `1,800` | `1,000` | — |
| **`0.05`** | `hoeffding_spectral_valid` (Spectral) | **`0.7917`** | **`1,128`** | **`600`** | **`-40.0%` empirical shots (`1000 -> 600`)** |
| **`0.05`** | `intersection_eb_hoeff_valid` (Loose) | `1.0000` | `365` | `400` | — |
| **`0.05`** | `intersection_spectral_valid` (Spectral) | **`0.7917`** | **`310`** | **`300`** | **`-25.0%` empirical shots (`400 -> 300`)** |
| **`0.10`** | `hoeffding_valid` (Loose) | `1.0000` | `450` | `200` | — |
| **`0.10`** | `hoeffding_spectral_valid` (Spectral) | **`0.7917`** | **`282`** | **`150`** | **`-25.0%` empirical shots (`200 -> 150`)** |

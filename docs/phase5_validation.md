# Phase 5 — Confirmation-Data Reuse & Blinding-Condition Validation (Active-Sensitivity Upgrade)

## 1. Resolving the Conservatism Masking Problem in Phase 5

In the initial $n=8, R=1.0$ experiment on high-variance bimodal graphs ($\sigma^2 \approx 0.074$), the Maurer–Pontil $O(R\ln(2/\alpha)/n)$ bias penalty at $n_{\mathrm{conf}}=150$ was so large ($\approx 0.141$) that even deliberately broken blinding protocols (Variant B: reusing screening data; Variant D: cherry-picking the best UCB across $M$ candidates) recorded `0.0000` false certifications under valid bounds.

To provide a sharp, informative test that distinguishes valid blinding (Variants A and C) from broken blinding (Variants B and D) **even when using mathematically valid non-asymptotic bounds**, the upgraded Phase 5 experiment (`codes/experiments/run_phase5_experiments.py`, `results_paper/metrics/confirmation_reuse_ablation.csv`) evaluates both:
1. **Interior QAOA Regime (`interior_qaoa_n12_nconf150`)** on $n=12$ Erdős–Rényi graphs ($R_{\mathrm{spec}} = 0.8500$), and
2. **Peaked Near-Optimal QAOA Regime (`peaked_near_optimal_qaoa_n12_nconf180`)** on $n=12$ graphs where $92.5\%$ of measurement outcomes land near the low-loss floor ($a_{\mathrm{spec}} + 0.02$) and $7.5\%$ land on excited cuts ($1.0$), so $\mu_{\mathrm{null}} > \tau_{\mathrm{null}}$ by $+0.002$, and we also include the deliberately tight diagnostic bound `eb_under_range_r05_invalid` ($R=0.45$).

---

## 2. Empirical Results in the Peaked Near-Optimal Regime (`confirmation_reuse_ablation.csv`)

| Protocol Variant | Blinding Valid? | `time_uniform_eb_cs_spectral_valid` False-Cert Rate ($H_0$) | `time_uniform_eb_cs_spectral_valid` Power ($H_1$) | `eb_under_range_r05_invalid` False-Cert Rate ($H_0$) | `student_t_heuristic` False-Cert Rate ($H_0$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Fresh Confirmation, Split RNG)** | **Yes** | **`0.0100`** (`[0.0027, 0.0254]`) $\le 0.05$ | **`0.9950`** | **`0.0925`** ($> 0.05$) | **`0.0725`** ($> 0.05$) |
| **Variant A (Fresh Confirmation, Shared RNG)** | **Yes** | **`0.0050`** (`[0.0006, 0.0179]`) $\le 0.05$ | **`0.9850`** | **`0.0800`** ($> 0.05$) | **`0.0650`** ($> 0.05$) |
| **Variant B (Reuse Screening Data in Confirmation)** | **No** | **`0.1925`** (`[0.1550, 0.2346]`) $\gg 0.05$ | `1.0000` | **`0.5125`** ($\gg 0.05$) | **`0.5650`** ($\gg 0.05$) |
| **Variant C (Adaptive Repeated Fresh, Bonferroni $K=3$)** | **Yes** | **`0.0100`** (`[0.0027, 0.0254]`) $\le 0.05$ | **`0.9925`** | **`0.1250`** ($> 0.05$) | **`0.1100`** ($> 0.05$) |
| **Variant C (Adaptive Repeated Fresh, Harmonic $K=3$)** | **Yes** | **`0.0025`** (`[0.0001, 0.0138]`) $\le 0.05$ | **`0.9300`** | **`0.0600`** ($> 0.05$) | **`0.0600`** ($> 0.05$) |
| **Variant D (Outcome-Dependent Cherry-Pick, $M=25$)** | **No** | **`0.2100`** (`[0.1711, 0.2532]`) $\gg 0.05$ | `1.0000` | **`0.8300`** ($\gg 0.05$) | **`0.8425`** ($\gg 0.05$) |

---

## 3. Key Diagnostic Conclusions

1. **Broken Blinding Invalidates Even Mathematically Valid Confidence Sequences:**
   - Under `time_uniform_eb_cs_spectral_valid` (which is mathematically proven to control Type-I error at $\alpha = 0.05$ under valid blinding), **Variant A (Split RNG and Shared RNG)** and **Variant C (Bonferroni and Harmonic $\alpha$-spending)** achieve empirical false-certification rates of **`0.25%–1.00%` ($\le 5\%$)** while achieving **`93.00%–99.50%` power** under the alternative.
   - When the blinding condition is violated by reusing screening draws inside the confirmation sample (**Variant B**) or cherry-picking across $M=25$ candidates (**Variant D**), the empirical false-certification rate of the **exact same valid bound** (`time_uniform_eb_cs_spectral_valid`) jumps to **`19.25%` (Variant B)** and **`21.00%` (Variant D)**—nearly $4\times$ the nominal $\alpha = 0.05$ ceiling!
2. **Deliberately Tight Bound (`eb_under_range_r05_invalid`) Fails Even Under Valid Blinding (Variant A):**
   - Under **Variant A (Valid Fresh Split RNG)**, `eb_under_range_r05_invalid` commits **`9.25%` false certifications (`95% CI [0.0660, 0.1252]`)** and `student_t_heuristic` commits **`7.25%` (`95% CI [0.0491, 0.1025]`)**, confirming that the calibration test actively detects overly tight bounds even when sample splitting is respected.
3. **Shared Pseudo-Random Generator Object vs. Statistical Data Leakage:**
   - Across all 6 rules and both regimes, **Variant A (Split RNG)** and **Variant A (Shared RNG)** have statistically indistinguishable false-certification rates (`0.0100` vs `0.0050` for `time_uniform_eb_cs_spectral_valid`; `0.0925` vs `0.0800` for `eb_under_range_r05_invalid`), confirming that statistical validity depends on drawing fresh post-selection observations rather than instantiating two `np.random.Generator` objects.

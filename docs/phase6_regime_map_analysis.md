# Phase 6 — Matched-Split & Multi-Scale ($n=8\dots 16, p=1\dots 4$) Joint Regime Map Analysis

## 1. Eliminating the Confirmation-Allocation Tautology

In our initial Phase 6 design, `fixed_shaydulin_fullconf` was given up to $450$ confirmation shots while adaptive search methods (`amfs_md_target`, `icans`, `cobyla_hao2025`, `bobyqa_trust`) were given only $100\text{–}200$ confirmation shots. Because a larger confirmation sample shrinks the confidence bound $U_n(\hat{\theta})$ as $O(1/\sqrt{n_{\mathrm{conf}}})$, that comparison conflated **method quality** with **confirmation shot allocation**.

In the upgraded Phase 6 benchmark (`codes/experiments/run_phase6_regime_map.py`), we enforce three rigorous controls across **68 deduplicated non-isomorphic graphs** spanning $n \in \{8, 12, 14, 16\}$ and circuit depths $p \in \{1, 2, 3, 4\}$:
1. **Matched Confirmation Split (`Evaluation_A_EndToEnd` with Matched $40\%$ Confirmation Reserve):** Every primary method—including zero-search parameter transfer (`fixed_shaydulin_rescaled`, `fixed_wurtz_love`) and all adaptive search methods (`amfs_md_target`, `cobyla_hao2025`, `icans`, `wecans`, `bobyqa_trust`, `dds`, `gpbo_tibaldi`, `spsa`)—receives the **exact same confirmation shot allocation** $B_{\mathrm{confirmation}} = 0.40 B$ ($120$ shots at $B=300$, $240$ shots at $B=600$, $480$ shots at $B=1200$). We retain `fixed_shaydulin_fullconf` ($\rho_{\mathrm{conf}} \le 450/B$) and `icans_starved_conf` ($B_{\mathrm{confirmation}} = 15$) solely as explicit allocation-boundary controls.
2. **Search/Confirmation Split-Ratio Optimization (`split_ratio_optimization.csv`):** We sweep $\rho_{\mathrm{conf}} = B_{\mathrm{confirmation}}/B \in \{0.25, 0.40, 0.60, 0.80\}$ on $n=12$ graphs at $B \in \{600, 1200\}$ for every core method.
3. **Multi-Scale ($n \in \{8, 12, 14, 16\}$) and Multi-Depth ($p \in \{1, 2, 3, 4\}$) Generalization (`scaling_n12_to_16_p1_to_4_summary.csv`):** We evaluate how true approximation ratio, optimality gap to the exact $p$-layer ceiling, and spectrally-tightened vs. loose ($R=1.0$) certification probability scale with qubit count and circuit depth.

---

## 2. Matched-Split vs. Unmatched Allocation Results (`joint_regime_map.csv`)

### 2.1 Overall $p=1$ Comparison Across Budgets $B \in \{300, 600, 1200\}$ ($n \in \{8, 12\}$)

| Total Budget $B$ | Method | Search Shots | Confirmation Shots | Mean True Ratio $r(\hat{\theta})$ | Strict Spectral CS Cert Prob ($\delta_{\mathrm{tol}}=0.00$) | Spectral Maurer–Pontil EB Cert Prob ($\delta_{\mathrm{tol}}=0.05$) | Mean Spectral UCB Loss |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`300`** | `icans` (Matched $40\%$ Conf) | `180` | **`120`** | `0.7717` | `0.4062` | **`0.5625`** | `0.3653` |
| **`300`** | `wecans` (Matched $40\%$ Conf) | `180` | **`120`** | `0.7713` | `0.4375` | **`0.5625`** | `0.3653` |
| **`300`** | `amfs_md_target` (Matched $40\%$ Conf) | `180` | **`120`** | **`0.7789`** | `0.4062` | `0.4688` | `0.3594` |
| **`300`** | `cobyla_hao2025` (Matched $40\%$ Conf) | `180` | **`120`** | `0.7705` | `0.3750` | `0.4688` | `0.3644` |
| **`300`** | `fixed_shaydulin_rescaled` (Matched $40\%$ Conf) | `0` | **`120`** | **`0.7829`** | **`0.4688`** | `0.5312` | **`0.3530`** |
| **`300`** | `fixed_wurtz_love` (Matched $40\%$ Conf) | `0` | **`120`** | `0.7739` | `0.2812` | `0.4062` | `0.3653` |
| **`300`** | *`fixed_shaydulin_fullconf` (Unmatched $100\%$ Conf)* | *`0`* | *`300`* | *`0.7829`* | *`0.8438`* | *`1.0000`* | *`0.3243`* |
| **`300`** | *`icans_starved_conf` (Starved $15$-Shot Conf)* | *`284`* | *`15`* | *`0.7729`* | *`0.0000`* | *`0.0000`* | *`0.6781`* |
| **`600`** | `icans` (Matched $40\%$ Conf) | `360` | **`240`** | `0.7815` | `0.6667` | `0.9375` | `0.3379` |
| **`600`** | `wecans` (Matched $40\%$ Conf) | `360` | **`240`** | **`0.7831`** | `0.6875` | **`0.9792`** | `0.3367` |
| **`600`** | `amfs_md_target` (Matched $40\%$ Conf) | `360` | **`240`** | `0.7802` | `0.6667` | `0.9583` | `0.3410` |
| **`600`** | `bobyqa_trust` (Matched $40\%$ Conf) | `360` | **`240`** | `0.7798` | `0.6042` | `0.9375` | `0.3402` |
| **`600`** | `cobyla_hao2025` (Matched $40\%$ Conf) | `360` | **`240`** | `0.7790` | `0.6458` | `0.9167` | `0.3423` |
| **`600`** | `fixed_shaydulin_rescaled` (Matched $40\%$ Conf) | `0` | **`240`** | **`0.7871`** | **`0.7500`** | **`0.9792`** | **`0.3329`** |
| **`600`** | `fixed_wurtz_love` (Matched $40\%$ Conf) | `0` | **`240`** | `0.7744` | `0.6875` | `0.9167` | `0.3454` |
| **`1200`** | `bobyqa_trust` (Matched $40\%$ Conf) | `662.5` | **`480`** | **`0.7821`** | **`0.9688`** | **`1.0000`** | **`0.3181`** |
| **`1200`** | `icans` (Matched $40\%$ Conf) | `720` | **`480`** | `0.7818` | **`0.9688`** | **`1.0000`** | `0.3198` |
| **`1200`** | `amfs_md_target` (Matched $40\%$ Conf) | `720` | **`480`** | `0.7792` | `0.9062` | **`1.0000`** | `0.3221` |
| **`1200`** | `fixed_shaydulin_rescaled` (Matched $40\%$ Conf) | `0` | **`480`** | **`0.7829`** | **`1.0000`** | **`1.0000`** | **`0.3176`** |
| **`1200`** | `fixed_wurtz_love` (Matched $40\%$ Conf) | `0` | **`480`** | `0.7739` | `0.9375` | **`1.0000`** | `0.3276` |

**What the Matched-Split Comparison Proves:**
- At $B = 300$, giving `fixed_shaydulin_fullconf` all $300$ shots for confirmation yields an `84.38%` strict certification rate, whereas giving `fixed_shaydulin_rescaled` the **matched $120$ confirmation shots** yields **`46.88%`** (`tol=0.00`) and **`53.12%`** (`EB tol=0.05`)—where `icans` (`56.25%` EB) and `wecans` (`56.25%` EB) actually slightly edge out transfer!
- Thus, the large certification gap between `fixed_shaydulin_fullconf` and adaptive search at small $B$ is **primarily a confirmation-budget allocation effect (`300` vs `120` shots)** rather than parameter-quality superiority.

---

## 3. Where Adaptive Search Beats Parameter Transfer Under Matched Confirmation Splits

When we break down the matched-split results by **graph topology, weight heterogeneity, and circuit depth $p$** (`joint_regime_map.csv`, `scaling_n12_to_16_p1_to_4_summary.csv`), a sharp regime boundary emerges:

1. **Non-Regular Graphs (`erdos_renyi`, $p=2$, Matched Confirmation Shots):**
   - Because Wurtz–Lykov / Shaydulin angles are derived from locally tree-like 3-regular subgraphs, on non-regular Erdős–Rényi graphs at $p=2$, `fixed_shaydulin_rescaled` and `fixed_wurtz_love` plateau at **`0.8339` true ratio** and **`87.50%` strict certification (`tol=0.00`)**.
   - Adaptive search methods refine the non-regular landscape and beat transfer under the **exact same confirmation budget**: `icans` reaches **`0.8623` (`+0.0284` over transfer)** and **`100.00%` certification**, `bobyqa_trust` reaches **`0.8494` (`+0.0155`)** and **`100.00%` certification**, and `amfs_md_target` reaches **`0.8386`** and **`100.00%` certification**.
2. **Bimodal Weighted Graphs (`weighted_3_regular_bimodal`, $p=1, 2$, Matched Confirmation Shots):**
   - Unrescaled universal transfer (`fixed_wurtz_love`) fails on bimodal edge weights (`0.7883` at $p=1$, `0.7884` at $p=2$, with only `71.43%–75.00%` certification), whereas `amfs_md_target` (`0.8163` at $p=1$, `0.8830` at $p=2$), `cobyla_hao2025` (`0.8869` at $p=2$), `icans` (`0.8162` at $p=1$, `0.8837` at $p=2$), and RMS-rescaled transfer (`0.8239` at $p=1$, `0.8843` at $p=2$) recover high approximation ratios and `89.29%–100.00%` certification.
3. **Deeper Circuits ($p=3, 4$) and Larger System Sizes ($n \in \{12, 14, 16\}$) (`scaling_n12_to_16_p1_to_4_summary.csv`):**
   - Across $n \in \{8, 12, 14, 16\}$, the exact $p$-layer QAOA ceiling rises monotonically from **`0.7798` ($p=1$)** $\to$ **`0.8591` ($p=2$)** $\to$ **`0.9044` ($p=3$)** $\to$ **`0.9356` ($p=4$)**.
   - At $p=3$ and $p=4$, adaptive search methods (`spsa` reaching `0.8971` at $p=3$ and `0.9258` at $p=4$; `icans` reaching `0.8937` at $p=3$ and **`0.8303` average ratio at $n=16$** vs `0.8265` for `fixed_shaydulin_rescaled`) outperform zero-search transfer on non-regular instances.
   - Across all $n \in \{8, 12, 14, 16\}$ and $p \in \{1, 2, 3, 4\}$, online spectral support tightening (`R_spec` averaging `0.895–0.913` across all families and `0.786–0.850` on Erdős–Rényi / Watts–Strogatz / Barabási–Albert) maintains **`0.0000` false certifications**.

---

## 4. Search vs. Confirmation Split-Ratio Optimization (`split_ratio_optimization.csv`)

Sweeping $\rho_{\mathrm{conf}} = B_{\mathrm{confirmation}}/B \in \{0.25, 0.40, 0.60, 0.80\}$ on $n=12$ graphs at $B=600$ and $B=1200$ demonstrates the exact trade-off between search quality and confirmation tightness:
- When $\rho_{\mathrm{conf}}$ increases from $0.25$ ($150$ confirmation shots at $B=600$) to $0.60$ ($360$ confirmation shots at $B=600$), the mean certified UCB loss `ucb_spectral_tu` drops from **`0.3541` to `0.3283`** for `amfs_md_target` and from **`0.3512` to `0.3340`** for `icans`, even though `icans` and `bobyqa_trust` achieve slightly higher true approximation ratios at $\rho_{\mathrm{conf}} = 0.25$ (`0.7837` and `0.7829`) than at $\rho_{\mathrm{conf}} = 0.80$ (`0.7727` and `0.7743`).
- Across all adaptive optimizers at $B \in [600, 1200]$, allocating **$\rho_{\mathrm{conf}}^* \in [0.40, 0.60]$** ($40\%\text{–}60\%$ of the total budget to fresh confirmation) is the empirical sweet spot that funds enough search iterations to refine parameters while funding enough confirmation shots ($240\text{–}720$) to make the spectrally-tightened non-asymptotic UCB radius smaller than $\delta = 0.03\text{–}0.05$.

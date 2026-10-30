"""Independent re-verification of Phase 7 from the raw CSVs (does not import analyze_phase7.py).

Checks
  7A  counts are internally consistent; every FRESH valid rule is <= alpha in every strict-null cell;
      the in-sample z=1.96 arm's headline number is recomputed with its own Clopper-Pearson interval.
  7B  resource accounting (search + confirmation <= budget) holds for all trials; the ER p>=2 paired
      comparison is recomputed from per-trial rows with a fresh Wilcoxon test.
  7C  valid rules are <= alpha under Variant A and D_bonf in every cell (their validity under correct
      blinding / correction); the largest valid-rule rate under B/D is recomputed.
Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, wilcoxon

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "results_raw" / "phase7"
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def cp_hi(k, n, a=0.05):
    return 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))


def cp_lo(k, n, a=0.05):
    return 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))


# ---------------- 7A
a = pd.read_csv(RAW / "exp7a_selection_bias_counts.csv")
check("7A: n_false + n_true == n_certified in every row", ((a.n_false_cert + a.n_true_cert) == a.n_certified).all())
check("7A: 40 seeds per (graph, budget, optimizer) cell", (a.n_trials == 40).all())
strict = a[a.delta_vs_global_min < 0]
fresh_valid = strict[strict.arm.isin(["fresh_hoeffding_spectral_valid", "fresh_eb_spectral_valid", "fresh_time_uniform_spectral_valid", "fresh_time_uniform_r1_valid"])]
rate = fresh_valid.groupby(["optimizer", "delta_vs_global_min", "arm"]).apply(lambda x: x.n_false_cert.sum() / x.n_trials.sum())
check("7A: every valid fresh rule <= 0.05 pooled in all strict-null (optimizer, delta, rule) cells", (rate <= 0.05).all(), f"max={rate.max():.4f}")
z = strict[(strict.arm == "insample_z196") & (strict.optimizer == "amfs_md_target") & (strict.delta_vs_global_min == -0.002)]
k, n = int(z.n_false_cert.sum()), int(z.n_trials.sum())
print(f"      amfs_md_target in-sample z=1.96 @ delta=-0.002: {k}/{n} = {k / n:.4f}, CP95 [{cp_lo(k, n):.4f}, {cp_hi(k, n):.4f}]")
check("7A: that cell's CI lower bound is above 0.05", cp_lo(k, n) > 0.05)
for opt in ["cobyla_hao2025", "icans", "spsa"]:
    zz = strict[(strict.arm == "insample_z196") & (strict.optimizer == opt)]
    r = zz.groupby("delta_vs_global_min").apply(lambda x: x.n_false_cert.sum() / x.n_trials.sum()).max()
    print(f"      {opt:15s} in-sample z=1.96 worst strict-null rate: {r:.4f}")

# ---------------- 7B
b = pd.read_csv(RAW / "exp7b_transfer_vs_adaptive_trials.csv")
check("7B: search + confirmation shots <= total budget in all trials", ((b.search_shots + b.confirmation_shots) <= b.total_budget).all())
er = b[(b.family == "erdos_renyi") & (b.p_depth == 2)]
g = er.groupby(["graph_idx", "method"]).true_ratio.mean().unstack()
d = (g["icans"] - g["transfer_matched40"])
print(f"      ER p=2 icans - matched transfer: mean {d.mean():+.4f}, wins {(d > 0).sum()}/{len(d)}, Wilcoxon p={wilcoxon(d).pvalue:.2e}")
check("7B: ER p=2 icans beats matched transfer on every one of the 20 fresh graphs", (d > 0).all())
rr = b[(b.family == "random_3_regular") & (b.p_depth == 1) & (b.total_budget == 300)]
g2 = rr.groupby(["graph_idx", "method"]).true_ratio.mean().unstack()
check("7B: random_3_regular p=1 B=300 transfer >= each adaptive method on average",
      all((g2[m] - g2["transfer_matched40"]).mean() <= 0 for m in ["icans", "spsa", "cobyla_hao2025", "amfs_md_target"]))

# ---------------- 7C
c = pd.read_csv(RAW / "exp7c_blinding_dose_response.csv")
valid = ["time_uniform_eb_cs_spectral_valid", "time_uniform_eb_cs_valid", "eb_spectral_valid", "intersection_spectral_valid", "hoeffding_spectral_valid"]
pc = c.groupby(["variant", "M", "n_conf", "rule", "delta"], as_index=False).agg(k=("n_false_cert", "sum"), n=("reps", "sum"))
pc["rate"] = pc.k / pc.n
va = pc[pc.rule.isin(valid)]
check("7C: valid rules <= 0.05 under Variant A in every cell", (va[va.variant == "A"].rate <= 0.05).all(), f"max={va[va.variant == 'A'].rate.max():.4f}")
check("7C: valid rules <= 0.05 under D_bonf (Bonferroni-corrected) in every cell", (va[va.variant == "D_bonf"].rate <= 0.05).all(), f"max={va[va.variant == 'D_bonf'].rate.max():.4f}")
bd = va[va.variant.isin(["B_s0", "B_s05", "D"])]
w = bd.sort_values("rate", ascending=False).head(1).iloc[0]
print(f"      largest valid-rule false-cert under B/D: {w.rate:.4f} ({w.variant}, M={int(w.M)}, n_conf={int(w.n_conf)}, {w.rule}, delta={w.delta})")
print("\nALL PHASE 7 VERIFICATION CHECKS PASSED" if ok else "\nSOME PHASE 7 CHECKS FAILED")
sys.exit(0 if ok else 1)

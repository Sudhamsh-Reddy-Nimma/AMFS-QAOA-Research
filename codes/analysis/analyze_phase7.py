"""Phase 7 analysis -- applies the verdict rules that were pre-specified in the headers of
run_phase7a_selection_bias.py, run_phase7b_transfer_vs_adaptive.py and run_phase7c_blinding_dose_response.py.

Reads  results_raw/phase7/*.csv
Writes results_paper/metrics/phase7*.csv, results_paper/figures/fig_phase7*.png,
       results_paper/metrics/phase7_claim_verdicts.json
No numbers are typed by hand; every figure in docs/phase7_claim_validation.md comes from these files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent.parent
RAW = BASE / "results_raw" / "phase7"
MET = BASE / "results_paper" / "metrics"
FIG = BASE / "results_paper" / "figures"
MET.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
VERDICTS: dict = {}


def cp(k: int, n: int, a: float = 0.05):
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pvals[i] * (m - rank)))
        adj[i] = run
    return adj


# ===================================================================== 7A
def analyze_7a():
    f = RAW / "exp7a_selection_bias_counts.csv"
    if not f.exists():
        print("[7A] raw file missing"); return
    d = pd.read_csv(f)
    in_arms = [a for a in d.arm.unique() if a.startswith("insample")]
    fresh_valid = ["fresh_hoeffding_spectral_valid", "fresh_eb_spectral_valid", "fresh_time_uniform_spectral_valid", "fresh_time_uniform_r1_valid"]

    def pool(keys):
        g = d.groupby(keys + ["arm", "delta_vs_global_min"], as_index=False).agg(
            n_trials=("n_trials", "sum"), n_certified=("n_certified", "sum"), n_false_cert=("n_false_cert", "sum"),
            n_true_cert=("n_true_cert", "sum"), n_claim_true=("n_selected_theta_claim_true", "sum"))
        g["false_cert_rate"] = g.n_false_cert / g.n_trials
        g["true_cert_rate"] = g.n_true_cert / g.n_trials
        ci = [cp(int(k), int(n)) for k, n in zip(g.n_false_cert, g.n_trials)]
        g["false_ci95_low"] = [c[0] for c in ci]
        g["false_ci95_high"] = [c[1] for c in ci]
        return g

    by_opt = pool(["optimizer"])
    by_opt.to_csv(MET / "phase7a_pooled_by_optimizer.csv", index=False)
    by_cfg = pool(["family", "p_depth", "total_budget", "optimizer"])
    by_cfg.to_csv(MET / "phase7a_by_config.csv", index=False)
    all_pool = pool([])
    all_pool.to_csv(MET / "phase7a_pooled_all.csv", index=False)

    strict = by_opt[by_opt.delta_vs_global_min < 0]
    # (a) some optimizer has in-sample z arm CI lower bound > alpha at some strict-null delta
    za = strict[strict.arm == "insample_z196"]
    a_hit = za[za.false_ci95_low > ALPHA]
    cond_a = len(a_hit) > 0
    # (b) every valid fresh rule has point estimate <= alpha in ALL strict-null (optimizer, delta) cells
    fv = strict[strict.arm.isin(fresh_valid)]
    cond_b = bool((fv.false_cert_rate <= ALPHA).all())
    # also the stricter per-(config,budget,optimizer) version, reported for transparency
    fv_cfg = by_cfg[(by_cfg.delta_vs_global_min < 0) & (by_cfg.arm.isin(fresh_valid))]
    worst_cfg = float(fv_cfg.false_cert_rate.max())
    # (c) at least one valid fresh rule has true-cert rate >= 0.20 at some delta > 0 (pooled over everything)
    pos = all_pool[(all_pool.delta_vs_global_min > 0) & (all_pool.arm.isin(fresh_valid))]
    cond_c = bool((pos.true_cert_rate >= 0.20).any())
    best_pos = pos.sort_values("true_cert_rate", ascending=False).head(1)[["arm", "delta_vs_global_min", "true_cert_rate"]].to_dict("records")
    VERDICTS["claim1"] = {
        "cond_a_insample_z_exceeds_alpha_for_some_optimizer": bool(cond_a),
        "cond_a_hits": a_hit[["optimizer", "delta_vs_global_min", "false_cert_rate", "false_ci95_low", "false_ci95_high", "n_trials"]].round(4).to_dict("records"),
        "cond_b_all_valid_fresh_rules_le_alpha_pooled": cond_b,
        "cond_b_worst_pooled_cell": float(fv.false_cert_rate.max()),
        "cond_b_worst_per_config_cell_(informational)": worst_cfg,
        "cond_c_valid_fresh_rule_nonvacuous_power": cond_c,
        "cond_c_best": best_pos,
        "VERDICT": "SUPPORTED" if (cond_a and cond_b and cond_c) else "NOT_SUPPORTED_AS_STATED",
    }
    # per-optimizer view of the in-sample z arm vs fresh arm, strict null
    view = strict[strict.arm.isin(["insample_z196", "insample_eb_spectral_valid", "insample_tu_spectral_valid",
                                    "fresh_student_t_heuristic", "fresh_time_uniform_spectral_valid"])]
    view.to_csv(MET / "phase7a_strict_null_view.csv", index=False)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deltas = sorted(strict.delta_vs_global_min.unique())
        fig, axes = plt.subplots(1, len(deltas), figsize=(4.2 * len(deltas), 4), sharey=True)
        arms = ["insample_z196", "insample_eb_spectral_valid", "insample_tu_spectral_valid",
                "fresh_student_t_heuristic", "fresh_time_uniform_spectral_valid"]
        opts = sorted(strict.optimizer.unique())
        for ax, dl in zip(np.atleast_1d(axes), deltas):
            for i, arm in enumerate(arms):
                sub = strict[(strict.delta_vs_global_min == dl) & (strict.arm == arm)].set_index("optimizer").loc[opts]
                x = np.arange(len(opts)) + (i - 2) * 0.16
                ax.bar(x, sub.false_cert_rate, width=0.16, label=arm.replace("_", " "),
                       yerr=[sub.false_cert_rate - sub.false_ci95_low, sub.false_ci95_high - sub.false_cert_rate], capsize=1.5)
            ax.axhline(ALPHA, color="k", ls="--", lw=1)
            ax.set_xticks(range(len(opts))); ax.set_xticklabels([o.replace("_", "\n") for o in opts], fontsize=7)
            ax.set_title(f"tau = global_min {dl:+.3f}")
        np.atleast_1d(axes)[0].set_ylabel("false-certification rate (same tau for every arm)")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        plt.tight_layout(); plt.savefig(FIG / "fig_phase7a_selection_bias.png", dpi=130); plt.close()
    except Exception as e:  # pragma: no cover
        print("figure 7A skipped:", e)


# ===================================================================== 7B
def analyze_7b():
    f = RAW / "exp7b_transfer_vs_adaptive_trials.csv"
    if not f.exists():
        print("[7B] raw file missing"); return
    t = pd.read_csv(f)
    g = t.groupby(["family", "p_depth", "total_budget", "graph_idx", "method"], as_index=False).agg(
        true_ratio=("true_ratio", "mean"), cert=("cert_tu_spectral_tol00", "mean"), cert05=("cert_tu_spectral_tol05", "mean"),
        claim_true=("claim_true_tol00", "mean"))
    adaptive = ["icans", "spsa", "cobyla_hao2025", "amfs_md_target"]
    rows = []
    rng = np.random.default_rng(0)
    for (fam, p, B), sub in g.groupby(["family", "p_depth", "total_budget"]):
        piv = sub.pivot(index="graph_idx", columns="method", values="true_ratio")
        pivc = sub.pivot(index="graph_idx", columns="method", values="cert")
        for m in adaptive + ["transfer_fullbudget"]:
            d = (piv[m] - piv["transfer_matched40"]).dropna()
            if len(d) == 0:
                continue
            try:
                pw = wilcoxon(d, zero_method="wilcox").pvalue if (d != 0).any() else 1.0
            except Exception:
                pw = np.nan
            boot = rng.choice(d.values, (5000, len(d))).mean(axis=1)
            dc = (pivc[m] - pivc["transfer_matched40"]).dropna()
            rows.append(dict(family=fam, p_depth=p, total_budget=B, method=m, n_graphs=len(d),
                             mean_true_ratio=float(piv[m].mean()), transfer_matched_mean_true_ratio=float(piv["transfer_matched40"].mean()),
                             mean_diff_vs_transfer_matched=float(d.mean()), ci95_low=float(np.percentile(boot, 2.5)), ci95_high=float(np.percentile(boot, 97.5)),
                             graphs_better=int((d > 0).sum()), graphs_worse=int((d < 0).sum()), wilcoxon_p=float(pw),
                             mean_cert_tol00=float(pivc[m].mean()), transfer_matched_cert_tol00=float(pivc["transfer_matched40"].mean()),
                             mean_cert_diff=float(dc.mean())))
    r = pd.DataFrame(rows)
    adaptive_rows = r[r.method.isin(adaptive)].copy()
    # H2c: Holm over ALL adaptive-vs-matched-transfer cells
    adaptive_rows["holm_p_all_cells"] = holm(adaptive_rows.wilcoxon_p.fillna(1.0).values)
    r = r.merge(adaptive_rows[["family", "p_depth", "total_budget", "method", "holm_p_all_cells"]], how="left", on=["family", "p_depth", "total_budget", "method"])
    # H2a: Holm over the 4 pre-specified tests
    h2a = r[(r.family == "erdos_renyi") & (r.p_depth.isin([2, 3])) & (r.method.isin(["icans", "spsa"]))].copy()
    h2a["holm_p_H2a"] = holm(h2a.wilcoxon_p.fillna(1.0).values)
    r = r.merge(h2a[["family", "p_depth", "total_budget", "method", "holm_p_H2a"]], how="left", on=["family", "p_depth", "total_budget", "method"])
    r.to_csv(MET / "phase7b_paired_true_ratio.csv", index=False)
    cond_h2a = bool(len(h2a) == 4 and (h2a.holm_p_H2a < ALPHA).all() and (h2a.mean_diff_vs_transfer_matched > 0).all())
    # H2b
    h2b = r[(r.family == "random_3_regular") & (r.p_depth == 1) & (r.method.isin(adaptive))]
    cond_h2b = bool((h2b.mean_diff_vs_transfer_matched <= 0).all())
    # H2c
    sig_pos = adaptive_rows[(adaptive_rows.holm_p_all_cells < ALPHA) & (adaptive_rows.mean_diff_vs_transfer_matched > 0)]
    sig_neg = adaptive_rows[(adaptive_rows.holm_p_all_cells < ALPHA) & (adaptive_rows.mean_diff_vs_transfer_matched < 0)]
    n_cells = len(adaptive_rows)
    # H2d: certification at B<=600 (p=1): full-budget transfer vs adaptive
    cert_rows = []
    for (fam, p, B), sub in g[(g.p_depth == 1)].groupby(["family", "p_depth", "total_budget"]):
        piv = sub.pivot(index="graph_idx", columns="method", values="cert")
        for m in adaptive:
            d = (piv["transfer_fullbudget"] - piv[m]).dropna()
            cert_rows.append(dict(family=fam, p_depth=p, total_budget=B, adaptive_method=m, fullbudget_transfer_cert=float(piv["transfer_fullbudget"].mean()),
                                  adaptive_cert=float(piv[m].mean()), diff=float(d.mean()), graphs_transfer_better=int((d > 0).sum()), graphs_adaptive_better=int((d < 0).sum())))
    cdf = pd.DataFrame(cert_rows)
    cdf.to_csv(MET / "phase7b_cert_fullbudget_transfer_vs_adaptive.csv", index=False)
    pooled = cdf.groupby(["total_budget", "adaptive_method"], as_index=False).agg(
        fullbudget_transfer_cert=("fullbudget_transfer_cert", "mean"), adaptive_cert=("adaptive_cert", "mean"))
    cond_h2d = bool((pooled.fullbudget_transfer_cert >= pooled.adaptive_cert - 1e-12).all())
    VERDICTS["claim2"] = {
        "H2a_ER_p2_p3_icans_spsa_beat_matched_transfer_after_holm": cond_h2a,
        "H2a_detail": h2a[["p_depth", "method", "mean_true_ratio", "transfer_matched_mean_true_ratio", "mean_diff_vs_transfer_matched", "graphs_better", "n_graphs", "wilcoxon_p", "holm_p_H2a"]].round(5).to_dict("records"),
        "H2b_regular_p1_transfer_not_beaten": cond_h2b,
        "H2b_detail": h2b[["total_budget", "method", "mean_diff_vs_transfer_matched", "graphs_better", "graphs_worse", "wilcoxon_p"]].round(5).to_dict("records"),
        "H2c_cells_total": int(n_cells), "H2c_cells_adaptive_sig_better": int(len(sig_pos)), "H2c_cells_adaptive_sig_worse": int(len(sig_neg)),
        "H2c_sig_better_cells": sig_pos[["family", "p_depth", "total_budget", "method", "mean_diff_vs_transfer_matched", "holm_p_all_cells"]].round(5).to_dict("records"),
        "H2d_fullbudget_transfer_cert_ge_every_adaptive_at_p1": cond_h2d,
        "VERDICT_narrowed_claim_(ER, p>=2)": "SUPPORTED" if cond_h2a else "NOT_SUPPORTED",
        "VERDICT_general_claim_(all regimes)": "SUPPORTED" if len(sig_pos) > n_cells / 2 else "NOT_SUPPORTED",
    }
    # compact table by family/regime: best/mean adaptive minus transfer
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fams = sorted(adaptive_rows.family.unique())
        regs = sorted(set(zip(adaptive_rows.p_depth, adaptive_rows.total_budget)))
        fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 3.8), sharey=True)
        for ax, fam in zip(np.atleast_1d(axes), fams):
            for j, m in enumerate(adaptive):
                sub = adaptive_rows[(adaptive_rows.family == fam) & (adaptive_rows.method == m)].set_index(["p_depth", "total_budget"]).loc[regs]
                x = np.arange(len(regs)) + (j - 1.5) * 0.2
                ax.bar(x, sub.mean_diff_vs_transfer_matched, width=0.2, label=m,
                       yerr=[sub.mean_diff_vs_transfer_matched - sub.ci95_low, sub.ci95_high - sub.mean_diff_vs_transfer_matched], capsize=1.5)
            ax.axhline(0, color="k", lw=1)
            ax.set_xticks(range(len(regs))); ax.set_xticklabels([f"p={p}\nB={B}" for p, B in regs], fontsize=7)
            ax.set_title(fam.replace("_", " "), fontsize=9)
        np.atleast_1d(axes)[0].set_ylabel("true ratio: adaptive - matched transfer\n(20 fresh graphs/family, 95% bootstrap CI)")
        np.atleast_1d(axes)[-1].legend(fontsize=6)
        plt.tight_layout(); plt.savefig(FIG / "fig_phase7b_transfer_vs_adaptive.png", dpi=130); plt.close()
    except Exception as e:  # pragma: no cover
        print("figure 7B skipped:", e)


# ===================================================================== 7C
def analyze_7c():
    f = RAW / "exp7c_blinding_dose_response.csv"
    if not f.exists():
        print("[7C] raw file missing"); return
    d = pd.read_csv(f)
    states = d.state_id.nunique()
    keys = ["variant", "M", "n_conf", "rule", "delta"]
    g = d.groupby(keys, as_index=False).agg(n_false_cert=("n_false_cert", "sum"), reps=("reps", "sum"))
    g["false_cert_rate"] = g.n_false_cert / g.reps
    ci = [cp(int(k), int(n)) for k, n in zip(g.n_false_cert, g.reps)]
    g["ci95_low"] = [c[0] for c in ci]
    g["ci95_high"] = [c[1] for c in ci]
    valid = ["time_uniform_eb_cs_spectral_valid", "time_uniform_eb_cs_valid", "eb_spectral_valid", "intersection_spectral_valid", "hoeffding_spectral_valid"]
    g["rule_is_valid"] = g.rule.isin(valid)
    g.to_csv(MET / "phase7c_pooled.csv", index=False)
    # per-state-family view
    gf = d.groupby(["family", "p_depth"] + keys, as_index=False).agg(n_false_cert=("n_false_cert", "sum"), reps=("reps", "sum"))
    gf["false_cert_rate"] = gf.n_false_cert / gf.reps
    gf.to_csv(MET / "phase7c_by_family_depth.csv", index=False)
    st = d.groupby("state_id").agg(family=("family", "first"), p_depth=("p_depth", "first"), mu_ref=("mu_ref", "first"), sd_ref=("sd_ref", "first"), ratio_ref=("ratio_ref", "first")).reset_index()
    st.to_csv(MET / "phase7c_state_summary.csv", index=False)

    a = g[(g.variant == "A") & g.rule_is_valid]
    a_ok = bool((a.false_cert_rate <= ALPHA).all())
    bonf = g[(g.variant == "D_bonf")]
    bonf_ok = bool((bonf.false_cert_rate <= ALPHA).all())
    viol = g[g.rule_is_valid & g.variant.isin(["B_s0", "B_s05", "D"]) & (g.ci95_low > ALPHA)].copy()
    # a violation only counts for the verdict if the same rule is <=alpha under A and under D_bonf in the same (n_conf, delta)
    counted = []
    for _, r in viol.iterrows():
        if r.rule not in ("time_uniform_eb_cs_spectral_valid", "eb_spectral_valid"):
            continue
        ra = g[(g.variant == "A") & (g.rule == r.rule) & (g.n_conf == r.n_conf) & (g.delta == r.delta)].false_cert_rate.iloc[0]
        rb = g[(g.variant == "D_bonf") & (g.rule == r.rule) & (g.n_conf == r.n_conf) & (g.delta == r.delta) & (g.M == r.M)]
        rb = rb.false_cert_rate.iloc[0] if len(rb) else np.nan
        if ra <= ALPHA and (np.isnan(rb) or rb <= ALPHA):
            counted.append(dict(r))
    cdf = pd.DataFrame(counted)
    small = cdf[cdf.M <= 25] if len(cdf) else cdf
    if len(small) > 0:
        verdict = "SUPPORTED"
    elif len(cdf) > 0:
        verdict = "SUPPORTED_ONLY_LARGE_POOLS"
    else:
        verdict = "NOT_SUPPORTED"
    diag = g[(~g.rule_is_valid) & (g.variant == "A")]
    VERDICTS["claim3"] = {
        "valid_rules_le_alpha_under_variant_A_everywhere": a_ok,
        "worst_valid_rule_rate_under_A": float(a.false_cert_rate.max()),
        "valid_rules_le_alpha_under_D_bonf_everywhere": bonf_ok,
        "worst_rate_D_bonf": float(bonf.false_cert_rate.max()),
        "n_valid_rule_cells_exceeding_alpha_(CI_low>0.05)_under_B_or_D": int(len(viol)),
        "n_such_cells_counted_for_verdict": int(len(cdf)),
        "counted_cells_M<=25": small[["variant", "M", "n_conf", "rule", "delta", "false_cert_rate", "ci95_low", "ci95_high"]].round(4).head(40).to_dict("records") if len(small) else [],
        "counted_cells_M=100_only": cdf[cdf.M == 100][["variant", "M", "n_conf", "rule", "delta", "false_cert_rate", "ci95_low", "ci95_high"]].round(4).head(40).to_dict("records") if len(cdf) else [],
        "max_false_cert_any_valid_rule_under_B_or_D_M<=25": float(g[g.rule_is_valid & g.variant.isin(["B_s0", "B_s05", "D"]) & (g.M <= 25)].false_cert_rate.max()),
        "max_false_cert_any_valid_rule_under_B_or_D_M=100": float(g[g.rule_is_valid & g.variant.isin(["B_s0", "B_s05", "D"]) & (g.M == 100)].false_cert_rate.max()),
        "diagnostic_rules_under_A_max": diag.groupby("rule").false_cert_rate.max().round(4).to_dict(),
        "VERDICT": verdict,
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, nc in zip(axes, [150, 300, 600]):
            for var, ls in [("D", "-"), ("B_s0", "--"), ("B_s05", ":"), ("D_bonf", "-.")]:
                for rule, col in [("time_uniform_eb_cs_spectral_valid", "C0"), ("eb_spectral_valid", "C1"), ("student_t_heuristic", "C3")]:
                    s = g[(g.variant == var) & (g.rule == rule) & (g.n_conf == nc) & (g.delta == 0.002)]
                    if var == "D_bonf" and rule == "student_t_heuristic":
                        continue
                    a1 = g[(g.variant == "A") & (g.rule == rule) & (g.n_conf == nc) & (g.delta == 0.002)]
                    s = pd.concat([a1.assign(M=1), s]).sort_values("M")
                    ax.plot(s.M, s.false_cert_rate, ls, color=col, marker="o", ms=3, label=f"{var} / {rule.replace('_', ' ')}" if nc == 150 else None)
            ax.axhline(ALPHA, color="k", lw=1, ls="--")
            ax.set_xscale("log"); ax.set_xlabel("pool size M"); ax.set_title(f"real QAOA states, n_conf={nc}, delta=0.002")
        axes[0].set_ylabel("false-certification rate (strict null)")
        axes[0].legend(fontsize=5)
        plt.tight_layout(); plt.savefig(FIG / "fig_phase7c_blinding_dose_response.png", dpi=130); plt.close()
    except Exception as e:  # pragma: no cover
        print("figure 7C skipped:", e)


if __name__ == "__main__":
    which = sys.argv[1:] or ["7a", "7b", "7c"]
    if "7a" in which: analyze_7a()
    if "7b" in which: analyze_7b()
    if "7c" in which: analyze_7c()
    out = MET / "phase7_claim_verdicts.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(VERDICTS)
    out.write_text(json.dumps(existing, indent=2, default=str))
    print(json.dumps(VERDICTS, indent=2, default=str))

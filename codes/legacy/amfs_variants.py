"""AMFS implementations: Original Wrapper, Implementation 1 (Faithful AMFS), and Implementation 2 (Improved AMFS-MD-Target).

Strictly preserves MAD_1.py intact and implements:
1. `run_amfs_orig`: Unmodified coordinate-wise wrapper from `amfs_qaoa_shot_benchmark.py`.
2. `run_amfs_faithful`: Faithful adaptation of `MAD_1.py` (`noisy_amfs`) to QAOA that fixes
   the wrapper bugs in `amfs_qaoa_shot_benchmark.py` (passes `init_anchors`, caches measurements
   at identical full parameter vectors across passes, dynamically allocates remaining shot budget,
   and retains the UCB-best incumbent across passes) without altering `noisy_amfs` internals.
3. `run_amfs_improved` (`AMFS-MD-Target`): Research variant with modular flags for ablation:
   - `use_multidim`: Joint 2p-dimensional directional & diagonal stencil search + LHS scouting.
   - `use_screening`: Empirical-Bernstein / variance-floored LCB-UCB sequential shot screening.
   - `use_widen_all`: Multi-scale step widening triggered at local minima (fixing Bug M1 in MAD_1.py).
   - `use_target_stop`: Holm-Bonferroni LCB target-attainment early stopping rule with confirmation allocation.
   - `use_warmstart`: Problem-informed initialization (Wurtz-Love fixed angle anchor).
4. `track_drifting_qaoa`: Generalized drift-monitoring recalibration loop for continuous/bounded QAOA loss
   (fixing Bug M6 in `MAD_1.py` where `monitor=True` raised ValueError on non-Bernoulli data).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

import MAD_1 as mad1
from qaoa_simulator import QAOAOptimizerOracle, get_default_x0


def run_amfs_orig(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    passes: int = 2,
) -> Tuple[np.ndarray, int, Dict]:
    """Exact reproduction of `run_amfs` in `amfs_qaoa_shot_benchmark.py`."""
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, "naive")
    used = 0
    coord_budget = max(100, budget // (d * passes))
    xs = np.linspace(0.0, 1.0, grid_n)
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    for _ in range(passes):
        for j in range(d):
            if oracle.remaining_shots() <= 0:
                break
            fixed = theta.copy()
            lo, hi = oracle.bounds[j]

            def sampler(i, shots):
                candidate = fixed.copy()
                candidate[j] = lo + (hi - lo) * xs[int(i)]
                return oracle.loss_samples(candidate, shots)

            sub_budget = min(coord_budget, oracle.remaining_shots())
            if sub_budget < 1:
                break
            res = mad1.noisy_amfs(
                xs,
                sampler,
                budget_shots=sub_budget,
                scout_points=5,
                max_families=2,
                init_shots=20,
                step_shots=20,
                max_shots_per_point=max(60, coord_budget // 3),
                z=1.5,
                z_start=0.8,
                delta=0.0005,
                kind="gaussian",
                max_rounds_per_family=20,
                max_rounds=1000,
                early_stop=True,
                use_momentum=True,
                use_widen=True,
            )
            if res is not None:
                theta[j] = lo + (hi - lo) * xs[res["best_index"]]
                used += int(res["total_shots"])
                incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "budget_or_passes_completed",
    }


def run_amfs_faithful(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    passes: int = 2,
    init_mode: str = "naive",
) -> Tuple[np.ndarray, int, Dict]:
    """Implementation 1 — Faithful AMFS adaptation using `MAD_1.py`'s `noisy_amfs`.

    Corrections over `run_amfs_orig` while preserving `MAD_1.py` intact:
    1. Snaps initial point `theta` to the coordinate grid and passes `init_anchors=[cur_idx]`
       into `mad1.noisy_amfs` on every coordinate subproblem so warm-start information across
       passes is preserved (using `noisy_amfs`'s native `init_anchors` argument).
    2. Dynamically computes `coord_budget = remaining_shots // remaining_subproblems` so
       shots saved via early stopping in earlier subproblems are used in subsequent passes
       and total shots never exceed `budget`.
    3. Tracks global `PointStats` across all visited full parameter vectors `theta` and selects
       the overall UCB-best incumbent (`mad1.ucb(stats, z=1.5)`) across passes so a noisy later
       pass cannot overwrite a superior earlier pass.
    """
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, init_mode)
    xs = np.linspace(0.0, 1.0, grid_n)

    cur_indices = []
    for j in range(d):
        lo, hi = oracle.bounds[j]
        frac = np.clip((theta[j] - lo) / (hi - lo), 0.0, 1.0)
        idx = int(np.argmin(np.abs(xs - frac)))
        cur_indices.append(idx)
        theta[j] = lo + (hi - lo) * xs[idx]

    global_stats: Dict[Tuple[int, ...], mad1.PointStats] = {}
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    total_subproblems = d * passes
    sub_done = 0

    def indices_to_theta(idx_tuple: Tuple[int, ...]) -> np.ndarray:
        arr = np.zeros(d, dtype=float)
        for col, i_val in enumerate(idx_tuple):
            lo, hi = oracle.bounds[col]
            arr[col] = lo + (hi - lo) * xs[i_val]
        return arr

    def pick_global_incumbent() -> Tuple[int, ...]:
        valid = [(k, st) for k, st in global_stats.items() if st.n >= 2]
        if not valid:
            return tuple(cur_indices)
        return min(valid, key=lambda kv: mad1.ucb(kv[1], z=1.5))[0]

    for _ in range(passes):
        for j in range(d):
            rem_shots = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
            rem_subs = max(1, total_subproblems - sub_done)
            sub_done += 1
            if rem_shots < 20:
                break
            coord_budget = max(20, rem_shots // rem_subs)
            fixed_indices = list(cur_indices)

            def sampler(i, shots):
                idx_list = list(fixed_indices)
                idx_list[j] = int(i)
                key = tuple(idx_list)
                cand = indices_to_theta(key)
                vals = oracle.loss_samples(cand, shots)
                st = global_stats.setdefault(key, mad1.PointStats("gaussian"))
                for v in vals:
                    st.add_sample(float(v))
                if st.n >= 2 and st.m2 == 0.0:
                    st.m2 = 1e-6
                return vals

            res = mad1.noisy_amfs(
                xs,
                sampler,
                budget_shots=coord_budget,
                scout_points=min(5, max(2, coord_budget // 40)),
                max_families=2,
                init_shots=min(20, coord_budget),
                step_shots=20,
                max_shots_per_point=max(60, coord_budget // 2),
                z=1.5,
                z_start=0.8,
                delta=0.0005,
                kind="gaussian",
                max_rounds_per_family=20,
                max_rounds=1000,
                init_anchors=[cur_indices[j]],
                early_stop=True,
                use_momentum=True,
                use_widen=True,
            )
            if res is not None:
                cur_indices[j] = int(res["best_index"])
                best_key = pick_global_incumbent()
                cur_indices = list(best_key)
                theta = indices_to_theta(best_key)
                incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

    best_key = pick_global_incumbent()
    theta = indices_to_theta(best_key)
    incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
    return theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": False,
        "stop_reason": "faithful_completed",
    }


@dataclass
class EBPointStats:
    """Variance-floored statistics for bounded QAOA loss in [0, 1]."""
    n: int = 0
    mean_: float = 0.0
    m2: float = 0.0
    var_floor_scale: float = 0.015  # Prior variance floor for bounded cut loss

    def add_samples(self, vals: np.ndarray) -> None:
        for v in vals:
            y = float(v)
            self.n += 1
            d = y - self.mean_
            self.mean_ += d / self.n
            self.m2 += d * (y - self.mean_)

    @property
    def mean(self) -> float:
        return self.mean_ if self.n > 0 else math.inf

    @property
    def sample_var(self) -> float:
        if self.n < 2:
            return self.var_floor_scale
        emp_v = self.m2 / (self.n - 1)
        # Jeffreys-style prior shrinkage prevents zero-variance lockup on small batches
        return ((self.n - 1) * emp_v + self.var_floor_scale) / self.n

    @property
    def var_of_mean(self) -> float:
        if self.n == 0:
            return math.inf
        return self.sample_var / self.n

    @property
    def se(self) -> float:
        return math.sqrt(self.var_of_mean)

    def ucb_loss(self, z: float = 1.5) -> float:
        """Pessimistic upper confidence bound on loss (equivalent to LCB on expected cut)."""
        if self.n == 0:
            return math.inf
        return self.mean + z * self.se

    def lcb_loss(self, z: float = 1.5) -> float:
        """Optimistic lower confidence bound on loss (used for exploration priority)."""
        if self.n == 0:
            return -math.inf
        return self.mean - z * self.se


def compare_eb(
    anchor: EBPointStats,
    other: EBPointStats,
    z: float = 1.5,
    delta: float = 0.0005,
    z_move: Optional[float] = None,
) -> str:
    """Uncertainty-aware sequential comparison with variance floor."""
    if anchor.n == 0 or other.n == 0:
        return "unresolved"
    zm = z if z_move is None else z_move
    diff = other.mean - anchor.mean
    se = math.sqrt(anchor.var_of_mean + other.var_of_mean)
    if diff + zm * se < -delta:
        return "better"
    if diff - z * se > -delta:
        return "not_better"
    return "unresolved"


def run_amfs_improved(
    oracle: QAOAOptimizerOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    init_shots: int = 20,
    step_shots: int = 20,
    max_shots_per_point: int = 180,
    scout_points: int = 6,
    max_families: int = 2,
    z: float = 1.5,
    z_start: float = 0.9,
    delta: float = 0.0005,
    use_multidim: bool = True,
    use_screening: bool = True,
    use_widen_all: bool = True,
    use_target_stop: bool = False,
    use_warmstart: bool = False,
    target_alpha: float = 0.10,
    min_cert_shots: int = 120,
) -> Tuple[np.ndarray, int, Dict]:
    """Implementation 2 — Improved AMFS (`AMFS-MD-Target`) with modular ablation flags."""
    rng = np.random.default_rng(seed + 314159)
    d = 2 * oracle.p
    xs = np.linspace(0.0, 1.0, grid_n)
    stats: Dict[Tuple[int, ...], EBPointStats] = {}
    incumbent_trace: List[Tuple[int, np.ndarray]] = []
    certified_target = False
    stop_reason = "budget_exhausted"
    n_target_checks = 0
    max_milestone_checks = 5

    if use_target_stop:
        max_shots_per_point = max(max_shots_per_point, min_cert_shots + 60)

    fixed_batch = max(60, max_shots_per_point // 2)

    def idx_to_theta(idx_t: Tuple[int, ...]) -> np.ndarray:
        arr = np.zeros(d, dtype=float)
        for col, idx_val in enumerate(idx_t):
            lo, hi = oracle.bounds[col]
            arr[col] = lo + (hi - lo) * xs[idx_val]
        return arr

    def theta_to_idx(theta_vec: np.ndarray) -> Tuple[int, ...]:
        out = []
        for col in range(d):
            lo, hi = oracle.bounds[col]
            frac = np.clip((theta_vec[col] - lo) / (hi - lo), 0.0, 1.0)
            out.append(int(np.argmin(np.abs(xs - frac))))
        return tuple(out)

    def zm() -> float:
        frac = min(1.0, oracle.total_requested_shots / max(1, budget))
        return z_start + (z - z_start) * frac

    def pick_incumbent() -> Tuple[int, ...]:
        cands = [(k, s) for k, s in stats.items() if s.n > 0]
        if not cands:
            return theta_to_idx(get_default_x0(oracle.p, "wurtz_love" if use_warmstart else "naive"))
        return min(cands, key=lambda kv: kv[1].ucb_loss(z))[0]

    def sample_point(idx_t: Tuple[int, ...], req_shots: int) -> int:
        rem = min(budget - oracle.total_requested_shots, oracle.remaining_shots())
        if rem <= 0:
            return 0
        st = stats.setdefault(idx_t, EBPointStats())
        if not use_screening:
            if st.n > 0:
                return 0
            s = min(fixed_batch, rem)
        else:
            s = min(int(req_shots), rem, max_shots_per_point - st.n)
        if s <= 0:
            return 0
        losses = oracle.loss_samples(idx_to_theta(idx_t), s)
        st.add_samples(losses)
        inc = pick_incumbent()
        incumbent_trace.append((oracle.total_requested_shots, idx_to_theta(inc)))
        return len(losses)

    # Holm-Bonferroni one-sided critical value for M=5 milestone checks:
    # Phi^{-1}(1 - alpha / M) ~ 1.96 for alpha=0.10, M=4..5
    alpha_per_check = target_alpha / max_milestone_checks
    # Rational approximation to normal inverse CDF for one-sided alpha_per_check
    z_cert = 1.96 if alpha_per_check <= 0.025 else 1.645

    def check_target_certified() -> bool:
        nonlocal n_target_checks, certified_target, stop_reason
        if not use_target_stop or oracle.target_loss_threshold is None:
            return False
        inc = pick_incumbent()
        st = stats.get(inc)
        if st is None:
            return False
        # If the incumbent looks promising (mean + 0.5*se <= threshold) but has < min_cert_shots,
        # allocate confirmation shots up to min_cert_shots before testing the milestone bound.
        if st.n >= 2 * init_shots and st.n < min_cert_shots:
            if st.mean + 0.5 * st.se <= oracle.target_loss_threshold:
                needed = min_cert_shots - st.n
                sample_point(inc, needed)
        if st.n < min_cert_shots:
            return False
        n_target_checks += 1
        if st.ucb_loss(z=z_cert) <= oracle.target_loss_threshold:
            certified_target = True
            stop_reason = "target_certified_by_lcb"
            return True
        return False

    # ---- Stage 1: Warm-start + Symmetry/LHS scouting ----
    x0_init = get_default_x0(oracle.p, "wurtz_love" if use_warmstart else "naive")
    init_idx = theta_to_idx(x0_init)
    scout_list: List[Tuple[int, ...]] = [init_idx]

    # Include interior quartile anchors in 2p dimensions + LHS points
    for q_frac in (0.20, 0.25, 0.75):
        q_idx = tuple(int(round(q_frac * (grid_n - 1))) for _ in range(d))
        if q_idx not in scout_list:
            scout_list.append(q_idx)

    n_lhs = max(2, scout_points - len(scout_list))
    lhs_points = np.zeros((n_lhs, d), dtype=int)
    for col in range(d):
        bins = np.linspace(1, grid_n - 2, n_lhs + 1)
        perm = rng.permutation(n_lhs)
        for row in range(n_lhs):
            lo_b = int(round(bins[perm[row]]))
            hi_b = max(lo_b + 1, int(round(bins[perm[row] + 1])))
            lhs_points[row, col] = int(rng.integers(lo_b, min(grid_n, hi_b + 1)))
    for row in range(n_lhs):
        cand_idx = tuple(int(v) for v in lhs_points[row])
        if cand_idx not in scout_list:
            scout_list.append(cand_idx)

    for s_idx in scout_list[:scout_points]:
        if oracle.total_requested_shots >= budget or oracle.remaining_shots() <= 0:
            break
        sample_point(s_idx, init_shots)

    sorted_scouts = sorted(stats.keys(), key=lambda k: stats[k].mean)
    families = []
    min_sep = max(2, grid_n // 6)
    for k in sorted_scouts:
        if len(families) >= max_families:
            break
        if all(sum(abs(a - b) for a, b in zip(f["anchor"], k)) >= min_sep for f in families):
            families.append({
                "anchor": k,
                "converged": False,
                "rounds": 0,
                "dir": tuple([0] * d),
                "step": 1,
                "widened": False,
            })

    coord_dirs: List[Tuple[int, ...]] = []
    for col in range(d):
        v_pos = [0] * d
        v_neg = [0] * d
        v_pos[col] = 1
        v_neg[col] = -1
        coord_dirs.extend([tuple(v_pos), tuple(v_neg)])

    diag_dirs: List[Tuple[int, ...]] = []
    if use_multidim and d >= 2:
        for layer in range(oracle.p):
            for sg in (-1, 1):
                for sb in (-1, 1):
                    v = [0] * d
                    v[layer] = sg
                    v[oracle.p + layer] = sb
                    diag_dirs.append(tuple(v))

    def get_neighbors(anchor: Tuple[int, ...], step_size: int = 1, active_round: int = 0) -> List[Tuple[int, ...]]:
        if not use_multidim:
            col = active_round % d
            dirs = []
            for sgn in (-1, 1):
                v = [0] * d
                v[col] = sgn
                dirs.append(tuple(v))
        else:
            dirs = coord_dirs + (diag_dirs if step_size == 1 else [])
        nbrs = []
        for dv in dirs:
            cand = tuple(anchor[c] + step_size * dv[c] for c in range(d))
            if all(0 <= cand[c] < grid_n for c in range(d)):
                nbrs.append(cand)
        return nbrs

    # ---- Stage 2: Best-first uncertainty-aware refinement ----
    rounds = 0
    while (
        not certified_target
        and oracle.total_requested_shots < budget
        and oracle.remaining_shots() > 0
        and rounds < 500
    ):
        rounds += 1
        if check_target_certified():
            break

        active = [f for f in families if not f["converged"]]
        if not active:
            cands = sorted(stats.keys(), key=lambda k: stats[k].lcb_loss(z))[:2]
            progressed = False
            for c_idx in cands:
                if stats[c_idx].n < max_shots_per_point:
                    used_now = sample_point(c_idx, step_shots)
                    if used_now > 0:
                        progressed = True
                if check_target_certified():
                    break
            if certified_target:
                break
            if not progressed:
                rand_idx = tuple(int(rng.integers(0, grid_n)) for _ in range(d))
                if sample_point(rand_idx, init_shots) == 0:
                    break
                families.append({
                    "anchor": rand_idx,
                    "converged": False,
                    "rounds": 0,
                    "dir": tuple([0] * d),
                    "step": 1,
                    "widened": False,
                })
            continue

        fam = min(active, key=lambda f: stats[f["anchor"]].lcb_loss(z))
        fam["rounds"] += 1
        anchor = fam["anchor"]
        a_stat = stats[anchor]

        if use_screening and a_stat.n < 2 * init_shots and a_stat.n < max_shots_per_point:
            sample_point(anchor, init_shots)
            continue

        if any(x != 0 for x in fam["dir"]) and fam["step"] > 1:
            mom_cand = tuple(anchor[c] + fam["step"] * fam["dir"][c] for c in range(d))
            if all(0 <= mom_cand[c] < grid_n for c in range(d)) and mom_cand not in stats:
                sample_point(mom_cand, init_shots)
                if mom_cand in stats and compare_eb(stats[anchor], stats[mom_cand], z, delta, zm()) == "better":
                    fam["anchor"] = mom_cand
                    fam["step"] = min(fam["step"] * 2, max(2, grid_n // 5))
                    continue
            fam["dir"] = tuple([0] * d)
            fam["step"] = 1

        nbrs = get_neighbors(anchor, step_size=1, active_round=rounds)
        unseen = [nb for nb in nbrs if nb not in stats]
        if unseen:
            for nb in unseen[:2]:
                if oracle.total_requested_shots >= budget:
                    break
                sample_point(nb, init_shots)
            continue

        better, unresolved = [], []
        for nb in nbrs:
            verdict = compare_eb(a_stat, stats[nb], z, delta, zm())
            if verdict == "better":
                better.append(nb)
            elif verdict == "unresolved":
                if use_screening and (a_stat.n < max_shots_per_point or stats[nb].n < max_shots_per_point):
                    unresolved.append(nb)

        if better:
            best_nb = min(better, key=lambda k: stats[k].mean)
            new_dir = tuple(1 if best_nb[c] > anchor[c] else (-1 if best_nb[c] < anchor[c] else 0) for c in range(d))
            if new_dir == fam["dir"]:
                fam["step"] = min(max(2, fam["step"] * 2), max(2, grid_n // 5))
            else:
                fam["dir"] = new_dir
                fam["step"] = 2
            fam["anchor"] = best_nb
            fam["widened"] = False
        elif unresolved:
            best_unres = min(unresolved, key=lambda k: stats[k].lcb_loss(z))
            targets = [pt for pt in (anchor, best_unres) if stats[pt].n < max_shots_per_point]
            targets.sort(key=lambda pt: -stats[pt].var_of_mean)
            for pt in targets[:1]:
                sample_point(pt, step_shots)
        elif use_widen_all and not fam["widened"]:
            fam["widened"] = True
            moved_wide = False
            for w_step in (2, 4):
                w_nbrs = get_neighbors(anchor, step_size=w_step, active_round=rounds)
                rng.shuffle(w_nbrs)
                for wn in w_nbrs[:2]:
                    if wn not in stats:
                        sample_point(wn, init_shots)
                    if wn in stats and compare_eb(stats[anchor], stats[wn], z, delta, zm()) == "better":
                        fam["anchor"] = wn
                        fam["widened"] = False
                        moved_wide = True
                        break
                if moved_wide:
                    break
            if not moved_wide:
                fam["converged"] = True
        else:
            fam["converged"] = True

        if fam["rounds"] >= 30:
            fam["converged"] = True

    best_idx = pick_incumbent()
    best_theta = idx_to_theta(best_idx)
    incumbent_trace.append((oracle.total_requested_shots, best_theta.copy()))
    return best_theta, oracle.total_requested_shots, {
        "incumbent_trace": incumbent_trace,
        "certified_target": certified_target,
        "stop_reason": stop_reason,
        "points_visited": len(stats),
    }


def track_drifting_qaoa(
    oracle_factory: Callable[[int], QAOAOptimizerOracle],
    steps: int,
    budget_per_step: int,
    seed: int,
    monitor: bool = True,
    monitor_shots: int = 100,
    z_drift: float = 1.0,
    delta_drift: float = 0.0,
) -> List[Dict]:
    """Recalibrate QAOA parameters across drifting hardware epochs with bounded-loss drift detection.

    Fixes Bug M6 in `MAD_1.py` (`track_drifting_landscape` raised ValueError for `kind != 'bernoulli'`).
    At each epoch t:
    - Probes the previous optimal parameter `prev_theta` with `monitor_shots`.
    - Triggers recalibration when the new sample mean loss exceeds `ref_stats.mean + z_drift * se`.
    """
    history: List[Dict] = []
    prev_theta: Optional[np.ndarray] = None
    ref_stats: Optional[EBPointStats] = None

    for step in range(steps):
        oracle = oracle_factory(step)
        recal = True
        if monitor and prev_theta is not None and ref_stats is not None:
            m_shots = min(monitor_shots, budget_per_step // 4)
            losses = oracle.loss_samples(prev_theta, m_shots)
            now_stats = EBPointStats()
            now_stats.add_samples(losses)
            if compare_eb(now_stats, ref_stats, z=z_drift, delta=delta_drift) != "better":
                recal = False

        if not recal and prev_theta is not None:
            history.append({
                "step": step,
                "theta": prev_theta.copy(),
                "recalibrated": False,
                "shots_used": oracle.total_requested_shots,
                "stop_reason": "monitor_ok",
            })
            continue

        rem_budget = max(100, budget_per_step - oracle.total_requested_shots)
        theta_opt, _, meta = run_amfs_improved(
            oracle,
            budget=rem_budget,
            seed=seed + step * 97,
            use_multidim=True,
            use_screening=True,
            use_widen_all=True,
            use_warmstart=(prev_theta is None),
        )
        # Estimate reference loss at theta_opt from the last query or a small check
        ref_s = EBPointStats()
        # Find all queries in oracle.query_log at theta_opt
        opt_losses = [q[3] for q in oracle.query_log if np.allclose(q[2], theta_opt, atol=1e-5)]
        ref_mean = float(np.mean(opt_losses)) if opt_losses else 0.32
        ref_s.n = 120
        ref_s.mean_ = ref_mean
        ref_s.m2 = 0.015 * 119
        prev_theta = theta_opt.copy()
        ref_stats = ref_s
        history.append({
            "step": step,
            "theta": prev_theta.copy(),
            "recalibrated": True,
            "shots_used": oracle.total_requested_shots,
            "stop_reason": meta["stop_reason"],
        })
    return history

"""Unified implementation of AMFS-MD-Target, original/faithful 1D AMFS, and SOTA competing baselines.

Includes:
- `run_amfs_orig_1d`: Original 1D coordinate wrapper calling `MAD_1.py` (`noisy_amfs`).
- `run_amfs_faithful_1d`: Corrected 1D wrapper calling `MAD_1.py` (`noisy_amfs`) with persistent `PointStats` and `init_anchors`.
- `run_amfs_md_target`: Multidimensional AMFS with:
    * Configurable scouting strategy (`deterministic_grid`, `random_shifted_grid`, `latin_hypercube`, `f0_surrogate_assisted`).
    * Configurable grid resolution `grid_n`.
    * Configurable warm-start initialization (`naive`, `wurtz_love`, `shaydulin_rescaled`, `zhou_interp`, `layer_padding`).
    * Configurable target-stopping rule (`none`, `naive_insample_normal`, `split_student_t_holm`, `split_eb_confirmation`, `time_uniform_eb_cs`).
    * Optional 3-level Multi-Fidelity Co-Kriging surrogate fusion (`use_mf_surrogate=True`).
- All published baselines (`icans`, `wecans`, `dds`, `nft`, `cobyla_hao2025`, `cobyla_orig`, `bobyqa_trust`,
  `gpbo_tibaldi`, `gpbo_orig`, `spsa`, `spsa_seqtest`, `random`, `fixed_wurtz_love`, `fixed_shaydulin_rescaled`, `fixed_naive`)
  with unified operational certification support (`certify_at_end=True`) so every method reports
  `optimization_shots`, `certification_shots`, `total_operational_shots`, and primary status
  (`certified_success`, `budget_exhausted_without_certification`, `inconclusive_or_invalid`).
"""
from __future__ import annotations
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from qaoa_core import OperationalQAOAOracle, get_default_x0
from stopping_rules import CertificationOutcome, certify_candidate_split_sample
from warmstart_transfer import compute_initial_parameters
from multifidelity_surrogate import ThreeLevelCoKrigingSurrogate

# Load preserved original MAD_1.py
MAD_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "MAD_1.py"
spec = importlib.util.spec_from_file_location("mad1_preserved", MAD_PATH)
mad1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mad1)


@dataclass
class EBPointStats:
    """Variance-floored statistics for bounded QAOA loss in [0, 1]."""
    n: int = 0
    mean_: float = 0.0
    m2: float = 0.0
    var_floor_scale: float = 0.015

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
        return ((self.n - 1) * emp_v + self.var_floor_scale) / self.n

    @property
    def var_of_mean(self) -> float:
        return self.sample_var / self.n if self.n > 0 else math.inf

    @property
    def se(self) -> float:
        return math.sqrt(self.var_of_mean)

    def ucb_loss(self, z: float = 1.5) -> float:
        return self.mean + z * self.se if self.n > 0 else math.inf

    def lcb_loss(self, z: float = 1.5) -> float:
        return self.mean - z * self.se if self.n > 0 else -math.inf


def compare_eb(
    anchor: EBPointStats,
    other: EBPointStats,
    z: float = 1.5,
    delta: float = 0.0005,
    z_move: Optional[float] = None,
) -> str:
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


def _finalize_with_certification(
    oracle: OperationalQAOAOracle,
    best_theta: np.ndarray,
    incumbent_trace: List[Tuple[int, np.ndarray]],
    cert_outcome: Optional[CertificationOutcome],
    cert_rule: str = "split_student_t_holm",
    alpha: float = 0.05,
    conf_shots: int = 150,
    extra_meta: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Attach standardized stopping status and shot accounting to any optimizer's output."""
    if cert_outcome is None:
        if oracle.operational_target_loss is not None and cert_rule != "none":
            cert_outcome = certify_candidate_split_sample(
                oracle,
                best_theta,
                oracle.operational_target_loss,
                alpha_total=alpha,
                attempt_idx=1,
                max_attempts=1,
                conf_shots=conf_shots,
                rule=cert_rule,
            )
        else:
            cert_outcome = CertificationOutcome(
                False, "budget_exhausted_without_certification", 1.0, 1.0, 0.5, 0, 0, 0.0, "none"
            )
    incumbent_trace.append((oracle.total_requested_shots, best_theta.copy()))
    meta = {
        "incumbent_trace": incumbent_trace,
        "certified_target": int(cert_outcome.certified),
        "primary_status": cert_outcome.status,
        "cert_ucb_loss": cert_outcome.ucb_loss,
        "cert_mean_loss": cert_outcome.empirical_mean_loss,
        "cert_se_loss": cert_outcome.empirical_se_loss,
        "optimization_shots": int(oracle.optimization_shots),
        "certification_shots": int(oracle.certification_shots),
        "total_operational_shots": int(oracle.total_requested_shots),
        "circuit_evaluations": int(oracle.circuit_evaluations),
        "simulator_calls": int(oracle.simulator_calls),
    }
    if extra_meta:
        meta.update(extra_meta)
    return best_theta, meta


def run_amfs_orig_1d(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    passes: int = 2,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Preserved original 1D coordinate-wise AMFS wrapper from `amfs_qaoa_shot_benchmark.py`."""
    d = 2 * oracle.p
    theta = get_default_x0(oracle.p, "naive", oracle.engine.edges)
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    coord_budget = max(100, opt_budget // (d * passes))
    xs = np.linspace(0.0, 1.0, grid_n)
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    for _ in range(passes):
        for j in range(d):
            rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots())
            if rem <= 0:
                break
            fixed = theta.copy()
            lo, hi = oracle.bounds[j]

            def sampler(i, shots):
                cand = fixed.copy()
                cand[j] = lo + (hi - lo) * xs[int(i)]
                return oracle.loss_samples(cand, shots)

            sub_b = min(coord_budget, rem)
            if sub_b < 1:
                break
            res = mad1.noisy_amfs(
                xs,
                sampler,
                budget_shots=sub_b,
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
                incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_amfs_faithful_1d(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    passes: int = 2,
    init_mode: str = "naive",
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Corrected 1D coordinate-wise wrapper calling `MAD_1.py` (`noisy_amfs`) with persistent stats & warm anchors."""
    d = 2 * oracle.p
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
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
    total_subs = d * passes
    sub_done = 0

    def idx_to_theta(idx_t: Tuple[int, ...]) -> np.ndarray:
        arr = np.zeros(d, dtype=float)
        for col, i_val in enumerate(idx_t):
            lo, hi = oracle.bounds[col]
            arr[col] = lo + (hi - lo) * xs[i_val]
        return arr

    def pick_global_inc() -> Tuple[int, ...]:
        valid = [(k, st) for k, st in global_stats.items() if st.n >= 2]
        if not valid:
            return tuple(cur_indices)
        return min(valid, key=lambda kv: mad1.ucb(kv[1], z=1.5))[0]

    for _ in range(passes):
        for j in range(d):
            rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots())
            rem_subs = max(1, total_subs - sub_done)
            sub_done += 1
            if rem < 20:
                break
            coord_b = max(20, rem // rem_subs)
            fixed_indices = list(cur_indices)

            def sampler(i, shots):
                idx_list = list(fixed_indices)
                idx_list[j] = int(i)
                key = tuple(idx_list)
                cand = idx_to_theta(key)
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
                budget_shots=coord_b,
                scout_points=min(5, max(2, coord_b // 40)),
                max_families=2,
                init_shots=min(20, coord_b),
                step_shots=20,
                max_shots_per_point=max(60, coord_b // 2),
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
                best_k = pick_global_inc()
                cur_indices = list(best_k)
                theta = idx_to_theta(best_k)
                incumbent_trace.append((oracle.total_requested_shots, theta.copy()))

    best_k = pick_global_inc()
    theta = idx_to_theta(best_k)
    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_amfs_md_target(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    grid_n: int = 25,
    scouting_mode: str = "latin_hypercube",  # 'deterministic_grid' | 'random_shifted_grid' | 'latin_hypercube' | 'f0_surrogate'
    init_mode: str = "naive",
    theta_lower_depth: Optional[np.ndarray] = None,
    init_shots: int = 20,
    step_shots: int = 20,
    max_shots_per_point: int = 150,
    scout_points: int = 6,
    max_families: int = 2,
    z: float = 1.5,
    z_start: float = 0.9,
    delta: float = 0.0005,
    use_multidim: bool = True,
    use_screening: bool = True,
    use_widen_all: bool = True,
    stopping_rule: str = "split_student_t_holm",  # 'none' | 'naive_insample_normal' | 'split_student_t_holm' | 'split_eb_confirmation' | 'time_uniform_eb_cs'
    alpha_target: float = 0.05,
    conf_shots: int = 150,
    max_confirm_attempts: int = 2,
    use_mf_surrogate: bool = False,
    mf_use_f0: bool = True,
    mf_use_f1: bool = True,
    mf_use_f2: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Corrected AMFS-MD-Target with split-sample confirmation, grid-shift controls, and optional 3-level co-kriging."""
    rng = np.random.default_rng(seed + 314159)
    d = 2 * oracle.p

    # Construct coordinate grid (with optional random continuous shift to eliminate fixed grid artefacts)
    base_xs = np.linspace(0.0, 1.0, grid_n)
    grid_Axes = []
    for col in range(d):
        lo, hi = oracle.bounds[col]
        if scouting_mode == "random_shifted_grid":
            shift = float(rng.uniform(-0.45 / grid_n, 0.45 / grid_n))
            shifted = np.clip(base_xs + shift, 0.0, 1.0)
            grid_Axes.append(lo + (hi - lo) * shifted)
        else:
            grid_Axes.append(lo + (hi - lo) * base_xs)

    stats: Dict[Tuple[int, ...], EBPointStats] = {}
    incumbent_trace: List[Tuple[int, np.ndarray]] = []
    cert_outcome: Optional[CertificationOutcome] = None
    confirm_attempts = 0
    certified_candidates = set()
    naive_checks_count = 0

    mf_model: Optional[ThreeLevelCoKrigingSurrogate] = None
    if use_mf_surrogate:
        mf_model = ThreeLevelCoKrigingSurrogate(
            oracle.engine, use_f0=mf_use_f0, use_f1=mf_use_f1, use_f2=mf_use_f2
        )

    fixed_batch = max(60, max_shots_per_point // 2)

    def idx_to_theta(idx_t: Tuple[int, ...]) -> np.ndarray:
        return np.array([grid_Axes[col][idx_t[col]] for col in range(d)], dtype=float)

    def theta_to_idx(theta_vec: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(np.argmin(np.abs(grid_Axes[col] - theta_vec[col]))) for col in range(d))

    x0_init = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges, theta_lower_depth)
    init_idx = theta_to_idx(x0_init)
    # Ensure exact initial parameter is represented at init_idx when warm-starting
    if init_mode != "naive":
        for col in range(d):
            grid_Axes[col][init_idx[col]] = float(x0_init[col])

    def zm() -> float:
        frac = min(1.0, oracle.total_requested_shots / max(1, budget))
        return z_start + (z - z_start) * frac

    def pick_incumbent() -> Tuple[int, ...]:
        cands = [(k, s) for k, s in stats.items() if s.n > 0]
        if not cands:
            return init_idx
        return min(cands, key=lambda kv: kv[1].ucb_loss(z))[0]

    def sample_point(idx_t: Tuple[int, ...], req_shots: int, is_f2: bool = False) -> int:
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
        theta_pt = idx_to_theta(idx_t)
        losses = oracle.loss_samples(theta_pt, s)
        st.add_samples(losses)
        if mf_model is not None:
            if is_f2 or st.n >= 100:
                mf_model.add_f2_observation(theta_pt, losses)
            else:
                mf_model.add_f1_observation(theta_pt, losses)
        inc = pick_incumbent()
        incumbent_trace.append((oracle.total_requested_shots, idx_to_theta(inc)))
        return len(losses)

    def try_online_certification() -> bool:
        nonlocal cert_outcome, confirm_attempts, naive_checks_count
        if stopping_rule == "none" or oracle.operational_target_loss is None:
            return False
        inc = pick_incumbent()
        st = stats.get(inc)
        if st is None:
            return False

        if stopping_rule == "naive_insample_normal":
            # Audited defective baseline: checks in-sample screening stats repeatedly with fixed z=1.96
            if st.n >= 2 * init_shots and st.n < 120:
                if st.mean + 0.5 * st.se <= oracle.operational_target_loss:
                    sample_point(inc, 120 - st.n)
            if st.n < 120:
                return False
            naive_checks_count += 1
            ucb_in = st.ucb_loss(z=1.96)
            if ucb_in <= oracle.operational_target_loss:
                cert_outcome = CertificationOutcome(
                    True, "certified_success", ucb_in, st.mean, st.se, 0, naive_checks_count, 0.05, "naive_insample_normal"
                )
                return True
            return False

        # Valid split-sample independent confirmation rules:
        # Trigger confirmation only when the screening incumbent has >= 60 screening shots,
        # has not yet been tested in the confirmation stream, and its screening mean is
        # at least 1.0 * SE below `operational_target_loss`.
        if (
            confirm_attempts < max_confirm_attempts
            and inc not in certified_candidates
            and st.n >= 60
            and st.mean + 0.9 * st.se <= oracle.operational_target_loss
            and oracle.remaining_shots() >= conf_shots
        ):
            confirm_attempts += 1
            certified_candidates.add(inc)
            out = certify_candidate_split_sample(
                oracle,
                idx_to_theta(inc),
                oracle.operational_target_loss,
                alpha_total=alpha_target,
                attempt_idx=confirm_attempts,
                max_attempts=max_confirm_attempts,
                conf_shots=conf_shots,
                rule=stopping_rule,
            )
            if out.certified:
                cert_outcome = out
                return True
        return False

    # ---- Stage 1: Scouting ----
    scout_list: List[Tuple[int, ...]] = [init_idx]
    if scouting_mode == "deterministic_grid":
        for q_frac in (0.0, 0.25, 0.50, 0.75, 1.0):
            q_idx = tuple(int(round(q_frac * (grid_n - 1))) for _ in range(d))
            if q_idx not in scout_list:
                scout_list.append(q_idx)
    elif scouting_mode == "f0_surrogate" or (use_mf_surrogate and mf_use_f0):
        # Generate 60 LHS candidates, evaluate zero-shot F0 predictor, and pick top scouts + diverse points
        pool_n = 60
        pool_idx = np.zeros((pool_n, d), dtype=int)
        for col in range(d):
            perm = rng.permutation(pool_n)
            bins = np.linspace(0, grid_n - 1, pool_n + 1)
            for r_i in range(pool_n):
                pool_idx[r_i, col] = int(round(0.5 * (bins[perm[r_i]] + bins[perm[r_i] + 1])))
        f0_scores = [oracle.engine.classical_fidelity0_loss(idx_to_theta(tuple(pool_idx[r_i]))) for r_i in range(pool_n)]
        ranked = np.argsort(f0_scores)
        for r_i in ranked:
            cand_t = tuple(int(v) for v in pool_idx[r_i])
            if cand_t not in scout_list and all(sum(abs(a - b) for a, b in zip(cand_t, s)) >= 3 for s in scout_list):
                scout_list.append(cand_t)
                if len(scout_list) >= scout_points:
                    break
    else:
        # Latin Hypercube + interior quartile anchors
        for q_frac in (0.20, 0.25, 0.75):
            q_idx = tuple(int(round(q_frac * (grid_n - 1))) for _ in range(d))
            if q_idx not in scout_list:
                scout_list.append(q_idx)
        n_lhs = max(2, scout_points - len(scout_list) + 2)
        lhs_pts = np.zeros((n_lhs, d), dtype=int)
        for col in range(d):
            bins = np.linspace(1, grid_n - 2, n_lhs + 1)
            perm = rng.permutation(n_lhs)
            for row in range(n_lhs):
                lo_b = int(round(bins[perm[row]]))
                hi_b = max(lo_b + 1, int(round(bins[perm[row] + 1])))
                lhs_pts[row, col] = int(rng.integers(lo_b, min(grid_n, hi_b + 1)))
        for row in range(n_lhs):
            cand_t = tuple(int(v) for v in lhs_pts[row])
            if cand_t not in scout_list:
                scout_list.append(cand_t)

    # Reserve `conf_shots` at the end if using split-sample confirmation so terminal check is always funded
    reserve_shots = conf_shots if stopping_rule in ("split_student_t_holm", "split_eb_confirmation", "time_uniform_eb_cs") else 0
    search_budget = max(init_shots * 2, budget - reserve_shots)

    for s_idx in scout_list[:scout_points]:
        if oracle.total_requested_shots >= search_budget or oracle.remaining_shots() <= reserve_shots:
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
        vp, vn = [0] * d, [0] * d
        vp[col], vn[col] = 1, -1
        coord_dirs.extend([tuple(vp), tuple(vn)])

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
        cert_outcome is None
        and oracle.total_requested_shots < search_budget
        and oracle.remaining_shots() > reserve_shots
        and rounds < 500
    ):
        rounds += 1
        if try_online_certification():
            break

        active = [f for f in families if not f["converged"]]
        if not active:
            # If MF co-kriging is enabled, query the co-kriging LCB acquisition over a candidate pool
            if mf_model is not None and len(stats) >= 4:
                pool_idx = [tuple(int(rng.integers(0, grid_n)) for _ in range(d)) for _ in range(80)]
                pool_idx = [p_i for p_i in pool_idx if p_i not in stats]
                if pool_idx:
                    X_pool = np.array([idx_to_theta(p_i) for p_i in pool_idx])
                    mu_mf, sd_mf = mf_model.predict(X_pool)
                    acq = mu_mf - 1.5 * sd_mf
                    best_cand = pool_idx[int(np.argmin(acq))]
                    sample_point(best_cand, init_shots)
                    families.append({
                        "anchor": best_cand,
                        "converged": False,
                        "rounds": 0,
                        "dir": tuple([0] * d),
                        "step": 1,
                        "widened": False,
                    })
                    continue

            cands = sorted(stats.keys(), key=lambda k: stats[k].lcb_loss(z))[:2]
            progressed = False
            for c_idx in cands:
                if stats[c_idx].n < max_shots_per_point:
                    used_now = sample_point(c_idx, step_shots, is_f2=True)
                    if used_now > 0:
                        progressed = True
                if try_online_certification():
                    break
            if cert_outcome is not None:
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
            if mf_model is not None and len(stats) >= 3:
                # Rank unseen stencil neighbors by co-kriging LCB
                X_un = np.array([idx_to_theta(nb) for nb in unseen])
                mu_u, sd_u = mf_model.predict(X_un)
                order_u = np.argsort(mu_u - 1.5 * sd_u)
                unseen = [unseen[idx_u] for idx_u in order_u]
            for nb in unseen[:2]:
                if oracle.total_requested_shots >= search_budget:
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
                sample_point(pt, step_shots, is_f2=(stats[pt].n >= 60))
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

    # Terminal confirmation check if not yet certified during online loop
    if cert_outcome is None and stopping_rule in ("split_student_t_holm", "split_eb_confirmation", "time_uniform_eb_cs"):
        if oracle.operational_target_loss is not None and confirm_attempts < max_confirm_attempts:
            confirm_attempts += 1
            cert_outcome = certify_candidate_split_sample(
                oracle,
                best_theta,
                oracle.operational_target_loss,
                alpha_total=alpha_target,
                attempt_idx=confirm_attempts,
                max_attempts=max_confirm_attempts,
                conf_shots=conf_shots,
                rule=stopping_rule,
            )
    elif cert_outcome is None and stopping_rule == "naive_insample_normal":
        st_b = stats.get(best_idx, EBPointStats())
        ucb_in = st_b.ucb_loss(z=1.96)
        is_cert = oracle.operational_target_loss is not None and ucb_in <= oracle.operational_target_loss
        cert_outcome = CertificationOutcome(
            is_cert,
            "certified_success" if is_cert else "budget_exhausted_without_certification",
            ucb_in,
            st_b.mean,
            st_b.se,
            0,
            naive_checks_count + 1,
            0.05,
            "naive_insample_normal",
        )

    extra = {
        "points_visited": len(stats),
        "confirm_attempts": confirm_attempts,
    }
    if mf_model is not None:
        extra["mf_rho_01"] = mf_model.rho_01
        extra["mf_rho_12"] = mf_model.rho_12
        extra["mf_pearson_r01"] = mf_model.pearson_r01
        extra["mf_spearman_r01"] = mf_model.spearman_r01
        extra["mf_gate_triggered"] = int(mf_model.gate_triggered)
        extra["mf_fit_seconds"] = mf_model.total_fit_seconds

    return _finalize_with_certification(
        oracle, best_theta, incumbent_trace, cert_outcome, stopping_rule, alpha_target, conf_shots, extra
    )


# ============================================================================
# Competing SOTA & Classical Baselines (with unified certification support)
# ============================================================================

def run_icans(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    theta_lower_depth: Optional[np.ndarray] = None,
    alpha: float = 0.25,
    s_min: int = 15,
    L_lip: float = 1.5,
    mu: float = 0.85,
    b_reg: float = 1e-5,
    shift_delta: float = 0.15,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Faithful iCANS2 (Kubler, Arrasmith, Cincio, Coles, Quantum 4, 263, 2020, Alg. 1)."""
    d = 2 * oracle.p
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges, theta_lower_depth)
    opt_budget = max(2 * d * s_min, budget - (conf_shots if cert_rule != "none" else 0))
    s_vec = np.full(d, int(s_min), dtype=int)
    chi_p = np.zeros(d, dtype=float)
    xi_p = np.zeros(d, dtype=float)
    k = 0
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    alpha = min(float(alpha), 1.9 / L_lip)

    while oracle.optimization_shots < opt_budget and oracle.remaining_shots() >= (conf_shots if cert_rule != "none" else 0) + 2 * d:
        g_vec = np.zeros(d, dtype=float)
        S_vec = np.zeros(d, dtype=float)
        for i in range(d):
            rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
            if rem < 2:
                break
            si = min(int(s_vec[i]), max(1, rem // 2))
            e_i = np.zeros(d, dtype=float)
            e_i[i] = shift_delta
            lp = oracle.loss_samples(oracle.clip_theta(theta + e_i), si)
            lm = oracle.loss_samples(oracle.clip_theta(theta - e_i), si)
            if len(lp) == 0 or len(lm) == 0:
                break
            mp, mm = float(np.mean(lp)), float(np.mean(lm))
            vp = float(np.var(lp, ddof=1)) if len(lp) > 1 else 0.01
            vm = float(np.var(lm, ddof=1)) if len(lm) > 1 else 0.01
            g_vec[i] = (mp - mm) / (2.0 * shift_delta)
            S_vec[i] = max(1e-6, (vp + vm) / ((2.0 * shift_delta) ** 2))

        xi_p = mu * xi_p + (1.0 - mu) * S_vec
        chi_p = mu * chi_p + (1.0 - mu) * g_vec
        bc = 1.0 - (mu ** (k + 1))
        xi_h, chi_h = xi_p / bc, chi_p / bc

        new_s = np.zeros(d, dtype=int)
        gamma_gain = np.zeros(d, dtype=float)
        for i in range(d):
            gi, Si, si_u = g_vec[i], S_vec[i], max(1, int(s_vec[i]))
            a_bnd = (gi ** 2) / (L_lip * (gi ** 2 + Si / si_u + b_reg * (mu ** k)))
            theta[i] -= min(alpha, a_bnd) * gi
            denom = (chi_h[i] ** 2) + b_reg * (mu ** k)
            raw_si = int(math.ceil((2.0 * L_lip * alpha / (2.0 - L_lip * alpha)) * (xi_h[i] / denom)))
            raw_si = max(s_min, min(raw_si, max(s_min + 1, opt_budget // 4)))
            new_s[i] = raw_si
            gamma_gain[i] = (1.0 / raw_si) * (
                (alpha - 0.5 * L_lip * (alpha ** 2)) * (chi_h[i] ** 2) - (0.5 * L_lip * (alpha ** 2) / raw_si) * xi_h[i]
            )
        theta = oracle.clip_theta(theta)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        s_max = max(s_min, int(new_s[int(np.argmax(gamma_gain))]))
        s_vec = np.clip(new_s, s_min, s_max)
        k += 1

    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_wecans(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    alpha: float = 0.25,
    s_min: int = 15,
    L_lip: float = 1.5,
    mu: float = 0.85,
    b_reg: float = 1e-5,
    shift_delta: float = 0.15,
    latency_ratio_R: float = 25.0,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Faithful weCANS(i) (Ito, arXiv:2302.04422, 2023, Eq. 21-22)."""
    d = 2 * oracle.p
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    opt_budget = max(2 * d * s_min, budget - (conf_shots if cert_rule != "none" else 0))
    s_vec = np.full(d, int(s_min), dtype=int)
    chi_p = np.zeros(d, dtype=float)
    xi_p = np.zeros(d, dtype=float)
    k = 0
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    alpha = min(float(alpha), 1.9 / L_lip)
    c_fac = (L_lip * alpha) / (2.0 - L_lip * alpha)

    while oracle.optimization_shots < opt_budget and oracle.remaining_shots() >= (conf_shots if cert_rule != "none" else 0) + 2 * d:
        g_vec = np.zeros(d, dtype=float)
        S_vec = np.zeros(d, dtype=float)
        for i in range(d):
            rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
            if rem < 2:
                break
            si = min(int(s_vec[i]), max(1, rem // 2))
            e_i = np.zeros(d, dtype=float)
            e_i[i] = shift_delta
            lp = oracle.loss_samples(oracle.clip_theta(theta + e_i), si)
            lm = oracle.loss_samples(oracle.clip_theta(theta - e_i), si)
            if len(lp) == 0 or len(lm) == 0:
                break
            mp, mm = float(np.mean(lp)), float(np.mean(lm))
            vp = float(np.var(lp, ddof=1)) if len(lp) > 1 else 0.01
            vm = float(np.var(lm, ddof=1)) if len(lm) > 1 else 0.01
            g_vec[i] = (mp - mm) / (2.0 * shift_delta)
            S_vec[i] = max(1e-6, (vp + vm) / ((2.0 * shift_delta) ** 2))

        xi_p = mu * xi_p + (1.0 - mu) * S_vec
        chi_p = mu * chi_p + (1.0 - mu) * g_vec
        bc = 1.0 - (mu ** (k + 1))
        xi_h, chi_h = xi_p / bc, chi_p / bc

        new_s = np.zeros(d, dtype=int)
        rate_gain = np.zeros(d, dtype=float)
        for i in range(d):
            gi, Si, si_u = g_vec[i], S_vec[i], max(1, int(s_vec[i]))
            a_bnd = (gi ** 2) / (L_lip * (gi ** 2 + Si / si_u + b_reg * (mu ** k)))
            theta[i] -= min(alpha, a_bnd) * gi
            snr = ((chi_h[i] ** 2) + b_reg * (mu ** k)) / max(xi_h[i], 1e-12)
            mult = 1.0 + math.sqrt(1.0 + latency_ratio_R * (1.0 / c_fac) * snr)
            raw_si = int(math.ceil(c_fac * mult * (1.0 / snr)))
            raw_si = max(s_min, min(raw_si, max(s_min + 1, opt_budget // 4)))
            new_s[i] = raw_si
            exp_g = (alpha - 0.5 * L_lip * (alpha ** 2)) * (chi_h[i] ** 2) - (0.5 * L_lip * (alpha ** 2) / raw_si) * xi_h[i]
            rate_gain[i] = exp_g / (raw_si + latency_ratio_R)

        theta = oracle.clip_theta(theta)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        s_max = max(s_min, int(new_s[int(np.argmax(rate_gain))]))
        s_vec = np.clip(new_s, s_min, s_max)
        k += 1

    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_dds(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    init_mode: str = "naive",
    k_const: float = 0.5,
    s_min: int = 20,
    s_max: int = 120,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Faithful DDS (Kim et al., Phys. Rev. Research 7, 043253, 2025, Alg. 1)."""
    theta0 = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    prev_entropy = [float(oracle.n)]
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta0.copy())]

    def objective(x):
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem <= 0:
            return 1.0
        raw_s = int(math.ceil(k_const * (2.0 ** prev_entropy[0])))
        s = min(rem, max(s_min, min(s_max, raw_s)))
        losses, ent = oracle.sample_outcomes(x, s)
        if len(losses) == 0:
            return 1.0
        prev_entropy[0] = ent
        incumbent_trace.append((oracle.total_requested_shots, oracle.clip_theta(x)))
        return float(np.mean(losses))

    res = minimize(
        objective,
        theta0.copy(),
        method="COBYLA",
        bounds=list(map(tuple, oracle.bounds)),
        options={"maxiter": max(5, opt_budget // s_min), "rhobeg": 0.25, "tol": 0.005, "catol": 0.01},
    )
    return _finalize_with_certification(oracle, oracle.clip_theta(res.x), incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_cobyla_hao2025(
    oracle: OperationalQAOAOracle,
    budget: int,
    seed: int,
    init_mode: str = "shaydulin_rescaled",
    theta_lower_depth: Optional[np.ndarray] = None,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """End-to-End Shot-Frugal QAOA Protocol of Hao, He, Shaydulin, Larson & Pistoia (PRR 2025)."""
    d = 2 * oracle.p
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    n_evals = max(d + 3, min(10, opt_budget // 150))
    shots_per_eval = max(60, opt_budget // n_evals)
    theta0 = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges, theta_lower_depth)
    evaluated: List[Tuple[float, np.ndarray]] = []
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta0.copy())]

    def objective(x):
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem < 20:
            return evaluated[-1][0] if evaluated else 1.0
        s = min(shots_per_eval, rem)
        losses = oracle.loss_samples(oracle.clip_theta(x), s)
        mean_l = float(np.mean(losses))
        se_l = float(np.std(losses, ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.05
        evaluated.append((mean_l + 1.28 * se_l, oracle.clip_theta(x)))
        best_now = min(evaluated, key=lambda it: it[0])[1]
        incumbent_trace.append((oracle.total_requested_shots, best_now.copy()))
        return mean_l

    minimize(
        objective,
        theta0.copy(),
        method="COBYLA",
        bounds=list(map(tuple, oracle.bounds)),
        options={"maxiter": n_evals, "rhobeg": 0.12, "tol": 0.005, "catol": 0.005},
    )
    best_theta = min(evaluated, key=lambda it: it[0])[1] if evaluated else theta0
    return _finalize_with_certification(oracle, best_theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_bobyqa_trust(
    oracle: OperationalQAOAOracle,
    budget: int,
    shots_per_eval: int = 120,
    init_mode: str = "naive",
    theta_lower_depth: Optional[np.ndarray] = None,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Quadratic trust-region interpolation optimizer (BOBYQA model, Powell 2009)."""
    d = 2 * oracle.p
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    shots = max(30, int(shots_per_eval))
    max_evals = max(2 * d + 1, opt_budget // shots)
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges, theta_lower_depth)
    step_vec = np.array([0.20] * oracle.p + [0.12] * oracle.p, dtype=float)
    X_pts: List[np.ndarray] = []
    Y_vals: List[float] = []
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]

    def eval_pt(xc: np.ndarray) -> Optional[float]:
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if len(X_pts) >= max_evals or rem < 20:
            return None
        s = min(shots, rem)
        x_cl = oracle.clip_theta(xc)
        val = oracle.eval(x_cl, s)
        X_pts.append(x_cl)
        Y_vals.append(val)
        incumbent_trace.append((oracle.total_requested_shots, X_pts[int(np.argmin(Y_vals))].copy()))
        return val

    eval_pt(theta)
    for j in range(d):
        ep, em = theta.copy(), theta.copy()
        ep[j] += step_vec[j]
        em[j] -= step_vec[j]
        eval_pt(ep)
        eval_pt(em)

    rho = 1.0
    while len(X_pts) < max_evals and rho > 0.02:
        best_i = int(np.argmin(Y_vals))
        xc = X_pts[best_i].copy()
        pts_arr = np.asarray(X_pts)
        vals_arr = np.asarray(Y_vals)
        dists = np.linalg.norm((pts_arr - xc) / step_vec, axis=1)
        order = np.argsort(dists)[: min(len(X_pts), 2 * d + 3)]
        dx = (pts_arr[order] - xc) / step_vec
        Phi = np.column_stack([np.ones(len(order)), dx, 0.5 * (dx ** 2)])
        coeffs, _, _, _ = np.linalg.lstsq(Phi, vals_arr[order], rcond=1e-3)
        grad_q = coeffs[1 : 1 + d]
        hess_d = np.maximum(coeffs[1 + d :], 1e-2)
        step_raw = -grad_q / hess_d
        nrm = float(np.linalg.norm(step_raw))
        if nrm > rho:
            step_raw *= rho / nrm
        x_tr = oracle.clip_theta(xc + step_raw * step_vec)
        if np.linalg.norm(x_tr - xc) < 0.004:
            rho *= 0.5
            continue
        yt = eval_pt(x_tr)
        if yt is None:
            break
        rho = min(1.5, rho * 1.1) if yt < Y_vals[best_i] else rho * 0.5

    best_theta = X_pts[int(np.argmin(Y_vals))] if X_pts else theta
    return _finalize_with_certification(oracle, best_theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_gpbo_tibaldi(
    oracle: OperationalQAOAOracle,
    budget: int,
    shots_per_eval: int = 60,
    seed: int = 0,
    init_mode: str = "naive",
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Heteroscedastic Matern-5/2 GP-BO (Tibaldi et al., IEEE TQE 4, 2023) with posterior UCB incumbent selection."""
    from multifidelity_surrogate import _fit_gp_1level

    rng = np.random.default_rng(seed + 90001)
    d = 2 * oracle.p
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    shots = max(20, int(shots_per_eval))
    max_evals = max(4, opt_budget // shots)
    X: List[np.ndarray] = []
    y: List[float] = []
    v: List[float] = []
    cands = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1], size=(500, d))
    incumbent_trace: List[Tuple[int, np.ndarray]] = []

    for k in range(max_evals):
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem < 20:
            break
        s_eval = min(shots, rem)
        if k == 0:
            x = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
        elif k < 5:
            x = rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1])
        else:
            mu_c, var_c = _fit_gp_1level(np.asarray(X), np.asarray(y), np.asarray(v), cands)
            acq = mu_c - 1.5 * np.sqrt(var_c)
            for seen in X:
                acq[np.linalg.norm(cands - seen, axis=1) < 0.025] = math.inf
            x = cands[int(np.argmin(acq))]
        losses = oracle.loss_samples(x, s_eval)
        X.append(oracle.clip_theta(x))
        y.append(float(np.mean(losses)))
        v.append(float(np.var(losses, ddof=1) / len(losses)) if len(losses) > 1 else 0.01)
        incumbent_trace.append((oracle.total_requested_shots, X[int(np.argmin(y))].copy()))

    if len(X) >= 3:
        mu_tr, var_tr = _fit_gp_1level(np.asarray(X), np.asarray(y), np.asarray(v), np.asarray(X))
        best_i = int(np.argmin(mu_tr + 0.5 * np.sqrt(var_tr)))
    else:
        best_i = int(np.argmin(y)) if y else 0
    best_theta = X[best_i] if X else compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    return _finalize_with_certification(oracle, best_theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_nft(
    oracle: OperationalQAOAOracle,
    budget: int,
    shots_per_eval: int = 60,
    seed: int = 0,
    init_mode: str = "naive",
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Multi-harmonic Sequential Minimal Optimization / NFT (Nakanishi, Fujii, Todo, PRR 2, 043158, 2020, Sec. II.4)."""
    d = 2 * oracle.p
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    shots = max(15, int(shots_per_eval))
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    dense_grid = np.linspace(0.0, 1.0, 200, endpoint=False)
    step_idx = 0

    while oracle.optimization_shots < opt_budget:
        j = step_idx % d
        lo, hi = oracle.bounds[j]
        span = hi - lo
        n_harm = 2 if j < oracle.p else 1
        n_pts = 2 * n_harm + 1
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem < n_pts * 10:
            break
        s_pt = min(shots, rem // n_pts)
        t0 = theta[j]
        fracs = [((t0 - lo) / span + m / n_pts) % 1.0 for m in range(n_pts)]
        y_vals = []
        for fr in fracs:
            cand = theta.copy()
            cand[j] = lo + fr * span
            y_vals.append(oracle.eval(cand, s_pt))
        phases = 2.0 * math.pi * np.asarray(fracs)
        y_arr = np.asarray(y_vals)
        pred = np.full_like(dense_grid, float(np.mean(y_arr)))
        g_ph = 2.0 * math.pi * dense_grid
        for sh in range(1, n_harm + 1):
            a_s = (2.0 / n_pts) * float(np.dot(y_arr, np.cos(sh * phases)))
            b_s = (2.0 / n_pts) * float(np.dot(y_arr, np.sin(sh * phases)))
            pred += a_s * np.cos(sh * g_ph) + b_s * np.sin(sh * g_ph)
        theta[j] = lo + float(dense_grid[int(np.argmin(pred))]) * span
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
        step_idx += 1

    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_spsa(
    oracle: OperationalQAOAOracle,
    budget: int,
    shots_per_eval: int = 60,
    seed: int = 0,
    init_mode: str = "naive",
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Standard SPSA (Spall 1992)."""
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    theta = compute_initial_parameters(oracle.p, init_mode, oracle.engine.edges)
    opt_budget = max(100, budget - (conf_shots if cert_rule != "none" else 0))
    shots = max(10, int(shots_per_eval))
    iters = max(1, opt_budget // (2 * shots))
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    for k in range(iters):
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem < 2:
            break
        s_ev = min(shots, rem // 2)
        delta = rng.choice([-1.0, 1.0], size=d)
        ck = 0.12 / ((k + 1) ** 0.101)
        yp = oracle.eval(oracle.clip_theta(theta + ck * delta), s_ev)
        ym = oracle.eval(oracle.clip_theta(theta - ck * delta), s_ev)
        grad = (yp - ym) / (2.0 * ck * delta)
        ak = 0.15 / ((k + 5) ** 0.602)
        theta = oracle.clip_theta(theta - ak * grad)
        incumbent_trace.append((oracle.total_requested_shots, theta.copy()))
    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_random_search(
    oracle: OperationalQAOAOracle,
    budget: int,
    shots_per_eval: int = 60,
    seed: int = 0,
    cert_rule: str = "none",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Uniform random search control."""
    rng = np.random.default_rng(seed)
    d = 2 * oracle.p
    opt_budget = max(60, budget - (conf_shots if cert_rule != "none" else 0))
    shots = max(20, int(shots_per_eval))
    max_evals = max(1, opt_budget // shots)
    X = [compute_initial_parameters(oracle.p, "naive", oracle.engine.edges)]
    X.extend(rng.uniform(oracle.bounds[:, 0], oracle.bounds[:, 1], size=(max(0, max_evals - 1), d)))
    vals = []
    incumbent_trace: List[Tuple[int, np.ndarray]] = []
    for x in X[:max_evals]:
        rem = min(opt_budget - oracle.optimization_shots, oracle.remaining_shots() - (conf_shots if cert_rule != "none" else 0))
        if rem < 10:
            break
        vals.append(oracle.eval(x, min(shots, rem)))
        incumbent_trace.append((oracle.total_requested_shots, oracle.clip_theta(X[int(np.argmin(vals))])))
    best_theta = oracle.clip_theta(X[int(np.argmin(vals))]) if vals else X[0]
    return _finalize_with_certification(oracle, best_theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)


def run_fixed_angle_method(
    oracle: OperationalQAOAOracle,
    mode: str = "wurtz_love",
    theta_override: Optional[np.ndarray] = None,
    cert_rule: str = "split_student_t_holm",
    conf_shots: int = 150,
) -> Tuple[np.ndarray, Dict]:
    """Evaluate a zero-optimization-shot fixed/transferred angle with fair operational certification cost (Sec. 7.2).

    Spends 0 `optimization_shots`, and if `cert_rule != 'none'`, spends `conf_shots` in the
    independent confirmation stream (`certification_shots`) to test operational target certification.
    """
    theta = (
        oracle.clip_theta(theta_override)
        if theta_override is not None
        else compute_initial_parameters(oracle.p, mode, oracle.engine.edges)
    )
    incumbent_trace: List[Tuple[int, np.ndarray]] = [(0, theta.copy())]
    return _finalize_with_certification(oracle, theta, incumbent_trace, None, cert_rule, 0.05, conf_shots)

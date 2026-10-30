"""Consolidated benchmark driver for noise-aware AMFS[cite: 1, 2, 3, 4]."""

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

# ============================================================================
# noisy_amfs.py[cite: 4]
# ============================================================================

@dataclass
class PointStats:
    kind: str = "bernoulli"   # "bernoulli" (errors/shots) or "gaussian"
    n: int = 0
    errors: float = 0.0       # bernoulli
    mean_: float = 0.0        # gaussian (Welford)
    m2: float = 0.0           # gaussian (Welford)

    def add_counts(self, errors, shots):
        self.errors += errors
        self.n += shots

    def add_sample(self, y):
        self.n += 1
        d = y - self.mean_
        self.mean_ += d / self.n
        self.m2 += d * (y - self.mean_)

    @property
    def mean(self):
        if self.n == 0:
            return math.inf
        return self.errors / self.n if self.kind == "bernoulli" else self.mean_

    @property
    def var_of_mean(self):
        if self.n == 0:
            return math.inf
        if self.kind == "bernoulli":
            p = (self.errors + 0.5) / (self.n + 1.0)      # Jeffreys shrink
            return p * (1.0 - p) / self.n
        if self.n < 2:
            return math.inf
        return self.m2 / (self.n - 1) / self.n

    @property
    def se(self):
        return math.sqrt(self.var_of_mean)


def compare(anchor, other, z=2.0, delta=0.0, z_move=None):
    """z-test of `other` against `anchor` (lower is better)."""
    if anchor.n == 0 or other.n == 0:
        return "unresolved"
    zm = z if z_move is None else z_move
    diff = other.mean - anchor.mean
    se = math.sqrt(anchor.var_of_mean + other.var_of_mean)
    if not se < math.inf:
        return "unresolved"
    if diff + zm * se < -delta:
        return "better"
    if diff - z * se > -delta:
        return "not_better"
    return "unresolved"


def ucb(s, z=2.0):
    """Pessimistic score, used to CHOOSE the answer."""
    return s.mean + z * s.se


def lcb(s, z=2.0):
    """Optimistic score, used only to decide where to spend shots."""
    return s.mean - z * s.se


def pick_incumbent(stats, z=2.0):
    cands = [(j, s) for j, s in stats.items() if s.n > 0]
    return min(cands, key=lambda js: ucb(js[1], z))[0] if cands else None


@dataclass
class ConvergenceResult:
    status: str      # move | visit | resample | converged | converged_capped
    move_to: int = None
    to_visit: list = field(default_factory=list)
    to_resample: list = field(default_factory=list)
    capped: list = field(default_factory=list)


def convergence_test(stats, anchor, n_grid, z=2.0, delta=0.0, max_shots=None,
                     min_anchor_shots=0, z_move=None):
    """Is `anchor` a statistically confirmed grid-local minimum?"""
    a = stats.get(anchor)
    if a is None or a.n == 0:
        return ConvergenceResult("resample", to_resample=[anchor])
    if a.n < min_anchor_shots and (max_shots is None or a.n < max_shots):
        return ConvergenceResult("resample", to_resample=[anchor])

    better, unseen, unresolved, capped = [], [], [], []
    for j in (anchor - 1, anchor + 1):
        if not 0 <= j < n_grid:
            continue
        s = stats.get(j)
        if s is None or s.n == 0:
            unseen.append(j)
            continue
        verdict = compare(a, s, z, delta, z_move)
        if verdict == "better":
            better.append(j)
        elif verdict == "unresolved":
            if max_shots is not None and a.n >= max_shots and s.n >= max_shots:
                capped.append(j)
            else:
                unresolved.append(j)

    if better:
        return ConvergenceResult("move", move_to=min(better, key=lambda j: stats[j].mean))
    if unseen:
        return ConvergenceResult("visit", to_visit=unseen)
    if unresolved:
        j = max(unresolved, key=lambda k: a.var_of_mean + stats[k].var_of_mean)
        targets = [i for i in (anchor, j) if max_shots is None or stats[i].n < max_shots]
        targets.sort(key=lambda i: -stats[i].var_of_mean)
        return ConvergenceResult("resample", to_resample=targets, capped=capped)
    return ConvergenceResult("converged_capped" if capped else "converged", capped=capped)


def noisy_amfs(xs, obj_sampler, budget_shots=30000, scout_points=5,
               max_families=3, init_shots=1000, step_shots=1000,
               max_shots_per_point=8000, z=2.0, z_start=1.0, delta=0.001,
               kind="bernoulli", admit_quantile=0.5,
               coverage_gap_fraction=0.25, explore_every=5,
               max_rounds_per_family=50, init_anchors=None,
               min_anchor_shots=None, early_stop=True, use_momentum=True,
               use_widen=True, max_step_fraction=0.15, max_rounds=100000):
    """Search a noisy 1D grid under a shot budget."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n < 2 or np.any(np.diff(xs) <= 0):
        raise ValueError("xs must have >= 2 strictly increasing points")
    if min(budget_shots, init_shots, step_shots, scout_points, max_families) < 1:
        raise ValueError("budget, shot batches, scout_points, max_families must be >= 1")

    z_final = float(z)
    z0 = z_final if z_start is None else min(float(z_start), z_final)
    stats, families, trace = {}, [], []
    total = 0
    separation = max(2, n // (2 * max_families))
    max_step = max(2, int(max_step_fraction * n))
    gap_tol = max(1, coverage_gap_fraction * n)
    stop_reason = "budget exhausted"
    if min_anchor_shots is None:
        min_anchor_shots = 2 * init_shots

    def zm():
        return z0 + (z_final - z0) * min(1.0, total / budget_shots)

    def sample(i, shots):
        nonlocal total
        i = int(i)
        shots = min(int(shots), budget_shots - total)
        if max_shots_per_point is not None:
            shots = min(shots, max_shots_per_point - (stats[i].n if i in stats else 0))
        if shots <= 0:
            return 0
        res = obj_sampler(i, shots)
        st = stats.setdefault(i, PointStats(kind))
        if kind == "bernoulli":
            errors, used = res
            st.add_counts(errors, used)
            total += used
        else:
            vals = list(res)
            for v in vals:
                st.add_sample(float(v))
            total += len(vals)
        return shots

    def new_family(i):
        return {"anchor": int(i), "converged": False, "rounds": 0, "dir": 0, "step": 1}

    def reset(f, i):
        f.update(anchor=int(i), converged=False, rounds=0, dir=0, step=1)

    def largest_gap():
        bounds = [-1] + sorted(stats) + [n]
        length, lo, hi = max((b - a - 1, a + 1, b - 1) for a, b in zip(bounds[:-1], bounds[1:]))
        return (length, (lo + hi) // 2) if length > 0 else (0, None)

    def offer_family(i):
        for f in families:
            if abs(f["anchor"] - i) < separation:
                if compare(stats[f["anchor"]], stats[i], z_final, delta, zm()) == "better":
                    reset(f, i)
                return
        active = [f for f in families if not f["converged"]]
        if len(active) < max_families:
            families.append(new_family(i))
        else:
            worst = max(active, key=lambda f: ucb(stats[f["anchor"]], z_final))
            if compare(stats[worst["anchor"]], stats[i], z_final, delta, zm()) == "better":
                reset(worst, i)

    def maybe_admit(i):
        means = sorted(s.mean for s in stats.values())
        if stats[i].mean <= means[int(admit_quantile * (len(means) - 1))]:
            offer_family(i)

    def merge_families():
        kept = []
        for f in sorted(families, key=lambda f: ucb(stats[f["anchor"]], z_final)):
            if all(abs(k["anchor"] - f["anchor"]) >= separation for k in kept):
                kept.append(f)
        families[:] = kept

    def widen_step(fam):
        a = fam["anchor"]
        widths, w = [], 2
        while w < max_step:
            widths.append(w)
            w *= 2
        widths.append(max_step)
        for w in widths:
            js = [j for j in (a - w, a + w) if 0 <= j < n]
            for j in js:
                if j not in stats:
                    sample(j, init_shots)
            good = [j for j in js if j in stats
                    and compare(stats[a], stats[j], z_final, delta, zm()) == "better"]
            if good:
                j = min(good, key=lambda k: stats[k].mean)
                fam["dir"], fam["step"] = (1 if j > a else -1), 2
                fam["anchor"] = j
                return True
        return False

    # ---- stage 1: warm start + global scouting -----------------------------
    warm = [int(a) for a in (init_anchors or []) if 0 <= int(a) < n]
    for i in warm:
        sample(i, init_shots)
    for i in np.unique(np.linspace(0, n - 1, min(n, scout_points), dtype=int)):
        if int(i) not in stats:
            sample(i, init_shots)
    if not stats:
        return None
    for i in [a for a in warm if a in stats] + sorted(stats, key=lambda k: stats[k].mean):
        if len(families) >= max_families:
            break
        if all(abs(f["anchor"] - i) >= separation for f in families):
            families.append(new_family(i))

    # ---- stage 2: best-first refinement with reserved exploration ----------
    rounds = 0
    while total < budget_shots:
        rounds += 1
        if rounds > max_rounds:
            stop_reason = "max rounds"
            break
        trace.append((total, pick_incumbent(stats, z_final)))
        merge_families()
        active = [f for f in families if not f["converged"]]

        if not active:
            b = pick_incumbent(stats, z_final)
            if not any(f["anchor"] == b for f in families):
                families.append(new_family(b))
                continue
            length, mid = largest_gap()
            if mid is None:
                stop_reason = "grid fully covered"
                break
            if early_stop and length <= gap_tol:
                stop_reason = "converged"
                break
            if sample(mid, init_shots):
                maybe_admit(mid)
            continue

        if explore_every and rounds % explore_every == 0:
            length, mid = largest_gap()
            if mid is not None and length > gap_tol and sample(mid, init_shots):
                maybe_admit(mid)

        fam = min(active, key=lambda f: lcb(stats[f["anchor"]], z_final))
        fam["rounds"] += 1

        if use_momentum and fam["dir"] and fam["step"] > 1:
            j = fam["anchor"] + fam["dir"] * fam["step"]
            if 0 <= j < n and j not in stats:
                sample(j, init_shots)
                if j in stats and compare(stats[fam["anchor"]], stats[j],
                                          z_final, delta, zm()) == "better":
                    fam["anchor"] = j
                    fam["step"] = min(fam["step"] * 2, max_step)
                    continue
            fam["dir"], fam["step"] = 0, 1

        r = convergence_test(stats, fam["anchor"], n, z_final, delta,
                             max_shots_per_point, min_anchor_shots, zm())
        if r.status == "move":
            d = 1 if r.move_to > fam["anchor"] else -1
            if d == fam["dir"] and fam["step"] >= 2:
                fam["step"] = min(fam["step"] * 2, max_step)
            else:
                fam["dir"], fam["step"] = d, 2
            fam["anchor"] = r.move_to
        elif r.status == "visit":
            for j in r.to_visit:
                sample(j, init_shots)
        elif r.status == "resample":
            for j in r.to_resample:
                sample(j, step_shots)
        elif use_widen and r.status == "converged_capped" and widen_step(fam):
            pass
        else:
            fam["converged"] = True
        if fam["rounds"] >= max_rounds_per_family:
            fam["converged"] = True

    best_i = pick_incumbent(stats, z_final)
    trace.append((total, best_i))
    return {
        "best_x": float(xs[best_i]), "best_index": int(best_i),
        "mean_score": stats[best_i].mean, "total_shots": total,
        "stop_reason": stop_reason, "points_visited": len(stats),
        "converged_families": sum(f["converged"] for f in families),
        "stats": stats, "trace": trace,
    }


def track_drifting_landscape(xs, sampler, steps, shots_per_step, monitor=True,
                             monitor_shots=2000, z=2.0, delta=0.001, **kwargs):
    """Recalibrate repeatedly while the optimum drifts."""
    if monitor and kwargs.get("kind", "bernoulli") != "bernoulli":
        raise ValueError("monitor=True supports bernoulli data only")
    history, prev_i, ref = [], None, None
    for step in range(steps):
        def at_step(i, shots, _s=step):
            return sampler(i, shots, _s)

        used, recal = 0, True
        if monitor and prev_i is not None:
            e, s = at_step(prev_i, monitor_shots)
            used = s
            now = PointStats("bernoulli")
            now.add_counts(e, s)
            if compare(now, ref, z=z, delta=delta) != "better":
                recal = False
        if not recal:
            history.append({"step": step, "index": prev_i, "recalibrated": False,
                            "shots": used, "mean_score": now.mean,
                            "stop_reason": "monitor ok"})
            continue
        res = noisy_amfs(xs, at_step, budget_shots=max(1, shots_per_step - used),
                         init_anchors=None if prev_i is None else [prev_i],
                         z=z, delta=delta, **kwargs)
        prev_i, ref = res["best_index"], res["stats"][res["best_index"]]
        history.append({"step": step, "index": prev_i, "recalibrated": True,
                        "shots": used + res["total_shots"],
                        "mean_score": res["mean_score"],
                        "stop_reason": res["stop_reason"]})
    return history


# ============================================================================
# envs.py[cite: 3]
# ============================================================================

def drift_fractions(drift, steps, speed, rng):
    """Optimum position as a fraction in [0, 1] of the grid, per step."""
    t = np.arange(steps)
    if drift == "static":
        return np.full(steps, 0.5)
    if drift == "sinusoid":
        return 0.5 + 0.28 * np.sin(0.5 * speed * t)
    if drift == "jump":
        segs = max(1, int(round(3 * speed)))
        levels = np.array([0.25, 0.75, 0.40, 0.65, 0.15, 0.85, 0.50, 0.30])
        return levels[((t * segs) // max(steps, 1)) % len(levels)]
    if drift == "walk":
        return np.clip(0.5 + np.cumsum(rng.normal(0, 0.05 * speed, steps)), 0.05, 0.95)
    raise ValueError("unknown drift: %s" % drift)


class SyntheticEnv:
    def __init__(self, n=91, steps=12, drift="sinusoid", landscape="unimodal",
                 base=0.05, curvature=0.0002, speed=1.0, seed=0):
        self.n, self.steps, self.base, self.curv = n, steps, base, curvature
        self.landscape = landscape
        self.xs = np.linspace(0.01, 0.10, n)
        self.rng = np.random.default_rng([seed, 1])
        frac = drift_fractions(drift, steps, speed, np.random.default_rng([seed, 0]))
        self.opt = np.round(frac * (n - 1)).astype(int)

    def true_p(self, step, i):
        i = np.asarray(i, dtype=float)
        scale = 91.0 / self.n
        p = self.base + self.curv * ((i - self.opt[step]) * scale) ** 2
        if self.landscape == "bimodal":
            decoy = (self.opt[step] + self.n // 3) % self.n
            p = np.minimum(p, self.base + 0.006 + 0.25 * self.curv * ((i - decoy) * scale) ** 2)
        return np.clip(p, 0.001, 0.999)

    def sampler(self, i, shots, step):
        return int(self.rng.binomial(shots, float(self.true_p(step, i)))), shots

    def regret(self, step, i):
        return float(self.true_p(step, i) - self.true_p(step, np.arange(self.n)).min())


def _circuit(d, rounds, p_data, q_meas):
    import stim
    return stim.Circuit.generated(
        "repetition_code:memory", distance=d, rounds=rounds,
        before_round_data_depolarization=p_data,
        before_measure_flip_probability=q_meas)


_DECODER_CACHE = {}


def _decoders(d, rounds, p_data, qdec):
    """One PyMatching decoder per assumed measurement-flip probability."""
    import pymatching
    key = (d, rounds, p_data, tuple(qdec))
    if key not in _DECODER_CACHE:
        _DECODER_CACHE[key] = [
            pymatching.Matching.from_detector_error_model(
                _circuit(d, rounds, p_data, q).detector_error_model(decompose_errors=True))
            for q in qdec]
    return _DECODER_CACHE[key]


def _table_row(args):
    d, rounds, p_data, qdec, q_true, shots, seed = args
    sam = _circuit(d, rounds, p_data, q_true).compile_detector_sampler(seed=seed)
    det, obs = sam.sample(shots, separate_observables=True)
    return [float(np.mean(np.any(m.decode_batch(det) != obs, axis=1)))
            for m in _decoders(d, rounds, p_data, qdec)]


def build_stim_table(p_data=0.10, d=5, rounds=5, n_dec=41, levels=21,
                     shots=100000, q_dec_range=(0.01, 0.40),
                     q_true_range=(0.02, 0.30), cache_dir="cache", workers=None):
    """Estimate (or load) T[k, i] = P_L(true q_k, decoder-assumed q_i)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "stim_rep_d%d_r%d_p%.3f_n%d_k%d_s%d.npz"
                        % (d, rounds, p_data, n_dec, levels, shots))
    if os.path.exists(path):
        return path
    qdec = tuple(float(x) for x in np.geomspace(*q_dec_range, n_dec))
    qtrue = np.geomspace(*q_true_range, levels)
    jobs = [(d, rounds, p_data, qdec, float(q), shots, 1000 + k)
            for k, q in enumerate(qtrue)]
    workers = workers or os.cpu_count() or 1
    if workers <= 1:
        rows = [_table_row(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_table_row, jobs))
    np.savez(path, table=np.array(rows), qdec=np.array(qdec), qtrue=qtrue,
             p_data=p_data, d=d, rounds=rounds)
    return path


class StimTableEnv:
    def __init__(self, table, steps=12, drift="sinusoid", speed=1.0, seed=0, **_):
        self.T = np.asarray(table["table"])
        self.xs = np.asarray(table["qdec"])
        K, self.n = self.T.shape
        self.steps = steps
        self.rng = np.random.default_rng([seed, 1])
        frac = drift_fractions(drift, steps, speed, np.random.default_rng([seed, 0]))
        self.level = np.round(frac * (K - 1)).astype(int)
        self.opt = self.T[self.level].argmin(axis=1)
        self.meta = {k: table[k] for k in ("p_data", "d", "rounds", "qtrue") if k in table}

    def true_p(self, step, i):
        return self.T[self.level[step], i]

    def sampler(self, i, shots, step):
        return int(self.rng.binomial(shots, float(self.T[self.level[step], i]))), shots

    def regret(self, step, i):
        row = self.T[self.level[step]]
        return float(row[i] - row.min())


class StimLiveEnv(StimTableEnv):
    """Every batch is sampled with Stim and decoded with PyMatching."""
    def __init__(self, table, steps=12, drift="sinusoid", speed=1.0, seed=0, **kw):
        super().__init__(table, steps, drift, speed, seed)
        m = self.meta
        self._d, self._r, self._p = int(m["d"]), int(m["rounds"]), float(m["p_data"])
        self._qtrue = np.asarray(m["qtrue"])
        self._dec = _decoders(self._d, self._r, self._p, tuple(float(x) for x in self.xs))
        self._sam = {}

    def sampler(self, i, shots, step):
        k = int(self.level[step])
        if k not in self._sam:
            seed = int(self.rng.integers(2 ** 31 - 1))
            self._sam[k] = _circuit(self._d, self._r, self._p,
                                    float(self._qtrue[k])).compile_detector_sampler(seed=seed)
        det, obs = self._sam[k].sample(shots, separate_observables=True)
        pred = self._dec[int(i)].decode_batch(det)
        return int(np.any(pred != obs, axis=1).sum()), shots


# ============================================================================
# baselines.py[cite: 1]
# ============================================================================

PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _bern_var(errors, shots):
    p = (errors + 0.5) / (shots + 1.0)
    return p * (1.0 - p) / shots


def b_uniform(n, sample, budget, prev, rng, **_):
    """Spread the budget evenly over the grid; take the lowest mean."""
    per = max(1, budget // n)
    means = [sample(i, per)[0] / per for i in range(n)]
    return int(np.argmin(means)), per * n


def b_pattern(n, sample, budget, prev, rng, probe_shots=1000, **_):
    """Warm-started pattern search with step halving and averaged probes."""
    cur = n // 2 if prev is None else int(prev)
    step, used, est = max(1, n // 8), 0, {}

    def m(i):
        nonlocal used
        if i not in est:
            if used + probe_shots > budget:
                return None
            e, s = sample(i, probe_shots)
            est[i], used = e / s, used + s
        return est[i]

    out = False
    while step >= 1 and not out:
        vals = {}
        for j in [cur] + [k for k in (cur - step, cur + step) if 0 <= k < n]:
            v = m(j)
            if v is None:
                out = True
                break
            vals[j] = v
        if not vals:
            break
        best = min(vals, key=vals.get)
        if best != cur:
            cur = best
        else:
            step //= 2
    return cur, used


def b_golden(n, sample, budget, prev, rng, **_):
    """Golden-section search over the whole grid with averaged probes."""
    n_probes = 2 * int(math.ceil(math.log(max(n, 3)) / math.log(1 / PHI))) + 3
    per = max(1, budget // n_probes)
    cache, used = {}, 0

    def f(i):
        nonlocal used
        if i not in cache:
            if used + per > budget:
                return None
            e, s = sample(i, per)
            cache[i], used = e / s, used + s
        return cache[i]

    a, b = 0, n - 1
    while b - a > 3:
        c = int(math.floor(b - PHI * (b - a)))
        d = int(math.ceil(a + PHI * (b - a)))
        fc, fd = f(c), f(d)
        if fc is None or fd is None:
            break
        if fc <= fd:
            b = d
        else:
            a = c
    for i in range(a, b + 1):
        if f(i) is None:
            break
    if not cache:
        return n // 2, 0
    return int(min(cache, key=cache.get)), used


def b_spsa(n, sample, budget, prev, rng, a=0.4, c=0.1, probe_shots=1000,
           alpha=0.602, gamma=0.101, **_):
    """1D SPSA on u = index/(n-1), two probes per iteration, warm-started."""
    K = max(1, budget // (2 * probe_shots))
    A = 0.1 * K
    u = 0.5 if prev is None else float(prev) / (n - 1)
    used = 0
    for k in range(K):
        ak = a / (k + 1 + A) ** alpha
        ck = max(c / (k + 1) ** gamma, 1.0 / (n - 1))
        s_ = rng.choice([-1.0, 1.0])
        up, um = min(1.0, u + ck * s_), max(0.0, u - ck * s_)
        e1, s1 = sample(int(round(up * (n - 1))), probe_shots)
        e2, s2 = sample(int(round(um * (n - 1))), probe_shots)
        used += s1 + s2
        g = (e1 / s1 - e2 / s2) / (up - um) if up != um else 0.0
        u = min(1.0, max(0.0, u - ak * g))
    return int(round(u * (n - 1))), used


_LENGTHSCALES = (0.05, 0.1, 0.2, 0.4)


def _rbf(a, b, ls):
    d = a[:, None] - b[None, :]
    return np.exp(-0.5 * (d / ls) ** 2)


def _gp_predict(x, y, v, grid):
    """GP regression with known per-point noise variances v."""
    mu = y.mean()
    sd = max(float(y.std()), math.sqrt(float(np.mean(v))), 1e-6)
    ys, vs = (y - mu) / sd, v / sd ** 2 + 1e-6
    best = None
    for ls in _LENGTHSCALES:
        K = _rbf(x, x, ls) + np.diag(vs)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        al = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        lml = -0.5 * float(ys @ al) - float(np.log(np.diag(L)).sum())
        if best is None or lml > best[0]:
            best = (lml, ls, L, al)
    if best is None:
        return None
    _, ls, L, al = best
    Ks = _rbf(x, grid, ls)
    mean = Ks.T @ al
    w = np.linalg.solve(L, Ks)
    var = np.maximum(1.0 - np.sum(w ** 2, axis=0), 1e-12)
    return mean * sd + mu, np.sqrt(var) * sd


def b_gp(n, sample, budget, prev, rng, probe_shots=1000, beta=2.0, **_):
    """1D Gaussian-process Bayesian optimisation with heteroscedastic noise."""
    grid = np.linspace(0.0, 1.0, n)
    obs, used = {}, 0
    queue = [] if prev is None else [int(prev)]
    queue += [int(round(q * (n - 1))) for q in (0.5, 0.17, 0.83)]
    seen_q = set()
    queue = [i for i in queue if not (i in seen_q or seen_q.add(i))]

    def fit():
        idx = np.array(sorted(obs))
        y = np.array([obs[i][0] / obs[i][1] for i in idx])
        v = np.array([_bern_var(*obs[i]) for i in idx])
        return idx, _gp_predict(grid[idx], y, v, grid)

    while used + probe_shots <= budget:
        if queue:
            i = queue.pop(0)
        else:
            _, pred = fit()
            if pred is None:
                i = int(rng.integers(n))
            else:
                m, s = pred
                i = int(np.argmin(m - beta * s))
        e, s_ = sample(i, probe_shots)
        old = obs.get(i, (0, 0))
        obs[i] = (old[0] + e, old[1] + s_)
        used += s_
    if not obs:
        return n // 2, 0
    if len(obs) >= 2:
        idx, pred = fit()
        if pred is not None:
            return int(np.argmin(pred[0])), used
    return int(min(obs, key=lambda i: obs[i][0] / obs[i][1])), used


def b_tpe(n, sample, budget, prev, rng, probe_shots=1000, n_startup=5, **_):
    """TPE via Optuna; each trial is one batch of shots."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(rng.integers(2 ** 31 - 1)),
                                           n_startup_trials=n_startup))
    if prev is not None:
        study.enqueue_trial({"i": int(prev)})
    obs, used = {}, [0]

    def objective(trial):
        i = trial.suggest_int("i", 0, n - 1)
        e, s = sample(i, probe_shots)
        old = obs.get(i, (0, 0))
        obs[i] = (old[0] + e, old[1] + s)
        used[0] += s
        return e / s

    study.optimize(objective, n_trials=max(1, budget // probe_shots))
    if not obs:
        return n // 2, 0
    score = {i: e / s + math.sqrt(_bern_var(e, s)) for i, (e, s) in obs.items()}
    return int(min(score, key=score.get)), used[0]


REGISTRY = {"uniform": b_uniform, "pattern": b_pattern, "golden": b_golden,
            "spsa": b_spsa, "gp_bo": b_gp, "tpe": b_tpe}


# ============================================================================
# benchmark.py[cite: 2]
# ============================================================================

METHODS = {
    "amfs":             ("amfs", {"monitor": True}),
    "amfs_nomonitor":   ("amfs", {"monitor": False}),
    "amfs_nomomentum":  ("amfs", {"monitor": True, "use_momentum": False}),
    "amfs_nowiden":     ("amfs", {"monitor": True, "use_widen": False}),
    "amfs_nominanchor": ("amfs", {"monitor": True, "min_anchor_shots": 0}),
    "amfs_fixedz":      ("amfs", {"monitor": True, "z_start": None}),
    "static":           ("static", {}),
    "uniform": ("base", {}), "pattern": ("base", {}), "golden": ("base", {}),
    "spsa": ("base", {}), "gp_bo": ("base", {}), "tpe": ("base", {}),
}
ABLATIONS = ["amfs", "amfs_nomonitor", "amfs_nomomentum", "amfs_nowiden",
             "amfs_nominanchor", "amfs_fixedz"]
MAIN = ["amfs", "amfs_nomonitor", "pattern", "uniform", "golden", "spsa",
        "gp_bo", "tpe", "static"]

GRIDS = {
    "amfs": [dict(z=zc, z_start=zs, delta=dl)
             for zc in (1.5, 2.0, 2.5) for zs in (0.5, 1.0) for dl in (0.002, 0.005, 0.01)],
    "pattern": [dict(probe_frac=f) for f in (1 / 30, 1 / 15, 1 / 8)],
    "spsa": [dict(a=a, c=c, probe_frac=f)
             for a in (0.1, 0.4, 1.6) for c in (0.05, 0.15) for f in (1 / 30, 1 / 12)],
    "gp_bo": [dict(probe_frac=f, beta=b) for f in (1 / 30, 1 / 15) for b in (1.0, 2.0)],
    "tpe": [dict(probe_frac=f) for f in (1 / 30, 1 / 15, 1 / 8)],
}
TUNE_METHOD = {"amfs": "amfs", "pattern": "pattern", "spsa": "spsa",
               "gp_bo": "gp_bo", "tpe": "tpe"}

SCEN = {
    "synthetic": {"tune": [("walk", "unimodal"), ("sinusoid", "bimodal")],
                  "eval": [("sinusoid", "unimodal"), ("jump", "bimodal"),
                           ("walk", "bimodal"), ("static", "bimodal")]},
    "stim": {"tune": [("walk", ""), ("sinusoid", "")],
             "eval": [("sinusoid", ""), ("jump", ""), ("walk", ""), ("static", "")]},
}
TUNE_SEED_OFFSET = 100000


def amfs_kwargs(budget, p):
    """AMFS shot batches scale with the per-step budget (~15 batches)."""
    init = max(50, budget // 15)
    return dict(scout_points=5, init_shots=init, step_shots=init,
                max_shots_per_point=4 * init, z=p.get("z", 2.0),
                z_start=p.get("z_start", 1.0), delta=p.get("delta", 0.002))


@lru_cache(maxsize=8)
def _load_table(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_env(spec, seed, live=False):
    if spec["env"] == "synthetic":
        return SyntheticEnv(n=spec.get("n", 91), steps=spec["steps"], drift=spec["drift"],
                            landscape=spec["landscape"], base=spec.get("noise", 0.05),
                            speed=spec.get("speed", 1.0), seed=seed)
    cls = StimLiveEnv if live else StimTableEnv
    return cls(_load_table(spec["table"]), steps=spec["steps"], drift=spec["drift"],
               speed=spec.get("speed", 1.0), seed=seed)


def run_method(method, params, env, budget, rng):
    """Return (regret per step, shots per step) for one method on one env."""
    kind, over = METHODS[method]
    steps, n = env.steps, env.n
    if kind in ("amfs", "static"):
        kw = amfs_kwargs(budget, params.get("amfs", {}))
        kw.update({k: v for k, v in over.items() if k != "monitor"})
        if kind == "static":
            res = noisy_amfs(env.xs, lambda i, s: env.sampler(i, s, 0),
                             budget_shots=budget, **kw)
            return (np.array([env.regret(t, res["best_index"]) for t in range(steps)]),
                    np.array([res["total_shots"]] + [0] * (steps - 1)))
        h = track_drifting_landscape(env.xs, env.sampler, steps, budget,
                                     monitor=over.get("monitor", False),
                                     monitor_shots=2 * kw["init_shots"], **kw)
        return (np.array([env.regret(r["step"], r["index"]) for r in h]),
                np.array([r["shots"] for r in h]))
    fn = REGISTRY[method]
    bp = dict(params.get(method, {}))
    if "probe_frac" in bp:
        bp["probe_shots"] = max(50, int(budget * bp.pop("probe_frac")))
    prev, regrets, shots = None, [], []
    for t in range(steps):
        idx, used = fn(n, lambda i, s, _t=t: env.sampler(i, s, _t), budget, prev, rng, **bp)
        prev = idx
        regrets.append(env.regret(t, idx))
        shots.append(used)
    return np.array(regrets), np.array(shots)


def run_task(task):
    spec, method, params, seed, budget, live = task
    env = make_env(spec, seed, live)
    rng = np.random.default_rng([seed, 12345])
    t0 = time.time()
    reg, sh = run_method(method, params, env, budget, rng)
    return dict(env=spec["env"], live=int(live), drift=spec["drift"],
                landscape=spec["landscape"], speed=spec.get("speed", 1.0),
                noise=spec.get("noise", 0.0), method=method, budget=budget,
                seed=seed, regret=float(np.mean(reg)), shots=float(np.mean(sh)),
                seconds=time.time() - t0)


def run_tasks(tasks, workers, label=""):
    out, t0, total = [], time.time(), len(tasks)
    step = max(1, total // 10)
    it = (map(run_task, tasks) if workers <= 1 else None)
    ex = None
    if workers > 1:
        ex = ProcessPoolExecutor(workers)
        it = ex.map(run_task, tasks, chunksize=4)
    try:
        for k, r in enumerate(it, 1):
            out.append(r)
            if k % step == 0 or k == total:
                print("  [%s] %d/%d tasks, %.0fs" % (label, k, total, time.time() - t0), flush=True)
    finally:
        if ex:
            ex.shutdown()
    return out


def _stats(v):
    v = np.asarray(v, dtype=float)
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return v.mean(), se, float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


def group(rows, keys):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def print_summary(rows, scen_keys, title):
    print("\n" + "=" * 92 + "\n" + title + "\n" + "=" * 92)
    for sk, srows in sorted(group(rows, scen_keys).items(), key=lambda kv: str(kv[0])):
        print("\n" + ", ".join("%s=%s" % (k, v) for k, v in zip(scen_keys, sk)))
        print("%-18s %9s %9s %9s %18s %10s" %
              ("method", "mean", "se", "median", "IQR", "shots/step"))
        for (m,), mrows in group(srows, ["method"]).items():
            mean, se, med, q1, q3 = _stats([r["regret"] for r in mrows])
            print("%-18s %9.5f %9.5f %9.5f   [%7.5f,%7.5f] %10.0f" %
                  (m, mean, se, med, q1, q3, np.mean([r["shots"] for r in mrows])))


def print_paired(rows, scen_keys, ref, title):
    """Paired difference (method - ref) of regret, by seed. Positive = ref better."""
    print("\n" + "-" * 92 + "\n" + title + "\n" + "-" * 92)
    for sk, srows in sorted(group(rows, scen_keys).items(), key=lambda kv: str(kv[0])):
        by = {m: {r["seed"]: r["regret"] for r in mr}
              for (m,), mr in group(srows, ["method"]).items()}
        if ref not in by:
            continue
        print("\n" + ", ".join("%s=%s" % (k, v) for k, v in zip(scen_keys, sk)))
        print("%-18s %14s %9s %9s" % ("method", "regret - ref", "se", "ref wins"))
        for m, d in by.items():
            if m == ref:
                continue
            seeds = sorted(set(d) & set(by[ref]))
            diff = np.array([d[s] - by[ref][s] for s in seeds])
            se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
            print("%-18s %+14.5f %9.5f %8.0f%%" % (m, diff.mean(), se, 100 * np.mean(diff > 0)))


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return None


def plot_budget(rows, path, title):
    plt = _plt()
    if plt is None:
        return
    scens = sorted(group(rows, ["drift", "landscape"]))
    fig, axs = plt.subplots(1, len(scens), figsize=(4.2 * len(scens), 3.6), squeeze=False)
    for ax, sk in zip(axs[0], scens):
        srows = [r for r in rows if (r["drift"], r["landscape"]) == sk]
        for (m,), mr in group(srows, ["method"]).items():
            xs, ys, es = [], [], []
            for (b,), br in sorted(group(mr, ["budget"]).items()):
                mean, se, *_ = _stats([r["regret"] for r in br])
                xs.append(np.mean([r["shots"] for r in br]))
                ys.append(mean)
                es.append(0 if np.isnan(se) else se)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2, label=m,
                        lw=2.2 if m == "amfs" else 1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("%s %s" % sk, fontsize=9)
        ax.set_xlabel("shots per step used")
        ax.set_ylabel("excess error (regret)")
    axs[0][0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_param(rows, param, path, title):
    plt = _plt()
    if plt is None:
        return
    scens = sorted(group(rows, ["drift", "landscape"]))
    fig, axs = plt.subplots(1, len(scens), figsize=(4.2 * len(scens), 3.6), squeeze=False)
    for ax, sk in zip(axs[0], scens):
        srows = [r for r in rows if (r["drift"], r["landscape"]) == sk]
        for (m,), mr in group(srows, ["method"]).items():
            xs, ys, es = [], [], []
            for (v,), vr in sorted(group(mr, [param]).items()):
                mean, se, *_ = _stats([r["regret"] for r in vr])
                xs.append(v)
                ys.append(mean)
                es.append(0 if np.isnan(se) else se)
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2, label=m,
                        lw=2.2 if m == "amfs" else 1.1)
        ax.set_yscale("log")
        ax.set_title("%s %s" % sk, fontsize=9)
        ax.set_xlabel(param)
        ax.set_ylabel("excess error (regret)")
    axs[0][0].legend(fontsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def spec_for(env, drift, land, table, steps=12, **over):
    s = {"env": env, "drift": drift, "landscape": land, "steps": steps, "table": table}
    s.update(over)
    return s


def tables_for(a, noises=None):
    """Build/load the Stim tables needed. Returns {p_data: path}."""
    noises = noises or [0.10]
    out = {}
    for p in noises:
        t0 = time.time()
        out[p] = build_stim_table(p_data=p, levels=a.table_levels, shots=a.table_shots,
                                  cache_dir=os.path.join(a.out, "cache"),
                                  workers=a.workers)
        print("  Stim table p_data=%.2f ready (%.0fs): %s" % (p, time.time() - t0, out[p]))
    return out


def env_list(a):
    return ["synthetic", "stim"] if a.env == "both" else [a.env]


def default_table(a):
    return tables_for(a)[0.10]


def do_tune(a, env):
    path = os.path.join(a.out, "tuned_%s.json" % env)
    table = default_table(a) if env == "stim" else None
    seeds = [TUNE_SEED_OFFSET + s for s in range(a.tune_seeds)]
    tuned = {}
    for fam, grid in GRIDS.items():
        if fam == "tpe" and a.no_tpe:
            continue
        grid = grid[::4] if a.quick else grid
        best = None
        for params in grid:
            tasks = [(spec_for(env, d, l, table), TUNE_METHOD[fam], {fam: params}, s, a.budget, False)
                     for d, l in SCEN[env]["tune"] for s in seeds]
            rows = run_tasks(tasks, a.workers, "tune %s" % fam)
            score = float(np.mean([r["regret"] for r in rows]))
            if best is None or score < best[0]:
                best = (score, params)
        tuned[fam] = best[1]
        print("tuned[%s][%s] = %s  (tuning regret %.5f)" % (env, fam, best[1], best[0]))
    os.makedirs(a.out, exist_ok=True)
    json.dump(tuned, open(path, "w"), indent=2)
    return tuned


def get_tuned(a, env):
    path = os.path.join(a.out, "tuned_%s.json" % env)
    if os.path.exists(path):
        return json.load(open(path))
    print("no tuned parameters for %s; tuning now" % env)
    return do_tune(a, env)


def do_ablate(a, env):
    tuned, table = get_tuned(a, env), (default_table(a) if env == "stim" else None)
    tasks = [(spec_for(env, d, l, table), m, tuned, s, a.budget, False)
             for d, l in SCEN[env]["eval"] for m in ABLATIONS for s in range(a.seeds)]
    rows = run_tasks(tasks, a.workers, "ablate %s" % env)
    for r in rows:
        r["env"] = env
    save_csv(rows, os.path.join(a.out, "ablation_%s.csv" % env))
    print_summary(rows, ["drift", "landscape"], "ABLATIONS [%s], %d shots/step" % (env, a.budget))
    print_paired(rows, ["drift", "landscape"], "amfs",
                 "Cost of removing each component (positive = full AMFS better)")


def do_sweep(a, env):
    tuned = get_tuned(a, env)
    methods = [m for m in MAIN if not (m == "tpe" and a.no_tpe)]
    budgets = (6000, 15000) if a.quick else (3000, 6000, 15000, 30000, 60000)
    speeds = (0.5, 2.0) if a.quick else (0.25, 0.5, 1.0, 2.0, 4.0)
    if env == "stim":
        noises = (0.06, 0.10) if a.quick else (0.06, 0.10, 0.14)
        tabs = tables_for(a, list(noises))
    else:
        noises = (0.02, 0.10) if a.quick else (0.02, 0.05, 0.10)
        tabs = {p: None for p in noises}
    base_table = tabs[0.10] if env == "stim" else None
    sc = SCEN[env]["eval"]
    eval_s = sc[:2] if a.quick else sc

    # 1. budget sweep (regret versus shots)
    tasks = [(spec_for(env, d, l, base_table), m, tuned, s, b, False)
             for d, l in eval_s for b in budgets for m in methods for s in range(a.seeds)]
    rows = run_tasks(tasks, a.workers, "budget %s" % env)
    save_csv(rows, os.path.join(a.out, "sweep_budget_%s.csv" % env))
    print_summary([r for r in rows if r["budget"] == 15000],
                  ["drift", "landscape"], "BUDGET SWEEP [%s] at 15000 shots/step" % env)
    plot_budget(rows, os.path.join(a.out, "sweep_budget_%s.png" % env),
                "Regret vs shots per step [%s]" % env)

    # 2. drift-speed sweep
    tasks = [(spec_for(env, d, l, base_table, speed=v), m, tuned, s, a.budget, False)
             for d, l in [x for x in eval_s if x[0] != "static"]
             for v in speeds for m in methods for s in range(a.seeds)]
    rows = run_tasks(tasks, a.workers, "speed %s" % env)
    save_csv(rows, os.path.join(a.out, "sweep_speed_%s.csv" % env))
    plot_param(rows, "speed", os.path.join(a.out, "sweep_speed_%s.png" % env),
               "Regret vs drift speed [%s]" % env)

    # 3. noise sweep
    tasks = [(spec_for(env, d, l, tabs[nz], noise=nz), m, tuned, s, a.budget, False)
             for d, l in eval_s for nz in noises for m in methods for s in range(a.seeds)]
    rows = run_tasks(tasks, a.workers, "noise %s" % env)
    save_csv(rows, os.path.join(a.out, "sweep_noise_%s.csv" % env))
    plot_param(rows, "noise", os.path.join(a.out, "sweep_noise_%s.png" % env),
               "Regret vs physical noise level [%s]" % env)
    print_paired([r for r in rows], ["drift", "landscape", "noise"], "amfs",
                 "Noise sweep [%s]: paired difference vs AMFS" % env)


def do_live(a):
    """Table env vs live Stim sampling: results should agree within noise."""
    tuned = get_tuned(a, "stim")
    table = default_table(a)
    seeds = range(a.live_seeds)
    methods = ["amfs", "pattern", "gp_bo"]
    rows = []
    for live in (False, True):
        tasks = [(spec_for("stim", d, "", table, steps=8), m, tuned, s, a.budget, live)
                 for d in ("sinusoid", "jump") for m in methods for s in seeds]
        rows += run_tasks(tasks, a.workers, "live" if live else "table")
    save_csv(rows, os.path.join(a.out, "validation_live.csv"))
    print("\nTABLE vs LIVE Stim sampling (regret mean +/- se; they should agree)")
    print("%-10s %-10s %-22s %-22s" % ("drift", "method", "table env", "live Stim"))
    for d in ("sinusoid", "jump"):
        for m in methods:
            cells = []
            for live in (0, 1):
                v = [r["regret"] for r in rows if r["drift"] == d and r["method"] == m
                     and r["live"] == live]
                mean, se, *_ = _stats(v)
                cells.append("%.5f +/- %.5f" % (mean, se))
            print("%-10s %-10s %-22s %-22s" % (d, m, cells[0], cells[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["quick", "tune", "ablate", "sweep", "live", "all"])
    ap.add_argument("--env", default="both", choices=["synthetic", "stim", "both"])
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--tune-seeds", type=int, default=8)
    ap.add_argument("--live-seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-tpe", action="store_true")
    ap.add_argument("--table-levels", type=int, default=21)
    ap.add_argument("--table-shots", type=int, default=100000)
    a = ap.parse_args()
    a.quick = a.mode == "quick"
    if a.quick:
        a.seeds, a.tune_seeds, a.live_seeds = min(a.seeds, 4), 2, 3
    os.makedirs(a.out, exist_ok=True)
    modes = ["tune", "ablate", "sweep", "live"] if a.mode in ("all", "quick") else [a.mode]
    for mode in modes:
        for env in env_list(a):
            if mode == "tune":
                do_tune(a, env)
            elif mode == "ablate":
                do_ablate(a, env)
            elif mode == "sweep":
                do_sweep(a, env)
        if mode == "live" and a.env in ("stim", "both"):
            do_live(a)


if __name__ == "__main__":
    main()
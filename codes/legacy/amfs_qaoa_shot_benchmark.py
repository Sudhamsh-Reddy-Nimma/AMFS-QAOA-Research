"""Proof-of-concept finite-shot QAOA MaxCut benchmark for AMFS.

Methods: AMFS coordinate search (using MAD_1.py's noisy_amfs), SPSA, a
prototype sequential-shot SPSA, SciPy COBYLA, NLopt BOBYQA, Gaussian-process
Bayesian optimisation, random search, and fixed initial parameters. This is an exact-statevector + finite-shot simulator,
not a real quantum-hardware experiment. It is an initial benchmark, not a
reproduction of iCANS/weCANS/DDS or a publication-grade comparison.

Example:
  python amfs_qaoa_shot_benchmark.py --graphs 6 --budget 3000 --shots-per-eval 100
"""
from __future__ import annotations
import argparse, csv, importlib.util, math, time
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

HERE = Path(__file__).resolve().parent
MAD_PATH = HERE / "MAD_1.py"
spec = importlib.util.spec_from_file_location("mad1_qaoa", MAD_PATH)
mad1 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mad1)

# p=1 QAOA angle domain for unweighted MaxCut.
BOUNDS = np.array([[0.0, np.pi], [0.0, np.pi / 2]])
X0 = np.array([0.30, 0.50])


def make_graph(seed: int, n_nodes: int = 8, degree: int = 3):
    return nx.random_regular_graph(degree, n_nodes, seed=seed)


def qaoa_distribution(edges, n: int, theta: np.ndarray):
    """Exact p=1 statevector probabilities, using C(z)=number of cut edges."""
    gamma, beta = float(theta[0]), float(theta[1])
    states = np.arange(1 << n, dtype=np.uint64)
    cuts = np.zeros(1 << n, dtype=np.float64)
    for u, v in edges:
        cuts += ((states >> u) ^ (states >> v)) & 1
    psi = np.full(1 << n, 1 / math.sqrt(1 << n), dtype=np.complex128)
    psi *= np.exp(-1j * gamma * cuts)
    # Apply exp(-i beta X) independently to each qubit.
    c, s = math.cos(beta), math.sin(beta)
    idx = np.arange(1 << n)
    for q in range(n):
        zero = idx[(idx & (1 << q)) == 0]
        one = zero | (1 << q)
        a0, a1 = psi[zero].copy(), psi[one].copy()
        psi[zero] = c * a0 - 1j * s * a1
        psi[one] = -1j * s * a0 + c * a1
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    return cuts.astype(int), probs


class QAOAOracle:
    def __init__(self, graph, seed):
        self.n = graph.number_of_nodes()
        self.edges = list(graph.edges())
        self.m = len(self.edges)
        self.rng = np.random.default_rng(seed)
        # Classical optimum is used only for evaluation, never by the optimizers.
        cuts, _ = qaoa_distribution(self.edges, self.n, np.array([0.0, 0.0]))
        self.opt_cut = int(cuts.max())
        self.cache = {}
        self.total_requested_shots = 0
        self.trace = []  # post-hoc diagnostics only; never used by optimizers

    def loss_samples(self, theta, shots):
        theta = np.array([np.clip(theta[0], *BOUNDS[0]), np.clip(theta[1], *BOUNDS[1])])
        shots = int(shots)
        self.total_requested_shots += shots
        self.trace.append((self.total_requested_shots, self.true_ratio(theta)))
        key = tuple(np.round(theta, 10))
        if key not in self.cache:
            cuts, probs = qaoa_distribution(self.edges, self.n, theta)
            self.cache[key] = (cuts, probs)
        cuts, probs = self.cache[key]
        observed = self.rng.choice(cuts, size=shots, p=probs)
        # Loss is 1 - cut / number_of_edges, so lower is better.
        return 1.0 - observed.astype(float) / self.m

    def true_ratio(self, theta):
        theta = np.array([np.clip(theta[0], *BOUNDS[0]), np.clip(theta[1], *BOUNDS[1])])
        cuts, probs = qaoa_distribution(self.edges, self.n, theta)
        return float(np.dot(cuts, probs) / self.opt_cut)

    def eval(self, theta, shots):
        vals = self.loss_samples(theta, shots)
        return float(np.mean(vals))


def clamp(theta):
    return np.clip(np.asarray(theta, float), BOUNDS[:, 0], BOUNDS[:, 1])


def run_amfs(oracle, budget, seed, grid_n=25, passes=2):
    """Coordinate-wise AMFS; total budget is divided across coordinates/passes."""
    theta = X0.copy()
    used = 0
    coord_budget = max(100, budget // (2 * passes))
    xs = np.linspace(0, 1, grid_n)
    for _ in range(passes):
        for j in range(2):
            fixed = theta.copy()
            lo, hi = BOUNDS[j]
            def sampler(i, shots):
                candidate = fixed.copy()
                candidate[j] = lo + (hi - lo) * xs[int(i)]
                return oracle.loss_samples(candidate, shots)
            res = mad1.noisy_amfs(
                xs, sampler, budget_shots=coord_budget, scout_points=5,
                max_families=2, init_shots=20, step_shots=20,
                max_shots_per_point=max(60, coord_budget // 3),
                z=1.5, z_start=0.8, delta=0.0005, kind="gaussian",
                max_rounds_per_family=20, max_rounds=1000,
                early_stop=True, use_momentum=True, use_widen=True)
            theta[j] = lo + (hi - lo) * xs[res["best_index"]]
            used += int(res["total_shots"])
    return theta, used


def run_spsa(oracle, budget, shots_per_eval, seed):
    rng = np.random.default_rng(seed)
    theta = X0.copy(); used = 0
    shots = max(1, shots_per_eval)
    iters = max(1, budget // (2 * shots))
    a, c, alpha, gamma = 0.15, 0.12, 0.602, 0.101
    for k in range(iters):
        delta = rng.choice([-1.0, 1.0], size=2)
        ck = c / ((k + 1) ** gamma)
        tp, tm = clamp(theta + ck * delta), clamp(theta - ck * delta)
        yp = oracle.eval(tp, shots); ym = oracle.eval(tm, shots); used += 2 * shots
        denom = 2.0 * ck * delta
        grad = (yp - ym) / denom
        ak = a / ((k + 5) ** alpha)
        theta = clamp(theta - ak * grad)
    return theta, used



def run_spsa_adaptive(oracle, budget, seed, pilot=20, max_per_side=160):
    """Prototype sequential shot allocation for SPSA; not an iCANS/weCANS reproduction."""
    rng = np.random.default_rng(seed)
    theta = X0.copy(); used = 0; k = 0
    while used + 2 * pilot <= budget:
        delta = rng.choice([-1.0, 1.0], size=2)
        ck = 0.12 / ((k + 1) ** 0.101)
        tp, tm = clamp(theta + ck * delta), clamp(theta - ck * delta)
        xp = list(oracle.loss_samples(tp, pilot)); xm = list(oracle.loss_samples(tm, pilot))
        used += 2 * pilot
        # Spend more shots only when the sign of the noisy comparison is uncertain.
        while len(xp) < max_per_side and len(xm) < max_per_side and used + 2 * pilot <= budget:
            mp, mm = float(np.mean(xp)), float(np.mean(xm))
            vp = float(np.var(xp, ddof=1)) / len(xp) if len(xp) > 1 else 0.25
            vm = float(np.var(xm, ddof=1)) / len(xm) if len(xm) > 1 else 0.25
            if abs(mp - mm) > 1.5 * math.sqrt(vp + vm):
                break
            xp.extend(oracle.loss_samples(tp, pilot)); xm.extend(oracle.loss_samples(tm, pilot))
            used += 2 * pilot
        yp, ym = float(np.mean(xp)), float(np.mean(xm))
        grad = (yp - ym) / (2.0 * ck * delta)
        ak = 0.15 / ((k + 5) ** 0.602)
        theta = clamp(theta - ak * grad)
        k += 1
    return theta, used


def run_cobyla(oracle, budget, shots_per_eval):
    used = [0]
    shots = max(1, shots_per_eval)
    max_evals = max(1, budget // shots)
    def objective(x):
        used[0] += shots
        return oracle.eval(clamp(x), shots)
    res = minimize(objective, X0.copy(), method="COBYLA",
                   bounds=list(map(tuple, BOUNDS)),
                   options={"maxiter": max_evals, "rhobeg": 0.25,
                            "tol": 0.02, "catol": 0.01})
    return clamp(res.x), used[0]


def run_gpbo(oracle, budget, shots_per_eval, seed):
    rng = np.random.default_rng(seed)
    shots = max(1, shots_per_eval); max_evals = max(1, budget // shots)
    X, y = [], []
    candidates = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(500, 2))
    for k in range(max_evals):
        if k == 0:
            x = X0.copy()
        elif k < 5:
            x = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
        else:
            kernel = ConstantKernel(1.0, (0.01, 10.0)) * Matern(length_scale=[0.7, 0.7], nu=2.5) + WhiteKernel(noise_level=0.01)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=0, random_state=seed + k,
                                          alpha=1e-4)
            gp.fit(np.asarray(X), np.asarray(y))
            mu, sd = gp.predict(candidates, return_std=True)
            acquisition = mu - 1.5 * sd  # lower loss is better
            for seen in X:
                acquisition[np.linalg.norm(candidates - seen, axis=1) < 0.025] = np.inf
            x = candidates[int(np.argmin(acquisition))]
        X.append(x.copy()); y.append(oracle.eval(x, shots))
    best = int(np.argmin(y))
    return clamp(X[best]), max_evals * shots



def run_bobyqa(oracle, budget, shots_per_eval):
    """Derivative-free BOBYQA via NLopt; requires the optional nlopt package."""
    try:
        import nlopt
    except ImportError as exc:
        raise RuntimeError("BOBYQA baseline requires `pip install nlopt`") from exc
    shots = max(1, shots_per_eval); max_evals = max(1, budget // shots)
    opt = nlopt.opt(nlopt.LN_BOBYQA, 2)
    opt.set_lower_bounds(BOUNDS[:, 0].tolist())
    opt.set_upper_bounds(BOUNDS[:, 1].tolist())
    opt.set_maxeval(max_evals)
    opt.set_initial_step([0.25, 0.15])
    opt.set_xtol_rel(0.01)
    def objective(x, grad):
        return oracle.eval(clamp(x), shots)
    opt.set_min_objective(objective)
    try:
        theta = clamp(opt.optimize(X0.copy()))
    except Exception:
        # NLopt may stop on roundoff-limited trust-region steps; use best available point.
        theta = X0.copy()
    return theta, min(max_evals * shots, oracle.total_requested_shots)


def run_random(oracle, budget, shots_per_eval, seed):
    rng = np.random.default_rng(seed)
    shots = max(1, shots_per_eval); max_evals = max(1, budget // shots)
    X = [X0.copy()]
    X.extend(rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(max(0, max_evals - 1), 2)))
    vals = [oracle.eval(x, shots) for x in X[:max_evals]]
    return clamp(X[int(np.argmin(vals))]), len(vals) * shots


def run_fixed(oracle, budget):
    # No optimisation: evaluate the common standard initial point once.
    shots = min(100, budget)
    oracle.eval(X0, shots)
    return X0.copy(), shots


def run_case(graph_seed, budget, shots_per_eval, seed, methods):
    graph = make_graph(graph_seed)
    out = []
    for method in methods:
        method_offset = {"amfs": 11, "spsa": 23, "spsa_adaptive": 29, "cobyla": 37, "bobyqa": 39, "gp_bo": 41, "random": 53, "fixed": 67}[method]
        oracle = QAOAOracle(graph, seed * 1009 + method_offset)
        start = time.perf_counter()
        if method == "amfs": theta, used = run_amfs(oracle, budget, seed)
        elif method == "spsa": theta, used = run_spsa(oracle, budget, shots_per_eval, seed)
        elif method == "spsa_adaptive": theta, used = run_spsa_adaptive(oracle, budget, seed)
        elif method == "cobyla": theta, used = run_cobyla(oracle, budget, shots_per_eval)
        elif method == "bobyqa": theta, used = run_bobyqa(oracle, budget, shots_per_eval)
        elif method == "gp_bo": theta, used = run_gpbo(oracle, budget, shots_per_eval, seed)
        elif method == "random": theta, used = run_random(oracle, budget, shots_per_eval, seed)
        elif method == "fixed": theta, used = run_fixed(oracle, budget)
        else: raise ValueError(method)
        elapsed = time.perf_counter() - start
        final_ratio = oracle.true_ratio(theta)
        target_shots = next((cum for cum, ratio in oracle.trace if ratio >= 0.78), None)
        out.append({"graph_seed": graph_seed, "seed": seed, "method": method,
                    "shots_used": int(used), "expected_cut_ratio": final_ratio,
                    "theta_gamma": float(theta[0]), "theta_beta": float(theta[1]),
                    "wall_seconds": elapsed, "target_078_final": int(final_ratio >= 0.78),
                    "shots_to_first_078_candidate": target_shots if target_shots is not None else "not_reached"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--shots-per-eval", type=int, default=100)
    ap.add_argument("--out", default="amfs_qaoa_shot_results.csv")
    args = ap.parse_args()
    if args.graphs < 1 or args.budget < 100 or args.shots_per_eval < 1:
        ap.error("graphs >= 1, budget >= 100, shots-per-eval >= 1 required")
    methods = ["amfs", "spsa", "spsa_adaptive", "cobyla", "bobyqa", "gp_bo", "random", "fixed"]
    rows = []
    for g in range(args.graphs):
        graph_seed = 100 + g
        for seed in range(2):
            rows.extend(run_case(graph_seed, args.budget, args.shots_per_eval,
                                 seed + g * 17, methods))
    out = Path(args.out).resolve()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {out}")
    print("\nSummary (higher expected_cut_ratio is better; post-hoc target=0.78):")
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        print(f"{method:7s} ratio={np.mean([r['expected_cut_ratio'] for r in subset]):.4f} "
              f"target_rate={np.mean([r['target_078_final'] for r in subset]):.1%} "
              f"median_shots_to_first_target={np.median([r['shots_to_first_078_candidate'] for r in subset if isinstance(r['shots_to_first_078_candidate'], int)]) if any(isinstance(r['shots_to_first_078_candidate'], int) for r in subset) else 'NA'} "
              f"shots={np.mean([r['shots_used'] for r in subset]):.0f} "
              f"wall={np.mean([r['wall_seconds'] for r in subset]):.3f}s")
    print("\nInterpretation: this is a small simulator benchmark. Do not claim hardware performance or novelty from these runs.")

if __name__ == "__main__":
    main()

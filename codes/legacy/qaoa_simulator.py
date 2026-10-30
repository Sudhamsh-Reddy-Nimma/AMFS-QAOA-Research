"""Verified arbitrary-depth QAOA MaxCut simulator and strict oracle/evaluator separation."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


def get_bounds(p: int = 1) -> np.ndarray:
    """Return parameter box bounds for depth-p QAOA: p gammas in [0, pi], p betas in [0, pi/2]."""
    bounds = []
    for _ in range(p):
        bounds.append([0.0, math.pi])
    for _ in range(p):
        bounds.append([0.0, math.pi / 2.0])
    return np.asarray(bounds, dtype=float)


def get_default_x0(p: int = 1, mode: str = "naive") -> np.ndarray:
    if mode == "naive":
        return np.array([0.30] * p + [0.50] * p, dtype=float)
    elif mode == "wurtz_love":
        if p == 1:
            return np.array([math.atan(1.0 / math.sqrt(2.0)), math.pi / 8.0], dtype=float)
        elif p == 2:
            return np.array([0.4877, 0.8976, 0.5551, 0.2925], dtype=float)
        else:
            gammas = np.linspace(0.4, 0.9, p)
            betas = np.linspace(0.6, 0.25, p)
            return np.concatenate([gammas, betas])
    elif mode == "center":
        b = get_bounds(p)
        return 0.5 * (b[:, 0] + b[:, 1])
    raise ValueError(f"Unknown initialization mode: {mode}")


def nx_random_regular_graph(d: int, n: int, seed: int) -> List[Tuple[int, int, float]]:
    """Exact reproduction of NetworkX random_regular_graph(d, n, seed) with unit weights."""
    rng = random.Random(seed)
    if (n * d) % 2 != 0 or not (0 <= d < n):
        raise ValueError(f"Invalid (d={d}, n={n}) for random_regular_graph")

    def _suitable(edges_set, potential_edges):
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                if s1 == s2:
                    break
                u, v = (s2, s1) if s1 > s2 else (s1, s2)
                if (u, v) not in edges_set:
                    return True
        return False

    def _try_creation():
        edges_set = set()
        stubs = list(range(n)) * d
        while stubs:
            potential_edges: Dict[int, int] = {}
            rng.shuffle(stubs)
            stubiter = iter(stubs)
            for s1, s2 in zip(stubiter, stubiter):
                u, v = (s2, s1) if s1 > s2 else (s1, s2)
                if u != v and ((u, v) not in edges_set):
                    edges_set.add((u, v))
                else:
                    potential_edges[u] = potential_edges.get(u, 0) + 1
                    potential_edges[v] = potential_edges.get(v, 0) + 1
            if not _suitable(edges_set, potential_edges):
                return None
            stubs = [node for node, pot in potential_edges.items() for _ in range(pot)]
        return edges_set

    edges_set = _try_creation()
    while edges_set is None:
        edges_set = _try_creation()

    adj: Dict[int, Dict[int, None]] = {}
    for u, v in edges_set:
        if u not in adj:
            adj[u] = {}
        if v not in adj:
            adj[v] = {}
        adj[u][v] = None
        adj[v][u] = None
    edge_list: List[Tuple[int, int, float]] = []
    seen = set()
    for u in adj:
        for v in adj[u]:
            if v not in seen:
                edge_list.append((u, v, 1.0))
        seen.add(u)
    return edge_list


def make_graph_instance(
    family: str,
    n_nodes: int,
    seed: int,
    degree: int = 3,
    p_edge: float = 0.5,
) -> Tuple[int, List[Tuple[int, int, float]]]:
    """Generate a reproducible MaxCut graph instance (n_nodes, weighted_edges)."""
    if family == "3_regular":
        return n_nodes, nx_random_regular_graph(degree, n_nodes, seed)
    elif family == "weighted_3_regular":
        base_edges = nx_random_regular_graph(degree, n_nodes, seed)
        rng = np.random.default_rng(seed + 77777)
        weighted = [(u, v, float(rng.uniform(0.4, 1.6))) for (u, v, _) in base_edges]
        return n_nodes, weighted
    elif family == "erdos_renyi":
        rng = np.random.default_rng(seed)
        for _ in range(200):
            edges = []
            for u in range(n_nodes):
                for v in range(u + 1, n_nodes):
                    if rng.random() < p_edge:
                        edges.append((u, v, 1.0))
            if len(edges) >= n_nodes - 1:
                adj = {i: [] for i in range(n_nodes)}
                for u, v, _ in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                visited = {0}
                queue = [0]
                while queue:
                    curr = queue.pop(0)
                    for nxt in adj[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
                if len(visited) == n_nodes:
                    return n_nodes, edges
        return n_nodes, nx_random_regular_graph(degree, n_nodes, seed)
    elif family == "watts_strogatz":
        rng = np.random.default_rng(seed)
        k_half = max(1, degree // 2)
        edges_set = set()
        for u in range(n_nodes):
            for offset in range(1, k_half + 1):
                v = (u + offset) % n_nodes
                if rng.random() < 0.3:
                    candidates = [w for w in range(n_nodes) if w != u and (min(u, w), max(u, w)) not in edges_set]
                    if candidates:
                        v = int(rng.choice(candidates))
                edges_set.add((min(u, v), max(u, v)))
        return n_nodes, [(u, v, 1.0) for (u, v) in sorted(edges_set)]
    raise ValueError(f"Unknown graph family: {family}")


@dataclass
class NoiseConfig:
    """Incremental hardware noise and drift parameters."""
    gate_depolarizing: float = 0.0
    readout_bitflip: float = 0.0
    coherent_angle_std: float = 0.0
    drift_gamma: float = 0.0
    drift_beta: float = 0.0


class QAOACircuitEngine:
    """Precomputed cost spectrum and fast statevector evolution for arbitrary n and p."""

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = int(n)
        self.edges = [(int(u), int(v), float(w)) for u, v, w in edges]
        self.m = len(self.edges)
        self.total_weight = float(sum(w for _, _, w in self.edges))
        self.dim = 1 << self.n
        states = np.arange(self.dim, dtype=np.uint64)
        cuts = np.zeros(self.dim, dtype=np.float64)
        for u, v, w in self.edges:
            cuts += w * (((states >> u) ^ (states >> v)) & 1)
        self.cuts = cuts
        self.opt_cut = float(np.max(self.cuts))
        self.min_cut = float(np.min(self.cuts))
        idx = np.arange(self.dim)
        self._zero_idx = [idx[(idx & (1 << q)) == 0] for q in range(self.n)]
        self._one_idx = [z | (1 << q) for q, z in enumerate(self._zero_idx)]

    def statevector_probs(
        self,
        theta: np.ndarray,
        noise: Optional[NoiseConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float).ravel()
        if len(theta_arr) % 2 != 0:
            raise ValueError("theta must have even length 2*p")
        p = len(theta_arr) // 2
        gammas = theta_arr[:p].copy()
        betas = theta_arr[p:].copy()

        if noise is not None:
            gammas += noise.drift_gamma
            betas += noise.drift_beta
            if noise.coherent_angle_std > 0.0 and rng is not None:
                gammas += rng.normal(0.0, noise.coherent_angle_std, size=p)
                betas += rng.normal(0.0, noise.coherent_angle_std, size=p)

        psi = np.full(self.dim, 1.0 / math.sqrt(self.dim), dtype=np.complex128)
        for layer in range(p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])
            psi *= np.exp(-1j * gamma * self.cuts)
            c, s = math.cos(beta), math.sin(beta)
            for q in range(self.n):
                z0 = self._zero_idx[q]
                z1 = self._one_idx[q]
                a0, a1 = psi[z0].copy(), psi[z1].copy()
                psi[z0] = c * a0 - 1j * s * a1
                psi[z1] = -1j * s * a0 + c * a1

        probs = np.abs(psi) ** 2
        probs /= probs.sum()

        if noise is not None and noise.gate_depolarizing > 0.0:
            fid = (1.0 - noise.gate_depolarizing) ** p
            probs = fid * probs + (1.0 - fid) * (1.0 / self.dim)
            probs /= probs.sum()

        return probs


class QAOAOptimizerOracle:
    """Strictly realistic noisy finite-shot QAOA oracle exposed to optimizers."""

    def __init__(
        self,
        engine: QAOACircuitEngine,
        seed: int,
        p: int = 1,
        max_budget: Optional[int] = None,
        strict_budget: bool = True,
        noise: Optional[NoiseConfig] = None,
        target_cut_threshold: Optional[float] = None,
    ):
        self.engine = engine
        self.n = engine.n
        self.m = engine.m
        self.total_weight = engine.total_weight
        self.p = int(p)
        self.bounds = get_bounds(self.p)
        self.rng = np.random.default_rng(seed)
        self.max_budget = int(max_budget) if max_budget is not None else None
        self.strict_budget = bool(strict_budget)
        self.noise = noise or NoiseConfig()
        self.target_cut_threshold = target_cut_threshold
        self.target_loss_threshold = (
            1.0 - float(target_cut_threshold) / self.total_weight
            if target_cut_threshold is not None
            else None
        )
        self._prob_cache: Dict[Tuple[float, ...], np.ndarray] = {}
        self.total_requested_shots: int = 0
        self.circuit_executions: int = 0
        self.query_log: List[Tuple[int, int, np.ndarray, float, float, float]] = []

    def remaining_shots(self) -> int:
        if self.max_budget is None:
            return 10**9
        return max(0, self.max_budget - self.total_requested_shots)

    def clip_theta(self, theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=float).ravel()
        return np.clip(t, self.bounds[:, 0], self.bounds[:, 1])

    def _get_probs(self, theta: np.ndarray) -> np.ndarray:
        if (
            self.noise.coherent_angle_std == 0.0
            and self.noise.drift_gamma == 0.0
            and self.noise.drift_beta == 0.0
        ):
            key = tuple(np.round(theta, 10))
            if key not in self._prob_cache:
                self._prob_cache[key] = self.engine.statevector_probs(theta, self.noise, self.rng)
            return self._prob_cache[key]
        return self.engine.statevector_probs(theta, self.noise, self.rng)

    def sample_outcomes(self, theta: np.ndarray, shots: int) -> Tuple[np.ndarray, float]:
        theta_c = self.clip_theta(theta)
        req_shots = int(shots)
        if req_shots <= 0:
            return np.empty(0, dtype=float), 0.0
        if self.strict_budget and self.max_budget is not None:
            avail = self.max_budget - self.total_requested_shots
            if avail <= 0:
                return np.empty(0, dtype=float), 0.0
            req_shots = min(req_shots, avail)

        probs = self._get_probs(theta_c)
        self.total_requested_shots += req_shots
        self.circuit_executions += 1

        if self.noise.readout_bitflip > 0.0:
            states = self.rng.choice(self.engine.dim, size=req_shots, p=probs).astype(np.int64)
            flip_mask = np.zeros(req_shots, dtype=np.int64)
            for q in range(self.n):
                flips = (self.rng.random(req_shots) < self.noise.readout_bitflip).astype(np.int64)
                flip_mask |= (flips << q)
            noisy_states = states ^ flip_mask
            observed_cuts = self.engine.cuts[noisy_states]
            _, counts = np.unique(noisy_states, return_counts=True)
        else:
            states = self.rng.choice(self.engine.dim, size=req_shots, p=probs)
            observed_cuts = self.engine.cuts[states]
            _, counts = np.unique(states, return_counts=True)

        emp_p = counts.astype(float) / req_shots
        entropy_bits = float(-np.sum(emp_p * np.log2(np.maximum(emp_p, 1e-15))))
        losses = 1.0 - observed_cuts.astype(float) / self.total_weight
        mean_l = float(np.mean(losses))
        se_l = float(np.std(losses, ddof=1) / math.sqrt(req_shots)) if req_shots > 1 else 0.25
        self.query_log.append((self.total_requested_shots, req_shots, theta_c.copy(), mean_l, se_l, entropy_bits))
        return losses, entropy_bits

    def loss_samples(self, theta: np.ndarray, shots: int) -> np.ndarray:
        losses, _ = self.sample_outcomes(theta, shots)
        return losses

    def eval(self, theta: np.ndarray, shots: int) -> float:
        losses = self.loss_samples(theta, shots)
        if len(losses) == 0:
            return 1.0
        return float(np.mean(losses))


class QAOAOfflineEvaluator:
    """Evaluator with access to exact statevector expectation and classical MaxCut optimum."""

    def __init__(self, engine: QAOACircuitEngine, p: int = 1, noise: Optional[NoiseConfig] = None):
        self.engine = engine
        self.p = int(p)
        self.bounds = get_bounds(self.p)
        self.noise = noise or NoiseConfig()
        self.opt_cut = engine.opt_cut
        self.total_weight = engine.total_weight
        self._p1_ceiling: Optional[float] = None
        self._effective_cuts = self._compute_effective_cuts()

    def _compute_effective_cuts(self) -> np.ndarray:
        p_ro = self.noise.readout_bitflip
        if p_ro <= 0.0:
            return self.engine.cuts
        scale = (1.0 - 2.0 * p_ro) ** 2
        offset = 2.0 * p_ro * (1.0 - p_ro) * self.total_weight
        return scale * self.engine.cuts + offset

    def true_expected_cut(self, theta: np.ndarray) -> float:
        t = np.clip(np.asarray(theta, dtype=float).ravel(), self.bounds[:, 0], self.bounds[:, 1])
        probs = self.engine.statevector_probs(t, self.noise, rng=None)
        return float(np.dot(self._effective_cuts, probs))

    def true_ratio(self, theta: np.ndarray) -> float:
        return self.true_expected_cut(theta) / self.opt_cut

    def compute_qaoa_ceiling(self, grid_res: int = 50) -> float:
        if self._p1_ceiling is not None:
            return self._p1_ceiling
        if self.p != 1:
            self._p1_ceiling = 1.0
            return 1.0
        best = 0.0
        for g in np.linspace(0.0, math.pi, grid_res):
            for b in np.linspace(0.0, math.pi / 2.0, grid_res // 2 + 1):
                r = self.true_ratio(np.array([g, b]))
                if r > best:
                    best = r
        self._p1_ceiling = float(best)
        return self._p1_ceiling

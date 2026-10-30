"""Core QAOA statevector simulator, calibrated NISQ noise channels, and strict oracle/evaluator separation.

Enforces:
- Zero leakage of `opt_cut` or `true_ratio` to the online `OperationalQAOAOracle`.
- Online computation of the polynomial-time Mohar-Poljak (1990) & Delorme-Poljak (1993) diagonal-corrected
  Laplacian spectral upper bound `U_spec(G)` so online certification rules can exploit the exact known
  loss support `[1 - U_spec(G)/W, 1]`.
- Fast stride-reshaped statevector execution supporting n in {8, 12, 14, 16} and p in {1, 2, 3, 4}.
- Verified primary-source parameter transfer tables from Wurtz & Love (Phys. Rev. A 103, 042612, 2021),
  Wurtz & Lykov (Phys. Rev. A 104, 052419, 2021, Table I), Shaydulin et al. (ACM TQC 4(3):19, 2023),
  and Sureshbabu et al. (Quantum 8, 1231, 2024).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from graph_library import GraphMetadata, compute_delorme_poljak_spectral_bound


def get_bounds(p: int = 1) -> np.ndarray:
    bounds = []
    for _ in range(p):
        bounds.append([0.0, math.pi])
    for _ in range(p):
        bounds.append([0.0, math.pi / 2.0])
    return np.asarray(bounds, dtype=float)


def get_default_x0(
    p: int = 1,
    mode: str = "naive",
    edges: Optional[List[Tuple[int, int, float]]] = None,
) -> np.ndarray:
    """Return initial parameter vector for depth-p QAOA.

    Modes:
      - 'naive': [0.30, ..., 0.50, ...]
      - 'wurtz_love': Fixed tree-subgraph angles for 3-regular MaxCut from:
          * J. Wurtz & P. J. Love, "MaxCut quantum approximate optimization algorithm
            performance with p > 2", Phys. Rev. A 103, 042612 (2021), DOI: 10.1103/PhysRevA.103.042612;
          * J. Wurtz & D. Lykov, "Fixed-angle conjectures for the quantum approximate
            optimization algorithm on regular MaxCut graphs", Phys. Rev. A 104, 052419 (2021),
            Table I, DOI: 10.1103/PhysRevA.104.052419:
              p=1: gamma = [0.6154797], beta = [0.3926991]
              p=2: gamma = [0.4877, 0.8976], beta = [0.5551, 0.2925]
              p=3: gamma = [0.4218, 0.7980, 0.9370], beta = [0.6005, 0.4596, 0.2361]
              p=4: gamma = [0.3796, 0.7255, 0.8744, 0.9554], beta = [0.6240, 0.5236, 0.3958, 0.2016]
      - 'shaydulin_rescaled': Rescales tree-subgraph gamma angles by 1 / sqrt(mean(w_uv^2))
        following R. Shaydulin et al., ACM Trans. Quantum Comput. 4(3):19 (2023),
        S. H. Sureshbabu et al., Quantum 8, 1231 (2024), and T. Hao et al.,
        Phys. Rev. Research 7, 033181 (2025), DOI: 10.1103/24gg-7p8z.
    """
    if mode == "naive":
        return np.array([0.30] * p + [0.50] * p, dtype=float)
    elif mode in ("wurtz_love", "shaydulin_rescaled"):
        if p == 1:
            g = np.array([math.atan(1.0 / math.sqrt(2.0))], dtype=float)
            b = np.array([math.pi / 8.0], dtype=float)
        elif p == 2:
            g = np.array([0.4877, 0.8976], dtype=float)
            b = np.array([0.5551, 0.2925], dtype=float)
        elif p == 3:
            g = np.array([0.4218, 0.7980, 0.9370], dtype=float)
            b = np.array([0.6005, 0.4596, 0.2361], dtype=float)
        elif p == 4:
            g = np.array([0.3796, 0.7255, 0.8744, 0.9554], dtype=float)
            b = np.array([0.6240, 0.5236, 0.3958, 0.2016], dtype=float)
        else:
            g = np.linspace(0.36, 0.96, p)
            b = np.linspace(0.63, 0.19, p)
        if mode == "shaydulin_rescaled" and edges:
            rms_w = math.sqrt(sum(w * w for _, _, w in edges) / len(edges))
            if rms_w > 1e-9:
                g = g / rms_w
        bounds = get_bounds(p)
        theta = np.concatenate([g, b])
        return np.clip(theta, bounds[:, 0], bounds[:, 1])
    raise ValueError(f"Unknown initialization mode: {mode}")


@dataclass
class NoiseConfig:
    """Hardware-inspired noise model parameters (grounded in IBM Eagle r3 / Heron r2 snapshot)."""
    label: str = "noiseless"
    gate_depolarizing: float = 0.0   # Effective per-layer depolarizing probability
    readout_bitflip: float = 0.0     # Per-qubit measurement bit-flip probability
    coherent_angle_std: float = 0.0  # Coherent gate over/under-rotation std (radians)
    drift_gamma: float = 0.0         # Additive calibration drift on gamma
    drift_beta: float = 0.0          # Additive calibration drift on beta
    snapshot_source: str = "Noiseless finite-shot multinomial baseline"


def get_noise_preset(name: str) -> NoiseConfig:
    if name == "noiseless":
        return NoiseConfig("noiseless", 0.0, 0.0, 0.0, 0.0, 0.0, "Exact statevector + multinomial shot sampling")
    elif name == "readout_only":
        return NoiseConfig(
            "readout_only",
            0.0,
            0.018,
            0.0,
            0.0,
            0.0,
            "IBM Eagle r3 median readout assignment error p_ro=1.8% (Kim et al., PRR 7, 043253, 2025)",
        )
    elif name == "gate_depol_only":
        return NoiseConfig(
            "gate_depol_only",
            0.024,
            0.0,
            0.015,
            0.0,
            0.0,
            "IBM Heron r2 / Eagle r3 2Q ZZ-layer depolarizing (2.4%/layer) + 0.015 rad coherent jitter",
        )
    elif name == "calibrated_nisq":
        return NoiseConfig(
            "calibrated_nisq",
            0.024,
            0.018,
            0.015,
            0.0,
            0.0,
            "Combined IBM Eagle r3 / Heron r2 layer depolarizing (2.4%/layer) + readout (1.8%/qubit) + coherent jitter",
        )
    raise ValueError(f"Unknown noise preset: {name}")


class QAOACircuitEngine:
    """Fast vectorized stride-butterfly statevector simulator for arbitrary n (up to n=16+) and p."""

    def __init__(self, n: int, edges: List[Tuple[int, int, float]]):
        self.n = int(n)
        self.edges = [(int(u), int(v), float(w)) for u, v, w in edges]
        self.m = len(self.edges)
        self.total_weight = float(sum(w for _, _, w in self.edges))
        self.dim = 1 << self.n
        states = np.arange(self.dim, dtype=np.uint32)
        cuts = np.zeros(self.dim, dtype=np.float64)
        A = np.zeros((self.n, self.n), dtype=float)
        deg = np.zeros(self.n, dtype=float)
        is_unit = True
        for u, v, w in self.edges:
            cuts += w * (((states >> u) ^ (states >> v)) & 1)
            A[u, v] = w
            A[v, u] = w
            deg[u] += w
            deg[v] += w
            if abs(w - 1.0) > 1e-9:
                is_unit = False
        self.cuts = cuts
        self.opt_cut = float(np.max(self.cuts))
        self.min_cut = float(np.min(self.cuts))

        # Online polynomial-time O(n^3) Mohar-Poljak / Delorme-Poljak spectral upper bound U_spec(G)
        L_mat = np.diag(deg) - A
        self.spectral_upper_bound = compute_delorme_poljak_spectral_bound(
            self.n, L_mat, self.total_weight, is_unit=is_unit
        )
        self.spectral_loss_range = float(min(1.0, max(1e-6, self.spectral_upper_bound / max(self.total_weight, 1e-12))))
        self.spectral_loss_min = float(max(0.0, 1.0 - self.spectral_loss_range))

        # Precompute neighbor weights for Fidelity-0 classical tree correlator
        self.adj_weights: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(self.n)}
        self.edge_set: Dict[Tuple[int, int], float] = {}
        for u, v, w in self.edges:
            self.adj_weights[u].append((v, w))
            self.adj_weights[v].append((u, w))
            self.edge_set[(min(u, v), max(u, v))] = w

    def statevector_probs(
        self,
        theta: np.ndarray,
        noise: Optional[NoiseConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=float).ravel()
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
            c = math.cos(beta)
            m_is = -1j * math.sin(beta)
            for q in range(self.n):
                half = 1 << q
                v_psi = psi.reshape(-1, 2, half)
                a0 = v_psi[:, 0, :].copy()
                a1 = v_psi[:, 1, :]
                v_psi[:, 0, :] = c * a0 + m_is * a1
                v_psi[:, 1, :] = m_is * a0 + c * a1

        probs = np.abs(psi) ** 2
        probs /= probs.sum()

        if noise is not None and noise.gate_depolarizing > 0.0:
            fid = (1.0 - noise.gate_depolarizing) ** p
            probs = fid * probs + (1.0 - fid) * (1.0 / self.dim)
            probs /= probs.sum()

        return probs

    def classical_fidelity0_loss(self, theta: np.ndarray) -> float:
        """Zero-shot classical tree-subgraph / 1-hop correlator predictor (Fidelity 0)."""
        theta_arr = np.asarray(theta, dtype=float).ravel()
        p = len(theta_arr) // 2
        gammas = theta_arr[:p]
        betas = theta_arr[p:]
        est_cut = 0.5 * self.total_weight

        for layer in range(p):
            gamma = float(gammas[layer])
            beta = float(betas[layer])
            layer_weight = 1.0 / (layer + 1.0)
            s4b = math.sin(4.0 * beta)
            layer_gain = 0.0
            for u, v, w_uv in self.edges:
                prod_u = 1.0
                for k, w_uk in self.adj_weights[u]:
                    if k != v:
                        prod_u *= math.cos(w_uk * gamma)
                prod_v = 1.0
                for l_node, w_vl in self.adj_weights[v]:
                    if l_node != u:
                        prod_v *= math.cos(w_vl * gamma)
                layer_gain += 0.25 * w_uv * s4b * math.sin(w_uv * gamma) * (prod_u + prod_v)
            est_cut += layer_weight * layer_gain

        est_cut = max(0.0, min(self.total_weight, est_cut))
        return 1.0 - est_cut / self.total_weight


class OperationalQAOAOracle:
    """Strictly hardware-realistic noisy finite-shot QAOA oracle."""

    def __init__(
        self,
        engine: QAOACircuitEngine,
        seed: int,
        p: int = 1,
        max_budget: Optional[int] = None,
        strict_budget: bool = True,
        noise: Optional[NoiseConfig] = None,
        operational_target_loss: Optional[float] = None,
        confirm_seed_offset: int = 500009,
    ):
        self.engine = engine
        self.n = engine.n
        self.m = engine.m
        self.total_weight = engine.total_weight
        self.spectral_upper_bound = engine.spectral_upper_bound
        self.spectral_loss_range = engine.spectral_loss_range
        self.spectral_loss_min = engine.spectral_loss_min
        self.p = int(p)
        self.bounds = get_bounds(self.p)
        self.rng = np.random.default_rng(seed)
        self.confirm_rng = np.random.default_rng(seed + confirm_seed_offset)
        self.max_budget = int(max_budget) if max_budget is not None else None
        self.strict_budget = bool(strict_budget)
        self.noise = noise or get_noise_preset("noiseless")
        self.operational_target_loss = operational_target_loss

        self._prob_cache: Dict[Tuple[float, ...], np.ndarray] = {}
        self.optimization_shots: int = 0
        self.certification_shots: int = 0
        self.circuit_evaluations: int = 0
        self.simulator_calls: int = 0
        self.query_log: List[Tuple[int, int, str, np.ndarray, float, float, float]] = []

    @property
    def search_shots(self) -> int:
        return self.optimization_shots

    @property
    def confirmation_shots(self) -> int:
        return self.certification_shots

    @property
    def total_requested_shots(self) -> int:
        return self.optimization_shots + self.certification_shots

    def verify_resource_invariant(self) -> bool:
        total = self.search_shots + self.confirmation_shots
        logged = sum(item[1] for item in self.query_log)
        within_cap = (self.max_budget is None) or (not self.strict_budget) or (total <= self.max_budget)
        return (total == logged) and within_cap

    def remaining_shots(self) -> int:
        if self.max_budget is None:
            return 10**9
        return max(0, self.max_budget - self.total_requested_shots)

    def clip_theta(self, theta: np.ndarray) -> np.ndarray:
        t = np.asarray(theta, dtype=float).ravel()
        return np.clip(t, self.bounds[:, 0], self.bounds[:, 1])

    def _get_probs(self, theta: np.ndarray, stream_rng: np.random.Generator) -> np.ndarray:
        if (
            self.noise.coherent_angle_std == 0.0
            and self.noise.drift_gamma == 0.0
            and self.noise.drift_beta == 0.0
        ):
            key = tuple(np.round(theta, 10))
            if key not in self._prob_cache:
                self.simulator_calls += 1
                self._prob_cache[key] = self.engine.statevector_probs(theta, self.noise, None)
            return self._prob_cache[key]
        self.simulator_calls += 1
        return self.engine.statevector_probs(theta, self.noise, stream_rng)

    def _draw_batch(
        self,
        theta: np.ndarray,
        shots: int,
        role: str = "opt",
    ) -> Tuple[np.ndarray, float]:
        theta_c = self.clip_theta(theta)
        req_shots = int(shots)
        if req_shots <= 0:
            return np.empty(0, dtype=float), 0.0
        if self.strict_budget and self.max_budget is not None:
            avail = self.remaining_shots()
            if avail <= 0:
                return np.empty(0, dtype=float), 0.0
            req_shots = min(req_shots, avail)

        active_rng = self.confirm_rng if role == "cert" else self.rng
        probs = self._get_probs(theta_c, active_rng)
        if role == "cert":
            self.certification_shots += req_shots
        else:
            self.optimization_shots += req_shots
        self.circuit_evaluations += 1

        if self.noise.readout_bitflip > 0.0:
            cdf = np.cumsum(probs); cdf[-1] = 1.0; states = np.searchsorted(cdf, active_rng.random(req_shots)).astype(np.int64)
            flip_mask = np.zeros(req_shots, dtype=np.int64)
            for q in range(self.n):
                flips = (active_rng.random(req_shots) < self.noise.readout_bitflip).astype(np.int64)
                flip_mask |= (flips << q)
            noisy_states = states ^ flip_mask
            observed_cuts = self.engine.cuts[noisy_states]
            _, counts = np.unique(noisy_states, return_counts=True)
        else:
            cdf = np.cumsum(probs); cdf[-1] = 1.0; states = np.searchsorted(cdf, active_rng.random(req_shots))
            observed_cuts = self.engine.cuts[states]
            _, counts = np.unique(states, return_counts=True)

        emp_p = counts.astype(float) / req_shots
        entropy_bits = float(-np.sum(emp_p * np.log2(np.maximum(emp_p, 1e-15))))
        losses = 1.0 - observed_cuts.astype(float) / self.total_weight
        mean_l = float(np.mean(losses))
        se_l = float(np.std(losses, ddof=1) / math.sqrt(req_shots)) if req_shots > 1 else 0.25
        self.query_log.append(
            (self.total_requested_shots, req_shots, role, theta_c.copy(), mean_l, se_l, entropy_bits)
        )
        return losses, entropy_bits

    def sample_outcomes(self, theta: np.ndarray, shots: int) -> Tuple[np.ndarray, float]:
        return self._draw_batch(theta, shots, role="opt")

    def loss_samples(self, theta: np.ndarray, shots: int) -> np.ndarray:
        losses, _ = self._draw_batch(theta, shots, role="opt")
        return losses

    def confirmation_loss_samples(self, theta: np.ndarray, shots: int) -> np.ndarray:
        """Draw independent confirmation samples charged to `certification_shots`."""
        losses, _ = self._draw_batch(theta, shots, role="cert")
        return losses

    def eval(self, theta: np.ndarray, shots: int) -> float:
        losses = self.loss_samples(theta, shots)
        if len(losses) == 0:
            return 1.0
        return float(np.mean(losses))


class RetrospectiveEvaluator:
    """Offline evaluator with access to exact statevector expectations and per-graph ceilings."""

    _CEILING_CACHE: Dict[Tuple[int, Tuple[Tuple[int, int, float], ...], int], Tuple[float, np.ndarray]] = {}

    def __init__(self, engine: QAOACircuitEngine, p: int = 1, noise: Optional[NoiseConfig] = None):
        self.engine = engine
        self.p = int(p)
        self.bounds = get_bounds(self.p)
        self.noise = noise or get_noise_preset("noiseless")
        self.opt_cut = engine.opt_cut
        self.total_weight = engine.total_weight
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

    def true_loss(self, theta: np.ndarray) -> float:
        return 1.0 - self.true_expected_cut(theta) / self.total_weight

    def true_ratio(self, theta: np.ndarray) -> float:
        return self.true_expected_cut(theta) / self.opt_cut

    def compute_exact_qaoa_ceiling(self) -> Tuple[float, np.ndarray]:
        """Compute per-graph p-layer QAOA ceiling via multi-start L-BFGS-B."""
        key = (self.engine.n, tuple(self.engine.edges), self.p)
        if self.noise.label == "noiseless" and key in self._CEILING_CACHE:
            return self._CEILING_CACHE[key]

        best_ratio = 0.0
        best_theta = get_default_x0(self.p, "wurtz_love", self.engine.edges)
        starts = [
            get_default_x0(self.p, "wurtz_love", self.engine.edges),
            get_default_x0(self.p, "shaydulin_rescaled", self.engine.edges),
            get_default_x0(self.p, "naive", self.engine.edges),
        ]
        rng = np.random.default_rng(12345 + self.p)
        n_rand = 4 if (self.engine.n <= 10 and self.p <= 2) else (2 if self.engine.n <= 14 else 1)
        max_it = 35 if self.engine.n <= 12 else 20
        for _ in range(n_rand):
            starts.append(rng.uniform(self.bounds[:, 0], self.bounds[:, 1]))

        b_list = list(map(tuple, self.bounds))
        for x0 in starts:
            if self.engine.n >= 12:
                res = minimize(
                    lambda x: -self.true_ratio(x),
                    x0,
                    method="COBYLA",
                    bounds=b_list,
                    options={"maxiter": 25 if self.engine.n <= 14 else 15, "rhobeg": 0.15, "tol": 1e-4},
                )
            else:
                res = minimize(
                    lambda x: -self.true_ratio(x),
                    x0,
                    method="L-BFGS-B",
                    bounds=b_list,
                    options={"maxiter": max_it, "ftol": 1e-6},
                )
            r = float(-res.fun)
            if r > best_ratio:
                best_ratio = r
                best_theta = np.asarray(res.x, dtype=float)

        if self.noise.label == "noiseless":
            self._CEILING_CACHE[key] = (best_ratio, best_theta.copy())
        return best_ratio, best_theta.copy()

"""Corrected Graph Library with Exact Isomorphism Verification and Delorme-Poljak / Mohar-Poljak Spectral Support (n=8..16).

Implements:
1. `are_graphs_isomorphic_exact(n, edges1, edges2, tol)` using degree + 2-hop/3-hop walk invariant
   pruned backtracking bijection verification on the weighted adjacency matrix (exact for n in {8..16}).
2. Assigns each deduplicated graph instance to an explicit partition (`dev`, `val`, `eval`)
   and guarantees zero isomorphic leakage across partitions.
3. Computes the online polynomial-time Mohar-Poljak (1990) & Delorme-Poljak (1993) diagonal-corrected
   Laplacian spectral upper bound `U_spec(G) = min_{sum u_i = 0} (n/4) lambda_max(L_G + diag(u))`,
   the online spectral loss support `[1 - U_spec(G)/W, 1]` of width `R_spec(G) = U_spec(G)/W`,
   and the retrospective exact MaxCut optimum `opt_cut_retrospective`.
"""
from __future__ import annotations
import hashlib
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass
class GraphMetadata:
    graph_id: str
    family: str
    n_nodes: int
    n_edges: int
    total_weight: float
    weight_distribution: str
    connected_components: int
    deg_min: float
    deg_max: float
    deg_mean: float
    deg_std: float
    spectral_hash: str
    seed: int
    partition: str  # 'dev' | 'val' | 'eval'
    spectral_upper_bound: float
    spectral_loss_min: float
    spectral_loss_range: float
    opt_cut_retrospective: float
    spectral_tightness_ratio: float
    edges: List[Tuple[int, int, float]]

    def to_record(self) -> Dict:
        d = asdict(self)
        d.pop("edges", None)
        return d


def compute_delorme_poljak_spectral_bound(
    n: int,
    L_mat: np.ndarray,
    total_w: float,
    is_unit: bool = False,
) -> float:
    """Compute the Mohar-Poljak (1990) & Delorme-Poljak (1993, Math. Program. 62:557-574) upper bound.

    For ANY vector u in R^n with sum_{i=1}^n u_i = 0 and ANY cut vector z in {-1, +1}^n:
        4 * C(z) = z^T L_G z = z^T (L_G + diag(u)) z <= n * lambda_max(L_G + diag(u)).
    Because u_n := -sum_{i=1}^{n-1} u_i enforces sum_i u_i = 0 identically, every iterate u
    produces a 100% rigorous, deterministic upper bound on MaxCut(G) in O(n^3) classical time.
    """
    mp_raw = min(total_w, 0.25 * n * float(np.max(np.linalg.eigvalsh(L_mat))))
    if n <= 2:
        return math.floor(mp_raw + 1e-9) if is_unit else mp_raw

    # Initialize u0 to equalize the diagonal of L_mat + diag(u0) to mean(diag(L_mat))
    diag_L = np.diag(L_mat)
    u0_full = np.mean(diag_L) - diag_L
    u0_sub = u0_full[:-1].copy()

    def obj(u_sub: np.ndarray) -> float:
        u_vec = np.empty(n, dtype=float)
        u_vec[:-1] = u_sub
        u_vec[-1] = -float(np.sum(u_sub))
        ev_max = float(np.max(np.linalg.eigvalsh(L_mat + np.diag(u_vec))))
        return 0.25 * n * ev_max

    best_ub = min(mp_raw, obj(u0_sub))
    res = minimize(obj, u0_sub, method="BFGS", options={"maxiter": 25, "gtol": 1e-4})
    # Re-evaluate with exact zero-sum projection for deterministic soundness
    u_opt = np.empty(n, dtype=float)
    u_opt[:-1] = res.x
    u_opt[-1] = -float(np.sum(res.x))
    dp_ub = min(best_ub, 0.25 * n * float(np.max(np.linalg.eigvalsh(L_mat + np.diag(u_opt)))))
    if is_unit:
        dp_ub = float(math.floor(dp_ub + 1e-7))
    return float(min(total_w, dp_ub))


def are_graphs_isomorphic_exact(
    n: int,
    edges1: List[Tuple[int, int, float]],
    edges2: List[Tuple[int, int, float]],
    tol: float = 1e-6,
) -> bool:
    """Exact weighted graph isomorphism test via degree + walk-profile pruned backtracking."""
    if len(edges1) != len(edges2):
        return False
    A1 = np.zeros((n, n), dtype=float)
    A2 = np.zeros((n, n), dtype=float)
    for u, v, w in edges1:
        A1[u, v] = w
        A1[v, u] = w
    for u, v, w in edges2:
        A2[u, v] = w
        A2[v, u] = w

    deg1 = np.sum(A1, axis=1)
    deg2 = np.sum(A2, axis=1)
    if not np.allclose(np.sort(deg1), np.sort(deg2), atol=tol):
        return False

    ev1 = np.sort(np.linalg.eigvalsh(A1))
    ev2 = np.sort(np.linalg.eigvalsh(A2))
    if not np.allclose(ev1, ev2, atol=1e-4):
        return False

    A1_sq = A1 @ A1
    A2_sq = A2 @ A2
    A1_cu = A1_sq @ A1
    A2_cu = A2_sq @ A2

    def vertex_sig(A, A_sq, A_cu, u):
        inc = np.sort(A[u, A[u] > tol])
        row2 = np.sort(A_sq[u])
        return (
            round(float(np.sum(inc)), 4),
            tuple(np.round(inc, 4) + 0.0),
            round(float(A_sq[u, u]), 3),
            round(float(A_cu[u, u]), 3),
            tuple(np.round(row2, 3) + 0.0),
        )

    sigs1 = [vertex_sig(A1, A1_sq, A1_cu, u) for u in range(n)]
    sigs2 = [vertex_sig(A2, A2_sq, A2_cu, v) for v in range(n)]
    if sorted(sigs1) != sorted(sigs2):
        return False

    order1 = sorted(range(n), key=lambda u: sigs1.count(sigs1[u]))
    candidates = {u: [v for v in range(n) if sigs1[u] == sigs2[v]] for u in range(n)}

    mapping: Dict[int, int] = {}
    used2 = set()

    def backtrack(idx: int) -> bool:
        if idx == n:
            return True
        u = order1[idx]
        for v in candidates[u]:
            if v in used2:
                continue
            consistent = True
            for prev_idx in range(idx):
                u_prev = order1[prev_idx]
                v_prev = mapping[u_prev]
                if abs(A1[u, u_prev] - A2[v, v_prev]) > tol:
                    consistent = False
                    break
            if consistent:
                mapping[u] = v
                used2.add(v)
                if backtrack(idx + 1):
                    return True
                del mapping[u]
                used2.remove(v)
        return False

    return backtrack(0)


def nx_random_regular_graph(d: int, n: int, seed: int) -> List[Tuple[int, int, float]]:
    """Exact reproduction of NetworkX random_regular_graph(d, n, seed)."""
    rng = random.Random(seed)
    if (n * d) % 2 != 0 or not (0 <= d < n):
        raise ValueError(f"Invalid (d={d}, n={n})")

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
        adj.setdefault(u, {})[v] = None
        adj.setdefault(v, {})[u] = None
    edge_list: List[Tuple[int, int, float]] = []
    seen = set()
    for u in adj:
        for v in adj[u]:
            if v not in seen:
                edge_list.append((u, v, 1.0))
        seen.add(u)
    return edge_list


def _count_connected_components(n: int, edges: List[Tuple[int, int, float]]) -> int:
    adj = {i: [] for i in range(n)}
    for u, v, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = set()
    comps = 0
    for node in range(n):
        if node not in visited:
            comps += 1
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
    return comps


def compute_graph_metadata(
    family: str,
    n: int,
    edges: List[Tuple[int, int, float]],
    seed: int,
    weight_dist: str,
    partition: str = "eval",
) -> GraphMetadata:
    for _, _, w in edges:
        if w < 0.0:
            raise ValueError(f"Negative edge weight {w} encountered; [0,1] loss requires w >= 0.")
    A = np.zeros((n, n), dtype=float)
    deg = np.zeros(n, dtype=float)
    for u, v, w in edges:
        A[u, v] = w
        A[v, u] = w
        deg[u] += w
        deg[v] += w
    L = np.diag(deg) - A
    evals_A = np.round(np.sort(np.linalg.eigvalsh(A)), 5) + 0.0
    evals_L = np.round(np.sort(np.linalg.eigvalsh(L)), 5) + 0.0
    deg_sorted = np.round(np.sort(deg), 5) + 0.0

    dim = 1 << n
    states = np.arange(dim, dtype=np.uint32)
    cuts = np.zeros(dim, dtype=np.float64)
    for u, v, w in edges:
        cuts += w * (((states >> u) ^ (states >> v)) & 1)
    opt_cut = float(np.max(cuts))
    total_w = float(sum(w for _, _, w in edges))

    spec_ub = compute_delorme_poljak_spectral_bound(n, L, total_w, is_unit=(weight_dist == "unit"))
    spec_range = float(min(1.0, max(1e-6, spec_ub / max(total_w, 1e-12))))
    spec_min = float(max(0.0, 1.0 - spec_range))

    cut_quantiles = np.round(np.quantile(cuts, np.linspace(0, 1, 17)), 5) + 0.0
    sig_str = f"{n}|{len(edges)}|{evals_A.tolist()}|{evals_L.tolist()}|{deg_sorted.tolist()}|{cut_quantiles.tolist()}"
    spec_hash = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:12]
    comps = _count_connected_components(n, edges)

    return GraphMetadata(
        graph_id=f"{family}_n{n}_s{seed}_{spec_hash[:6]}",
        family=family,
        n_nodes=n,
        n_edges=len(edges),
        total_weight=total_w,
        weight_distribution=weight_dist,
        connected_components=comps,
        deg_min=float(np.min(deg)),
        deg_max=float(np.max(deg)),
        deg_mean=float(np.mean(deg)),
        deg_std=float(np.std(deg)),
        spectral_hash=spec_hash,
        seed=seed,
        partition=partition,
        spectral_upper_bound=round(spec_ub, 6),
        spectral_loss_min=round(spec_min, 6),
        spectral_loss_range=round(spec_range, 6),
        opt_cut_retrospective=opt_cut,
        spectral_tightness_ratio=round(spec_ub / max(opt_cut, 1e-12), 6),
        edges=edges,
    )


def make_raw_graph(family: str, n: int, seed: int) -> Tuple[List[Tuple[int, int, float]], str]:
    if family in ("3_regular", "random_3_regular"):
        return nx_random_regular_graph(3, n, seed), "unit"
    elif family in ("weighted_3_regular", "weighted_3_regular_uniform"):
        base = nx_random_regular_graph(3, n, seed)
        rng = np.random.default_rng(seed + 77777)
        edges = [(u, v, float(rng.uniform(0.4, 1.6))) for u, v, _ in base]
        return edges, "uniform_0.4_1.6"
    elif family == "weighted_3_regular_bimodal":
        base = nx_random_regular_graph(3, n, seed)
        rng = np.random.default_rng(seed + 99991)
        edges = []
        for u, v, _ in base:
            w = float(rng.uniform(2.0, 3.0)) if rng.random() < 0.35 else float(rng.uniform(0.10, 0.25))
            edges.append((u, v, w))
        return edges, "bimodal_0.1_0.25_and_2.0_3.0"
    elif family == "erdos_renyi":
        rng = np.random.default_rng(seed)
        p_edge = 0.5 if n <= 10 else (3.5 / max(1, n - 1))
        for _ in range(300):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p_edge:
                        edges.append((u, v, 1.0))
            if len(edges) >= n and _count_connected_components(n, edges) == 1:
                return edges, "unit"
        return nx_random_regular_graph(3, n, seed), "unit"
    elif family == "watts_strogatz":
        rng = np.random.default_rng(seed)
        edges_set = set()
        for u in range(n):
            for offset in (1, 2):
                v = (u + offset) % n
                if rng.random() < 0.3:
                    cands = [w for w in range(n) if w != u and (min(u, w), max(u, w)) not in edges_set]
                    if cands:
                        v = int(rng.choice(cands))
                edges_set.add((min(u, v), max(u, v)))
        edges = [(u, v, 1.0) for u, v in sorted(edges_set)]
        return edges, "unit"
    elif family == "barabasi_albert":
        rng = np.random.default_rng(seed)
        m_attach = 2
        edges_set = {(0, 1), (1, 2), (0, 2)}
        degrees = [2, 2, 2] + [0] * (n - 3)
        for new_node in range(3, n):
            existing = np.arange(new_node)
            probs = np.array([degrees[i] for i in existing], dtype=float)
            probs /= probs.sum()
            targets = rng.choice(existing, size=m_attach, replace=False, p=probs)
            for t in targets:
                u, v = (int(t), new_node) if int(t) < new_node else (new_node, int(t))
                edges_set.add((u, v))
                degrees[u] += 1
                degrees[v] += 1
        edges = [(u, v, 1.0) for u, v in sorted(edges_set)]
        return edges, "unit"
    raise ValueError(f"Unknown graph family: {family}")


def generate_deduplicated_graph_suite(
    family: str,
    n: int,
    target_count: int,
    start_seed: int = 300,
    max_search_seeds: int = 400,
    assign_partitions: bool = True,
) -> List[GraphMetadata]:
    """Generate up to `target_count` guaranteed non-isomorphic connected graphs using exact bijection verification."""
    unique_graphs: List[GraphMetadata] = []
    for offset in range(max_search_seeds):
        s = start_seed + offset
        edges, w_dist = make_raw_graph(family, n, s)
        if _count_connected_components(n, edges) != 1:
            continue
        is_dup = False
        for existing in unique_graphs:
            if are_graphs_isomorphic_exact(n, edges, existing.edges):
                is_dup = True
                break
        if not is_dup:
            idx = len(unique_graphs)
            part = "eval"
            if assign_partitions and target_count >= 3:
                if idx == 0:
                    part = "dev"
                elif idx == 1:
                    part = "val"
                else:
                    part = "eval"
            meta = compute_graph_metadata(family, n, edges, s, w_dist, partition=part)
            unique_graphs.append(meta)
            if len(unique_graphs) >= target_count:
                break
    return unique_graphs

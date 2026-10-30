"""Deduplicated graph instance generator and classical upper-bound calculator (Workstreams 1.F & 5).

Supports six structurally diverse graph families with canonical spectral/degree/cut hashing
to guarantee non-isomorphic graph deduplication:
1. `random_3_regular`: Unweighted 3-regular graphs (Kim-Vu stub matching, deduplicated).
2. `weighted_3_regular_uniform`: 3-regular topology with uniform weights w_uv in [0.4, 1.6].
3. `weighted_3_regular_bimodal`: Deliberately difficult bimodal/heavy-tailed weights w_uv in [0.1, 0.25] U [2.0, 3.0]
   where unweighted fixed angles and simple RMS rescaling fail (counterexample regime).
4. `erdos_renyi`: Connected Erdos-Renyi G(n, p=0.5) irregular graphs.
5. `watts_strogatz`: Connected Watts-Strogatz small-world graphs (k=4, p=0.3).
6. `barabasi_albert`: Connected Barabasi-Albert preferential-attachment graphs (m=2, hub topology).

Also computes the polynomial-time Mohar-Poljak spectral Laplacian upper bound:
  U_spec(G) = min(W_tot, (n / 4) * lambda_max(L_G))
for hardware-realistic operational target normalization when the NP-hard optimum C_max is unknown.
"""
from __future__ import annotations
import hashlib
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np


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
    spectral_upper_bound: float
    opt_cut_retrospective: float
    edges: List[Tuple[int, int, float]]

    def to_record(self) -> Dict:
        d = asdict(self)
        d.pop("edges", None)
        return d


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
) -> GraphMetadata:
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
    states = np.arange(dim, dtype=np.uint64)
    cuts = np.zeros(dim, dtype=np.float64)
    for u, v, w in edges:
        cuts += w * (((states >> u) ^ (states >> v)) & 1)
    opt_cut = float(np.max(cuts))
    total_w = float(sum(w for _, _, w in edges))

    # Mohar-Poljak (1990) spectral Laplacian upper bound: MaxCut(G) <= (n / 4) * lambda_max(L)
    lambda_max_L = float(np.max(evals_L))
    spec_ub = min(total_w, 0.25 * n * lambda_max_L)
    if weight_dist == "unit":
        spec_ub = float(math.floor(spec_ub + 1e-9))

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
        spectral_upper_bound=spec_ub,
        opt_cut_retrospective=opt_cut,
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
            if rng.random() < 0.35:
                w = float(rng.uniform(2.0, 3.0))  # Heavy dominant edges
            else:
                w = float(rng.uniform(0.10, 0.25))  # Weak edges
            edges.append((u, v, w))
        return edges, "bimodal_0.1_0.25_and_2.0_3.0"
    elif family == "erdos_renyi":
        rng = np.random.default_rng(seed)
        for _ in range(300):
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.5:
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
        # Start with star/triangle on m_attach + 1 = 3 nodes
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
) -> List[GraphMetadata]:
    """Generate up to `target_count` guaranteed non-isomorphic connected graphs."""
    unique_graphs: List[GraphMetadata] = []
    seen_hashes = set()
    for offset in range(max_search_seeds):
        s = start_seed + offset
        edges, w_dist = make_raw_graph(family, n, s)
        meta = compute_graph_metadata(family, n, edges, s, w_dist)
        if meta.connected_components == 1 and meta.spectral_hash not in seen_hashes:
            seen_hashes.add(meta.spectral_hash)
            unique_graphs.append(meta)
            if len(unique_graphs) >= target_count:
                break
    return unique_graphs
